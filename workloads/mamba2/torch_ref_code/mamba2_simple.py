# Copyright (c) 2024, Tri Dao, Albert Gu.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn = None

class RMSNormGated(torch.nn.Module):

    def __init__(self, hidden_size, eps=1e-5, group_size=None, norm_before_gate=True, device=None, dtype=None):
        """If group_size is not None, we do GroupNorm with each group having group_size elements.
        group_size=None is equivalent to group_size=hidden_size (i.e. there's only 1 group).
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        self.register_parameter("bias", None)
        self.group_size = group_size
        self.norm_before_gate = norm_before_gate
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)

    def forward(self, x, z=None):
        """If z is not None, we do norm(x) * silu(z) if norm_before_gate, else norm(x * silu(z))
        """
        pass

# CPU implementation of mamba_chunk_scan_combined
def mamba_chunk_scan_combined(
    x,       # (B, L, H, P)
    dt,      # (B, L, H)
    A,       # (H,)
    B_,      # (B, L, G, N)
    C_,      # (B, L, G, N)
    ngroups: int,
    chunk_size=256,
    D=None,
    z=None, seq_idx=None,
    initial_states=None
):
    Bsz, L, H, P = x.shape
    N = B_.shape[-1]   # state dim d_state
    device = x.device
    dtype = x.dtype
    y = torch.zeros_like(x)

    heads_per_group = H // ngroups

    for b in range(Bsz):
        for h in range(H):
            if initial_states is not None:
                state = initial_states[b, h].to(device=device, dtype=dtype)
            else:
                state = torch.zeros(P, N, device=device, dtype=dtype)

            a_h = A[h]
            g = h // heads_per_group   # map head to group

            for l in range(L):
                dt_ = dt[b, l, h]
                state = torch.exp(dt_ * a_h) * state + B_[b, l, g] * x[b, l, h].unsqueeze(-1)
                y[b, l, h] = (C_[b, l, g] * state).sum(-1)
                if D is not None:
                    y[b, l, h] += D[h] * x[b, l, h]
    return y

# CPU implementation of mamba_split_conv1d_scan_combined
def mamba_split_conv1d_scan_combined(
    zxbcdt: torch.Tensor,       # (B, L, d_proj)
    conv1d_weight: torch.Tensor,# (d_conv, kernel_size)
    conv1d_bias: torch.Tensor,  # (d_conv,)
    dt_bias: torch.Tensor,      # (nheads,)
    A: torch.Tensor,            # (nheads,)
    D: torch.Tensor = None,     # (nheads,)
    d_inner: int = None,        # width of z and x split
    d_state: int = None,        # state dim for B and C
    headdim: int = None,        # usually d_inner // nheads
    ngroups: int = 1,
    chunk_size: int = 256,
    seq_idx=None,
    activation: str = "swish",
    rmsnorm_weight: torch.Tensor = None,
    rmsnorm_eps: float = 1e-5,
    outproj_weight: torch.Tensor = None,
    outproj_bias: torch.Tensor = None,
    norm_before_gate: bool = False,
    initial_states: torch.Tensor = None, # (B, nheads, headdim, d_state)
    dt_limit = (0.0, float("inf")),
):
    """
    CPU fallback for Mamba2 fused kernel.

    Args:
        zxbcdt: (B, L, d_proj) concatenated [z, x, B, C, dt]
        conv1d_weight: (d_conv, kernel_size) for depthwise conv
        conv1d_bias: (d_conv,)
        dt_bias: (nheads,)
        A: (nheads,)
        D: optional (nheads,)
        d_inner: size of z and x each
        d_state: state dimension for B and C
        headdim: usually = d_inner // nheads
        ngroups: groups for state projection
        mamba_chunk_scan_combined: CPU scan function
    Returns:
        out: (B, L, d_model)
    """
    Bsz, L, d_proj = zxbcdt.shape
    nheads = A.shape[0]
    assert dt_bias.shape[0] == nheads, "dt_bias mismatch"

    # Infer dims if not given
    if d_inner is None or d_state is None or headdim is None:
        # From formula: d_proj = 2*d_inner + 2*ngroups*d_state + nheads
        missing = d_proj - nheads
        # This is underdetermined, in real model code these are known
        # Usually headdim is known constant, so we assume it's provided
        raise ValueError("Must pass d_inner, d_state, headdim for Mamba2 CPU fallback")

    # Validate shape
    expected_d_proj = 2*d_inner + 2*ngroups*d_state + nheads
    assert d_proj == expected_d_proj, \
        f"d_proj mismatch: got {d_proj}, expected {expected_d_proj}"

    # ---- Split ----
    z, x, B_part, C_part, dt = torch.split(
        zxbcdt,
        [d_inner, d_inner, ngroups*d_state, ngroups*d_state, nheads],
        dim=-1
    )

    # ---- Process dt ----
    dt = torch.clamp(F.softplus(dt + dt_bias), dt_limit[0], dt_limit[1])  # (B, L, nheads)

    # ---- Depthwise conv on x, B, C combined ----
    xBC = torch.cat([x, B_part, C_part], dim=-1)  # (B, L, d_inner + 2*ngroups*d_state)
    xBC_conv_in = xBC.transpose(1, 2)  # (B, Cin, L)
    weight = conv1d_weight.unsqueeze(1)  # (Cin, 1, k)
    xBC_conv = F.conv1d(
        xBC_conv_in,
        weight=weight,
        bias=conv1d_bias,
        padding=conv1d_weight.shape[-1] - 1,
        groups=xBC_conv_in.shape[1],
    ).transpose(1, 2)[:, :L, :]  # back to (B, L, Cin) and trim to L

    # ---- Activation ----
    if activation in ("swish", "silu"):
        xBC_conv = F.silu(xBC_conv)
    else:
        raise NotImplementedError(f"Activation {activation} not supported")

    # ---- Split back into x, B, C after conv ----
    x_conv, B_conv, C_conv = torch.split(
        xBC_conv,
        [d_inner, ngroups*d_state, ngroups*d_state],
        dim=-1
    )

    # ---- Reshape for scan ----
    x_scan = rearrange(x_conv, "b l (h p) -> b l h p", h=nheads)          # (B,L,H,headdim)
    B_scan = rearrange(B_conv, "b l (g n) -> b l g n", g=ngroups)         # (B,L,ngroups,d_state)
    C_scan = rearrange(C_conv, "b l (g n) -> b l g n", g=ngroups)         # same

    # ---- Scan ----
    y_scan = mamba_chunk_scan_combined(
        x_scan,
        dt,
        A,
        B_scan,
        C_scan,
        ngroups = ngroups,
        chunk_size=chunk_size,
        D=D,
        z=None,
        seq_idx=seq_idx,
        initial_states=initial_states,
    )  # (B, L, H, headdim)

    y_flat = rearrange(y_scan, "b l h p -> b l (h p)")  # (B,L,d_inner)

    # ---- RMSNorm + gate ----
    if rmsnorm_weight is not None:
        mu = y_flat.pow(2).mean(-1, keepdim=True)
        y_flat = y_flat * torch.rsqrt(mu + rmsnorm_eps) * rmsnorm_weight

    y_flat = y_flat * torch.sigmoid(z)  # gate

    # ---- Out projection ----
    if outproj_weight is not None:
        out = F.linear(y_flat, outproj_weight, outproj_bias)
    else:
        out = y_flat

    return out


class Mamba2Simple(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=64,
        d_conv=4,
        conv_init=None,
        expand=2,
        headdim=128,
        ngroups=1,
        A_init_range=(1, 16),
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        learnable_init_states=False,
        activation="swish",
        bias=False,
        conv_bias=True,
        # Fused kernel and sharding options
        chunk_size=256,
        use_mem_eff_path=True,
        layer_idx=None,  # Absorb kwarg for general module
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.conv_init = conv_init
        self.expand = expand
        self.d_inner = self.expand * self.d_model
        self.headdim = headdim
        self.ngroups = ngroups
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim
        self.dt_limit = dt_limit
        self.learnable_init_states = learnable_init_states
        self.activation = activation
        self.chunk_size = chunk_size
        self.use_mem_eff_path = use_mem_eff_path
        self.layer_idx = layer_idx

        # Order: [z, x, B, C, dt]
        
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)

        conv_dim = self.d_inner + 2 * self.ngroups * self.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        if self.conv_init is not None:
            nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)
        # self.conv1d.weight._no_weight_decay = True

        if self.learnable_init_states:
            self.init_states = nn.Parameter(torch.zeros(self.nheads, self.headdim, self.d_state, **factory_kwargs))
            self.init_states._no_weight_decay = True

        self.act = nn.SiLU()

        # Initialize log dt bias
        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt_bias._no_weight_decay = True

        # A parameter
        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        A_log = torch.log(A).to(dtype=dtype)
        self.A_log = nn.Parameter(A_log)
        # self.register_buffer("A_log", torch.zeros(self.nheads, dtype=torch.float32, device=device), persistent=True)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.nheads, device=device))
        self.D._no_weight_decay = True

        # Extra normalization layer right before output projection
        #assert RMSNormGated is not None
        self.norm = RMSNormGated(self.d_inner, eps=1e-5, norm_before_gate=False, **factory_kwargs)

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, u, seq_idx=None):
        """
        u: (B, L, D)
        Returns: same shape as u
        """
        batch, seqlen, dim = u.shape

        zxbcdt = self.in_proj(u)  # (B, L, d_in_proj)
        A = -torch.exp(self.A_log)  # (nheads) or (d_inner, d_state)
        initial_states=repeat(self.init_states, "... -> b ...", b=batch) if self.learnable_init_states else None
        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)

        if self.use_mem_eff_path:
            # Fully fused path
            out = mamba_split_conv1d_scan_combined(
                zxbcdt,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.dt_bias,
                A,
                D=self.D,
                d_inner=self.d_inner,
                d_state=self.d_state,
                headdim=self.headdim,
                ngroups=self.ngroups,
                chunk_size=self.chunk_size,
                seq_idx=seq_idx,
                activation=self.activation,
                rmsnorm_weight=self.norm.weight,
                rmsnorm_eps=self.norm.eps,
                outproj_weight=self.out_proj.weight,
                outproj_bias=self.out_proj.bias,
                norm_before_gate=False,
                initial_states=initial_states,
                **dt_limit_kwargs,
            )
        else:
            z, xBC, dt = torch.split(
                zxbcdt, [self.d_inner, self.d_inner + 2 * self.ngroups * self.d_state, self.nheads], dim=-1
            )
            dt = F.softplus(dt + self.dt_bias)  # (B, L, nheads)
            assert self.activation in ["silu", "swish"]

            # 1D Convolution
            if causal_conv1d_fn is None or self.activation not in ["silu", "swish"]:
                xBC = self.act(
                    self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)
                )  # (B, L, self.d_inner + 2 * ngroups * d_state)
                xBC = xBC[:, :seqlen, :]
            else:
                xBC = causal_conv1d_fn(
                    x=xBC.transpose(1, 2),
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                ).transpose(1, 2)

            # Split into 3 main branches: X, B, C
            # These correspond to V, K, Q respectively in the SSM/attention duality
            x, B, C = torch.split(xBC, [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
            y = mamba_chunk_scan_combined(
                rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
                dt,
                A,
                rearrange(B, "b l (g n) -> b l g n", g=self.ngroups),
                rearrange(C, "b l (g n) -> b l g n", g=self.ngroups),
                chunk_size=self.chunk_size,
                D=self.D,
                z=None,
                seq_idx=seq_idx,
                initial_states=initial_states,
                **dt_limit_kwargs,
            )
            y = rearrange(y, "b l h p -> b l (h p)")

            # Multiply "gate" branch and apply extra normalization layer
            y = self.norm(y, z)
            out = self.out_proj(y)
        return out
