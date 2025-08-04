#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from .op import MathFidelity
from .tensor import DataType, Layout
from .buffer import TensorMemoryLayout
from .device import ARCH

from dataclasses import dataclass

@dataclass
class Conv2dConfig:
    weights_dtype                   : DataType
    activation                      : str
    shard_layout                    : TensorMemoryLayout
    deallocate_activation           : bool = False
    reallocate_halo_output          : bool = True
    reshard_if_not_optimal          : bool = False
    override_sharding_config        : bool = False
    transpose_shards                : bool = False
    enable_act_double_buffer        : bool = False
    enable_weights_double_buffer    : bool = False
    enable_split_reader             : bool = False
    enable_subblock_padding         : bool = False
    in_place                        : bool = False
    enable_kernel_stride_folding    : bool = False
    act_block_h_override            : int  = 0
    act_block_w_div                 : int  = 1
    output_layout                   : Layout = Layout.TILE_LAYOUT
    #core_grid                       : CoreRangeSet = null

@dataclass
class GrayskullComputeKernelConfig:
    math_fidelity    : MathFidelity = MathFidelity.LoFi
    math_approx_mode : bool = True
    dst_full_sync_en : bool = False

@dataclass
class WormholeComputeKernelConfig:
    math_fidelity    : MathFidelity = MathFidelity.LoFi
    math_approx_mode : bool         = True
    dst_full_sync_en : bool         = False
    fp32_dest_acc_en : bool         = False
    packer_l1_acc    : bool         = False
    #throttle_level   : ThrottleLevel = NO_THROTTLE

BlackholeComputeKernelConfig = WormholeComputeKernelConfig

#Compute Configs
def init_device_compute_kernel_config(
        dev_arch,
        math_fidelity,
        math_approx_mode = True,
        fp32_dest_acc_en = False,
        packer_l1_acc    = False,
        ):
    return {
            ARCH.GRAYSKULL  : GrayskullComputeKernelConfig(math_fidelity, math_approx_mode),
            ARCH.WORMHOLE_B0: WormholeComputeKernelConfig(math_fidelity,
                                                          math_approx_mode,
                                                          fp32_dest_acc_en,
                                                          packer_l1_acc),
            ARCH.BLACKHOLE  : BlackholeComputeKernelConfig(math_fidelity,
                                                           math_approx_mode,
                                                           fp32_dest_acc_en,
                                                           packer_l1_acc),
            }[dev_arch]

