import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import ttsim.front.functional.sim_ttnn as ttnn

class Conv:
    def __init__(
        self,
        conv_params,
        input_shape,
        parameters,
        *,
        act_block_h=None,
        reshard=False,
        deallocate=True,
        height_sharding=True,
        activation="",
        groups=1,
        dtype=ttnn.bfloat16,
    ) -> None:
        self.weights = parameters["weight"]
        if "bias" in parameters:
            self.bias = parameters["bias"]
        else:
            self.bias = None
        self.kernel_size = (self.weights.shape[2], self.weights.shape[3])
        self.conv_params = conv_params
        self.out_channels = self.weights.shape[0]
        self.act_block_h = act_block_h
        self.reshard = reshard
        self.deallocate = deallocate
        self.activation = activation
        self.groups = groups
        self.dtype = dtype
        self.shard_layout = (
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED if height_sharding else ttnn.TensorMemoryLayout.BLOCK_SHARDED
        )
        self.input_shape = input_shape

    def __call__(self, device, input_tensor):

        conv_config = ttnn.Conv2dConfig(
            weights_dtype=ttnn.bfloat16,
            math_fidelity=ttnn.MathFidelity.LoFi,
            activation=self.activation,
            shard_layout=self.shard_layout,
            fp32_dest_acc_enabled=False,
            packer_l1_accum_enabled=False,
            deallocate_activation=self.deallocate,
        )
        if self.act_block_h is not None:
            conv_config.act_block_h_override = self.act_block_h

        [output_tensor, _out_height, _out_width, self.weights, self.bias] = ttnn.conv2d(
            input_tensor=input_tensor,
            weight_tensor=self.weights,
            bias_tensor=self.bias,
            in_channels=self.input_shape[3],
            out_channels=self.out_channels,
            device=device,
            kernel_size=self.kernel_size,
            stride=(self.conv_params[0], self.conv_params[1]),
            padding=(self.conv_params[2], self.conv_params[3]),
            batch_size=self.input_shape[0],
            input_height=self.input_shape[1],
            input_width=self.input_shape[2],
            conv_config=conv_config,
            groups=self.groups,
            return_output_size=True,
            return_prepared_device_weights=True,
            dtype=self.dtype,
        )

        return output_tensor


class TTNNBasicBlock:

    def __init__(
        self,
        parameters,
    ) -> None:
        self.conv1 = Conv([1, 1, 1, 1], [8, 56, 56, 64], parameters=parameters["conv1"])
        self.conv2 = Conv([1, 1, 1, 1], [8, 56, 56, 64], parameters=parameters["conv2"])
        if "downsample" in parameters:
            self.downsample = parameters.downsample
        else:
            self.downsample = None

    def __call__(self, x, device):
        identity = x
        out = self.conv1(device, x)
        out = ttnn.relu(out)
        out = self.conv2(device, out)

        if self.downsample is not None:
            identity = self.downsample(x)
        
        out = ttnn.add(out, identity)
        out = ttnn.relu(out)
        return out

def run_model(model, torch_input_tensor, device):
    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)

    output_tensor = model(input_tensor, device)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)
    return output_tensor

def run_experiment():
    weight_data = torch.randn((64, 64, 3, 3), dtype=torch.float32)
    bias_data = torch.randn((1, 1, 1, 64), dtype=torch.float32)
    parameters =     {
        "conv1": {
            "weight": ttnn.Tensor(weight_data),
            "bias": ttnn.Tensor(bias_data)
        },
        "conv2": {
            "weight": ttnn.Tensor(weight_data),
            "bias": ttnn.Tensor(bias_data)
        }
    }

    device = ttnn.open_device(device_id=0, device_type="ttnn", device_name=None)
    ttnn_model = TTNNBasicBlock(parameters)
    torch_input_tensor = torch.rand((8, 64, 56, 56), dtype=torch.float32)
    output_tensor = run_model(ttnn_model, torch_input_tensor, device=device)
    ttnn.close_device(device)

if __name__ == "__main__":
    run_experiment()
