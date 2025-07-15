import torch
import numpy as np
import ttsim.front.functional.op as F
from ttsim.ops import SimOpFactory, SimTensor
import polaris

TILE_LAYOUT = 0
L1_MEMORY_CONFIG = 1
float32 = np.float32
bfloat16 = np.float16
uint32 = np.uint32
DRAM_MEMORY_CONFIG = 2
ROW_MAJOR_LAYOUT = 0

class TensorMemoryLayout:
    HEIGHT_SHARDED = 1
    BLOCK_SHARDED = 2

class MathFidelity:
    HiFi4 = "HiFi4"
    HiFi3 = "HiFi3"
    HiFi2 = "HiFi2"
    LoFi = "LoFi"

class Conv2dConfig:
    def __init__(
        self,
        weights_dtype=None,
        math_fidelity=None,
        activation="",
        shard_layout=None,
        fp32_dest_acc_enabled=False,
        packer_l1_accum_enabled=False,
        deallocate_activation=True,
        act_block_h_override=None,
    ):
        self.weights_dtype = weights_dtype
        self.math_fidelity = math_fidelity
        self.activation = activation
        self.shard_layout = shard_layout
        self.fp32_dest_acc_enabled = fp32_dest_acc_enabled
        self.packer_l1_accum_enabled = packer_l1_accum_enabled
        self.deallocate_activation = deallocate_activation
        self.act_block_h_override = act_block_h_override

def open_device(device_id=0, device_type='ttnn', device_name=None):
    # empty function to match the interface
    return

def close_device(device_id=0, device_type='ttnn', device_name=None):
    # empty function to match the interface
    return

def full(shape, fill_value, dtype, layout=None, device=None, memory_config=None, name=None):
    # Create a SimTensor with the specified shape
    return SimTensor({
        'name': name,
        'shape': list(shape),
        'dtype': dtype})

def ones(shape, dtype, layout=None, device=None, memory_config=None, name=None):
    # Create a SimTensor with the specified shape
    return SimTensor({
        'name': name,
        'shape': list(shape),
        'dtype': dtype})

zeros = ones

def Tensor(data, device=None, layout=None, name=None):
    data_type = data.dtype
    return SimTensor({
        'name': name,
        'shape': list(data.shape),
        'dtype': data_type
        })

def embedding(input_ids, weight, device=None, name=None): ## this may need to be changed to F.embedding
    # Create a SimTensor for the embedding operation
    input_shape = input_ids.shape
    weight_shape = weight.shape
    data_type = weight.dtype

    # Create a SimTensor with the specified shape and dtype
    return SimTensor({
        'name': name,
        'shape': [input_shape[0], input_shape[1], weight_shape[1]],
        'dtype': data_type,
        'data': None  # Placeholder for actual data
        })

def get_dev_cfg(op_type):
    device_config = polaris.device_config['device_list']
    assert device_config is not None, 'Stand-alone simulation isn\'t supported! Exiting.'
    device = polaris.device_config['device_list'][0][0]
    dev_obj = polaris.device_config['devspec'][device]
    op2rsrc = polaris.device_config['op2rsrc']
    op2dt = polaris.device_config['op2dt']
    return [op2rsrc[op_type], op2dt, dev_obj]

def relu(x, name='Relu'):
    op_handl = F.Relu(name)
    op_type = op_handl.optype.upper()
    [op2rsrc, op2dt, dev_obj] = get_dev_cfg(op_type)
    d = polaris.execute_simophandle(op_handl, [x], dev_obj, op2rsrc, op2dt)
    return d

def linear(x, w, bias, core_grid=None, activation=None, memory_config=L1_MEMORY_CONFIG, dtype=bfloat16, name='Linear'): # to implementation activation impact
    M, N = w.shape
    op_handl = F.Linear(name, M, N)
    op_type = op_handl.optype.upper()
    [op2rsrc, op2dt, dev_obj] = get_dev_cfg(op_type)
    d = polaris.execute_simophandle(op_handl, [x], dev_obj, op2rsrc, op2dt)
    return d

def permute(x, dims, name='Permute'):
    op_handl = F.Permute(name, [x.shape[i] for i in dims])
    op_type = op_handl.optype.upper()
    [op2rsrc, op2dt, dev_obj] = get_dev_cfg(op_type)
    d = polaris.execute_simophandle(op_handl, [x], dev_obj, op2rsrc, op2dt)
    return d

def layer_norm(x, weight=None, bias=None, epsilon=1e-5, memory_config=None, compute_kernel_config=None, name='LayerNorm'):
    op_handl = F.LayerNorm(name, x.shape[-1])
    op_type = op_handl.optype.upper()
    [op2rsrc, op2dt, dev_obj] = get_dev_cfg(op_type)
    d = polaris.execute_simophandle(op_handl, [x], dev_obj, op2rsrc, op2dt)
    return d

def reshape(x, shape, name='Reshape'):
    op_handl = F.ReshapeFixed(name, shape)
    op_type = op_handl.optype.upper()
    [op2rsrc, op2dt, dev_obj] = get_dev_cfg(op_type)
    d = polaris.execute_simophandle(op_handl, [x], dev_obj, op2rsrc, op2dt)
    return d

def conv2d(input_tensor, weight_tensor, bias_tensor, in_channels, out_channels, device, kernel_size,
           stride=(1, 1), padding=(0, 0), batch_size=1, input_height=1, input_width=1,
           conv_config=None, groups=1,
           return_output_size=False, return_prepared_device_weights=False, dtype=None):
    op_handl = F.Conv2d(
        name='Conv2d',
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size[0],
        stride=stride[0],
        padding=padding[0],
        groups=groups,
        dtype=dtype
    )
    _out_height = input_height
    _out_width = input_width
    weights = weight_tensor
    bias = bias_tensor
    op_type = op_handl.optype.upper()
    [op2rsrc, op2dt, dev_obj] = get_dev_cfg(op_type)
    output_tensor = polaris.execute_simophandle(op_handl, [input_tensor], dev_obj, op2rsrc, op2dt)
    return [output_tensor, _out_height, _out_width, weights, bias]

# add kwargs to match extra arguments
def add(x, y, name='AddOp'):
    op_handl = F.Add(name, params = [(0,x)], ipos=[1])
    op_type = op_handl.optype.upper()
    [op2rsrc, op2dt, dev_obj] = get_dev_cfg(op_type)
    d = polaris.execute_simophandle(op_handl, [y], dev_obj, op2rsrc, op2dt)
    return d

def mul(x, y, name='MulOp'):
    op_handl = F.Mul(name, params = [(0,x)], ipos=[1])
    op_type = op_handl.optype.upper()
    [op2rsrc, op2dt, dev_obj] = get_dev_cfg(op_type)
    d = polaris.execute_simophandle(op_handl, [y], dev_obj, op2rsrc, op2dt)
    return d

def matmul(x, y, memory_config=DRAM_MEMORY_CONFIG, name='MatMulOp'):
    op_handl = F.MatMul(name, params = [(0,x)], ipos=[1])
    op_type = op_handl.optype.upper()
    [op2rsrc, op2dt, dev_obj] = get_dev_cfg(op_type)
    d = polaris.execute_simophandle(op_handl, [y], dev_obj, op2rsrc, op2dt)
    return d

def softmax(x, dim=-1, name='Softmax'):
    op_handl = F.Softmax(name)
    op_type = op_handl.optype.upper()
    [op2rsrc, op2dt, dev_obj] = get_dev_cfg(op_type)
    d = polaris.execute_simophandle(op_handl, [x], dev_obj, op2rsrc, op2dt)
    return d

def to_layout(tt_tensor, layout):
    # empty function to match the interface
    return tt_tensor

def to_torch(tt_tensor, dtype=None, device=None):
    # empty function to match the interface
    return tt_tensor

def from_torch(torch_tensor, dtype=None, layout=None, device=None, name=None):
    np_data = torch_tensor.detach().cpu().numpy()
    return F._from_data(name, np_data)

def from_torch_var(name, torch_tensor, is_param=False, is_const=False):
    np_data = torch_tensor.detach().cpu().numpy()
    return F._from_data(name, np_data, is_param=is_param, is_const=is_const)

def from_device(tt_tensor, dtype=None, layout=None, device=None):
    # empty function to match the interface
    return tt_tensor

def to_device(tt_tensor, dtype=None, layout=None, device=None):
    # empty function to match the interface
    return tt_tensor
