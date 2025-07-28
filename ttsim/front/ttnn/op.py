#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from ttsim.ops.op import SimOpFactory
from .tensor import Tensor, DataType, Layout
from .buffer import TensorMemoryLayout
from .memory import MemoryConfig

from enum import Enum, auto
from itertools import count
from dataclasses import dataclass
import numpy as np

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

class MathFidelity(Enum):
    LOFI  = auto()
    HIFI2 = auto()
    HIFI3 = auto()
    HIFI4 = auto()

    @classmethod
    def enumvalue(cls, s:str):
        return MathFidelity[s.upper()]

    @property
    def cname(self)->str:
        return self.name.lower()

op_counter = count(start=1, step=1)
def generate_new_op_name(): return f"ttsim.ttnn.Op_{next(op_counter)}"

def single_output_immediate_op(optype, /, preprocess=None):

    def _impl(*args, **kwargs):

        if preprocess:
            args, kwargs = preprocess(args, kwargs)

        tensor_args = [x for x in args if isinstance(x, Tensor)]
        devchk_list = [x.device for x in tensor_args]
        device = devchk_list[0]
        assert device and all(x == device for x in devchk_list), f"device check: {devchk_list}"

        op_name = generate_new_op_name()
        opinfo  = {'name': op_name, 'optype': optype, 'inList': [], 'attrs': kwargs}
        C       = Tensor(name=op_name + '.out',  op_out= [op_name], device=device)

        for x in args:
            if isinstance(x, Tensor):
                x.op_in.append(op_name)
                opinfo['inList'].append(x.name)
        opinfo['outList'] = [C.name]

        opcls      = SimOpFactory(optype)
        opobj      = opcls(opinfo)
        perf_stats = opobj.get_perf_counts(args, [C])
        opobj.update_tensor_counts(args, [C])

        device.add_op(opobj)

        return C

    return _impl

def argmax_pp(args_list, kwargs_dict):
    #translate attribs
    kwargs_dict['axis'] = kwargs_dict.get('dim', 0)
    if 'keepdim' in kwargs_dict:
        kwargs_dict['keepdims'] = 1 if kwargs_dict['keepdim'] else 0
    else:
        kwargs_dict['keepdims'] = 0
    return args_list, kwargs_dict

def reshape_pp(args_list, kwargs_dict):
    assert len(args_list) == 2, f"ttnn.reshape has 2 inputs"
    inT      = args_list[0]
    outShape = args_list[1]
    assert isinstance(inT, Tensor), f"ttnn.reshape 1st input should be a ttnn.Tensor"
    assert isinstance(outShape, (list, tuple)), f"ttnn.reshape 2nd input should be a list|tuple of ints"
    assert all(isinstance(x, int) for x in outShape), f"ttnn.reshape 2nd input should be a list|tuple of ints"

    outData = np.array(outShape, dtype=np.int64)
    outT = Tensor(shape=outData.shape, dtype=DataType.INT64, device=inT.device, data=outData)
    return (inT, outT), kwargs_dict


#Pointwise Unary
cos         = single_output_immediate_op('Cos')
gelu        = single_output_immediate_op('Gelu')
identity    = single_output_immediate_op('Identity')
leaky_relu  = single_output_immediate_op('LeakyRelu')
neg         = single_output_immediate_op('Neg')
relu        = single_output_immediate_op('Relu')
sigmoid     = single_output_immediate_op('Sigmoid')
sin         = single_output_immediate_op('Sin')
softmax     = single_output_immediate_op('Softmax')
tanh        = single_output_immediate_op('Tanh')

#Pointwise Binary
add         = single_output_immediate_op('Add')
multiply    = single_output_immediate_op('Mul')
subtract    = single_output_immediate_op('Sub')
div         = single_output_immediate_op('Div')
pow         = single_output_immediate_op('Pow')

#Pointwise Ternary
where       = single_output_immediate_op('Where')

#Reduction
argmax      = single_output_immediate_op('ArgMax', preprocess=argmax_pp)

#Data Movement
concat      = single_output_immediate_op('Concat')
reshape     = single_output_immediate_op('Reshape', preprocess=reshape_pp)

#Normalization
layer_norm  = single_output_immediate_op('LayerNormalization')
batch_norm  = single_output_immediate_op('BatchNormalization')

#Convolution
conv2d      = single_output_immediate_op('Conv')

#Pooling
global_avg_pool2d = single_output_immediate_op('GlobalAveragePool')
max_pool2d        = single_output_immediate_op('MaxPool')

#Matrix Multiplication
matmul      = single_output_immediate_op('MatMul')

#NEEDED
#permute, embedding, Conv2dConfig

#TTNN EXPERIMENTAL
#experimental.dropout     = single_output_immediate_op('Dropout')
#experimental.gather      = single_output_immediate_op('Gather')

#NOT IN TTNN
#constant    = single_output_immediate_op('Constant')
#transpose   = single_output_immediate_op('Transpose')

# CHECK w/ ttsim.ops.op
#TileOp               : ['Tile'],
#SliceOp              : ['Slice'],
#TriluOp              : ['Trilu'],
#DropoutOp            : ['Dropout'],
#EqualOp              : ['Equal'],
#CastOp               : ['Cast'],
#ShapeOp              : ['Shape'],
#RangeOp              : ['Range'],
#ResizeOp             : ['Resize']
#NonMaxSuppressionOp  : ['NonMaxSuppression']
#FlattenOp            : ['Flatten'], #Yolo-v7
#AveragePoolOp        : ['AveragePool'],

#Mutli-operator functions
def linear(*args, **kwargs):
    assert len(args) == 2, f"linear args #-inputs({len(args)}) != 2"
    A, B        = args[0], args[1]
    bias        = kwargs.get('bias',                   None)
    act         = kwargs.get('activation',             None)
    #t_A         = kwargs.get('transpose_a',            False)
    #t_B         = kwargs.get('transpose_b',            False)
    #dtype       = kwargs.get('dtype',                  None)
    #otile       = kwargs.get('output_tile',            None)
    #opt_otensor = kwargs.get('optional_output_tensor', None)
    #core_grid   = kwargs.get('core_grid',              None)
    #mem_cfg     = kwargs.get('memory_config',          MemoryConfig.DRAM)
    #pgm_cfg     = kwargs.get('program_config',         None)
    #ckrnl_cfg   = kwargs.get('compute_kernel_config',  None)

    not_impl_attrs = {
            'transpose_a'           : False,
            'transpose_b'           : False,
            'dtype'                 : None,
            'output_tile'           : None,
            'optional_output_tensor': None,
            'core_grid'             : None,
            #'memory_config'         : MemoryConfig.DRAM,
            'program_config'        : None,
            'compute_kernel_config' : None,
            }

    for aname,adefval in not_impl_attrs.items():
        if aname in kwargs:
            assert kwargs[aname] == adefval, f"linear.attrib: {aname} = {kwargs[aname]} not implemented yet!!"

    C = matmul(A, B)
    if bias is not None:
        C = add(C, bias)
    if act is not None:
        act_op = { 'relu': relu }[act]
        C = act_op(C)
    return C

