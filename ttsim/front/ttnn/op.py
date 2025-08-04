#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from ttsim.ops.op import SimOpFactory
from .tensor import Tensor, DataType
from .memory import MemoryConfig

from enum import Enum, auto
from itertools import count
import numpy as np

class MathFidelity(Enum):
    LoFi  = auto()
    HiFi2 = auto()
    HiFi3 = auto()
    HiFi4 = auto()

    @classmethod
    def enumvalue(cls, s:str):
        return MathFidelity[s]

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
        device      = devchk_list[0]
        assert device and all(x == device for x in devchk_list), f"device check: {devchk_list}"

        op_name = generate_new_op_name()
        opinfo  = {'name': op_name, 'optype': optype, 'inList': [], 'attrs': kwargs}
        C       = Tensor(name=op_name + '.out',  op_out= [op_name], device=device)

        new_args = []
        for i,x in enumerate(args):
            if isinstance(x, Tensor):
                x.op_in.append(op_name)
                opinfo['inList'].append(x.name)
                new_args.append(x)
            elif isinstance(x, (int,float)):
                #print(f"FOUND not Tensor input in ttnn.op({optype}) : {type(x)}")
                if optype in ['Add', 'Sub', 'Mul']:
                    tmp = Tensor(name=f"{op_name}.in.{i}", shape=[], dtype=DataType.FLOAT32, device=device)
                    tmp.op_in.append(op_name)
                    opinfo['inList'].append(tmp.name)
                    new_args.append(tmp)
                else:
                    exit(0)
            else:
                assert False, f"Unknown input type in ttnn.op({optype}) : {type(x)}"
        opinfo['outList'] = [C.name]

        opcls      = SimOpFactory(optype)
        opobj      = opcls(opinfo)
        perf_stats = opobj.get_perf_counts(new_args, [C])
        #print(f"Op: {optype} perf_stats: {perf_stats}")
        opobj.update_tensor_counts(new_args, [C])

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

def permute_pp(args_list, kwargs_dict):
    inT = args_list[0]
    assert isinstance(inT, Tensor), f"ttnn.permute 1st input should be a ttnn.Tensor"
    assert isinstance(args_list[1], (list, tuple)), f"ttnn.permute 2nd input should be a list|tuple of ints"
    kwargs_dict['perm'] = list(args_list[1])
    return (inT, ), kwargs_dict

def embedding_pp(args_list, kwargs_dict):
    # TTNN passes in the order indices, weights while Polaris takes weights, indices
    assert len(args_list) == 2, f"ttnn.embedding has 2 inputs"
    input_tensor  = args_list[0]
    weight_tensor = args_list[1]
    assert isinstance(input_tensor, Tensor),  f"ttnn.embedding 1st input should be a ttnn.Tensor: {input_tensor}"
    assert isinstance(weight_tensor, Tensor), f"ttnn.embedding 2nd input should be a ttnn.Tensor: {weight_tensor}"
    return (weight_tensor, input_tensor), kwargs_dict

def layer_norm_pp(args_list, kwargs_dict):
    input_tensor          = args_list[0]
    weight_tensor         = kwargs_dict['weight']                if 'weight'                in kwargs_dict else None
    bias_tensor           = kwargs_dict['bias']                  if 'bias'                  in kwargs_dict else None
    epsilon               = kwargs_dict['epsilon']               if 'epsilon'               in kwargs_dict else None
    memory_config         = kwargs_dict['memory_config']         if 'memory_config'         in kwargs_dict else None
    compute_kernel_config = kwargs_dict['compute_kernel_config'] if 'compute_kernel_config' in kwargs_dict else None

    assert isinstance(input_tensor, Tensor), f"ttnn.layer_norm 1st input should be a ttnn.Tensor"
    assert isinstance(weight_tensor, Tensor), f"ttnn.layer_norm 2nd input should be a ttnn.Tensor"
    if bias_tensor is not None:
        assert isinstance(bias_tensor, Tensor), f"ttnn.layer_norm 3rd input should be a ttnn.Tensor"

    return (input_tensor, weight_tensor, bias_tensor), kwargs_dict

def conv2d_pp(args_list, kwargs_dict):
    input_tensor  = kwargs_dict['input_tensor']
    weight_tensor = kwargs_dict['weight_tensor']
    bias_tensor   = kwargs_dict['bias_tensor']
    padding_size  = kwargs_dict['padding'][0]
    pads = [padding_size for i in range(4)]
    kwargs_dict = {'pads': pads, 'kernel_shape': list(kwargs_dict['kernel_size'])}
    return (input_tensor, weight_tensor, bias_tensor), kwargs_dict

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
reshape     = single_output_immediate_op('Reshape',   preprocess=reshape_pp)
embedding   = single_output_immediate_op('Gather',    preprocess=embedding_pp)
permute     = single_output_immediate_op('Transpose', preprocess=permute_pp)

#Normalization
layer_norm  = single_output_immediate_op('LayerNormalization', preprocess=layer_norm_pp)
batch_norm  = single_output_immediate_op('BatchNormalization')

#Convolution
conv2d      = single_output_immediate_op('Conv', preprocess=conv2d_pp)

#Pooling
global_avg_pool2d = single_output_immediate_op('GlobalAveragePool')
max_pool2d        = single_output_immediate_op('MaxPool')

#Matrix Multiplication
matmul      = single_output_immediate_op('MatMul')


Tensor.__add__ = add
#Tensor.__sub__ = subtract
Tensor.__mul__ = multiply
#Tensor.__div__ = div
#Tensor.__pow__ = pow

Tensor.__matmul__ = matmul

Tensor.reshape = reshape

#Mutli-operator functions
def linear(*args, **kwargs):
    assert len(args) == 2, f"linear args #-inputs({len(args)}) != 2"
    A, B        = args[0], args[1]
    bias        = kwargs.get('bias',                   None)
    act         = kwargs.get('activation',             None)
    #t_A         = kwargs.get('transpose_a',            False)
    #t_B         = kwargs.get('transpose_b',            False)
    dtype       = kwargs.get('dtype',                  None)
    #otile       = kwargs.get('output_tile',            None)
    #opt_otensor = kwargs.get('optional_output_tensor', None)
    core_grid   = kwargs.get('core_grid',              None)
    #mem_cfg     = kwargs.get('memory_config',          MemoryConfig.DRAM)
    #pgm_cfg     = kwargs.get('program_config',         None)
    #ckrnl_cfg   = kwargs.get('compute_kernel_config',  None)

    not_impl_attrs = {
            'transpose_a'           : False,
            'transpose_b'           : False,
            #'dtype'                 : None,
            'output_tile'           : None,
            'optional_output_tensor': None,
            #'core_grid'             : None,
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
        act_op = { 'relu': relu, 'gelu': gelu }[act]
        C = act_op(C)
    return C

