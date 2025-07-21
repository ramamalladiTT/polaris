#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from ttsim.ops.op import SimOpFactory
from .tensor import Tensor

from itertools import count

op_counter = count(start=1, step=1)
def generate_new_op_name(): return f"ttsim.ttnn.Op_{next(op_counter)}"

def single_output_immediate_op(optype):

    def _impl(*args, **kwargs):
        devchk_list = [x.device for x in args]
        device = devchk_list[0]
        assert device and all(x == device for x in devchk_list), f"device check: {devchk_list}"

        op_name = generate_new_op_name()
        opinfo  = {'name': op_name, 'optype': optype, 'inList': [], 'attrs': kwargs}
        C       = Tensor(name=op_name + '.out',  op_out= [op_name], device=device)

        for x in args:
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

add         = single_output_immediate_op('Add')
mul         = single_output_immediate_op('Mul')
relu        = single_output_immediate_op('Relu')
softmax     = single_output_immediate_op('Softmax')
layer_norm  = single_output_immediate_op('LayerNormalization')
matmul      = single_output_immediate_op('MatMul')
linear      = matmul

#reshape, permute, embedding
#Conv2dConfig
#conv2d
#reshape     = single_output_immediate_op('Reshape')

