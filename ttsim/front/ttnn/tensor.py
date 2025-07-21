#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from ttsim.ops.tensor import SimTensor
from .device import Device

from enum import Enum, auto
from itertools import count
import numpy as np

########################################## DataType ##########################################
class DataType(Enum):
    UINT8     = auto()
    UINT16    = auto()
    INT32     = auto()
    UINT32    = auto()
    FLOAT32   = auto()
    BFLOAT16  = auto()
    BFLOAT8_B = auto()
    BFLOAT4_B = auto()

    @classmethod
    def enumvalue(cls, s:str):
        return DataType[s.upper()]

    @property
    def itemsize(self)->int:
        return {
                'UINT8'     : 1,
                'UINT16'    : 2,
                'INT32'     : 4,
                'UINT32'    : 4,
                'FLOAT32'   : 4,
                'BFLOAT16'  : 2,
                'BFLOAT8_B' : 2,
                'BFLOAT4_B' : 1,
                }[self.name]

    @property
    def to_numpy(self):
        return {
                'UINT8'     : np.dtype(np.uint8),
                'UINT16'    : np.dtype(np.uint16),
                'INT32'     : np.dtype(np.int32),
                'UINT32'    : np.dtype(np.uint32),
                'FLOAT32'   : np.dtype(np.float32),
                'BFLOAT16'  : np.dtype(np.float32), #float16 not supported in onnx dump!!
                'BFLOAT8_B' : np.dtype(np.float32),
                'BFLOAT4_B' : np.dtype(np.float32),
                }[self.name]

    @property
    def cname(self)->str:
        return self.name.lower()

class Layout(Enum):
    ROW_MAJOR_LAYOUT = auto()
    TILE_LAYOUT      = auto()

    @classmethod
    def enumvalue(cls, s:str):
        return Layout[s.upper()]

    @property
    def cname(self)->str:
        return self.name.lower()

class Tensor(SimTensor):
    tensor_counter = count(start=1, step=1)

    def __init__(self, *args, **kwargs):
        if args:
            assert len(args) == 1, f"More than 1 positional argument in Tensor constructor!!: {args}"
            tensor_like = args[0]
            assert isinstance(tensor_like, np.ndarray), "ERR"
            dtype, shape = tensor_like.dtype, tensor_like.shape
            #ignoring dtype for now -- eventually will need to reconcile these with kwargs!!
            kwargs['shape'] = tensor_like.shape

        typechecks = { 'dtype': DataType, 'layout': Layout, 'device': Device }
        for kk,cls in typechecks.items():
            if kk in kwargs:
                obj = kwargs[kk]
                assert isinstance(obj, cls), f"Error: Tensor Creation -- attribute {kk}={obj} should be of type {cls}"

        if 'dtype' in kwargs:
            kwargs['dtype'] = kwargs['dtype'].to_numpy

        if 'name' not in kwargs:
            kwargs['name'] = f"ttsim.ttnn.Tensor_{next(self.tensor_counter)}"

        if 'shape' in kwargs:
            obj = kwargs['shape']
            assert isinstance(obj, (list, tuple)), f"Error: Tensor Creation -- attribute {shape}={obj} should be op type list|tuple"
            if isinstance(obj, tuple):
                kwargs['shape'] = list(obj)

        super().__init__(kwargs)
        self.device     = kwargs.get('device',     None)
        self.layout     = kwargs.get('layout',     None)
        self.fill_value = kwargs.get('fill_value', None)

        if self.device:
            self.device.add_tensor(self)
        return

    def __str__(self):
        return f"{super().__str__()} ==> ttnn: {self.device}, {self.layout}"


def _rand(shape, dtype):
    return Tensor(shape=shape, dtype=dtype)

def zeros(shape, dtype, layout, device):
    return Tensor(shape=shape, dtype=dtype, layout=layout, device=device, fill_value=0)

def ones(shape, dtype, layout, device):
    return Tensor(shape=shape, dtype=dtype, layout=layout, device=device, fill_value=1)

def full(shape, fill_value, dtype, layout, device):
    return Tensor(shape=shape, dtype=dtype, layout=layout, device=device, fill_value=fill_value)

def from_torch(torch_tensor_like, **kwargs):
    for k,v in kwargs.items():
        if hasattr(torch_tensor_like, k):
            setattr(torch_tensor_like, k, v.to_numpy if k == 'dtype' else v)

    if 'device' in kwargs:
        torch_tensor_like = to_device(torch_tensor_like, kwargs['device'])

    return torch_tensor_like

def to_torch(tt_tensor_like):
    return tt_tensor_like

def to_layout(tt_tensor_like, layout):
    tt_tensor_like.layout = layout
    return tt_tensor_like

def to_device(tt_tensor_like, device):
    if tt_tensor_like.device:
        old_device = tt_tensor_like.device
        if tt_tensor_like.name in old_device.tensors:
            del old_device.tensors[tt_tensor_like.name]

    tt_tensor_like.device = device
    device.add_tensor(tt_tensor_like)

    return tt_tensor_like

