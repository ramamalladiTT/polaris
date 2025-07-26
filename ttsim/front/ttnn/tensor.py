#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from ttsim.ops.op import SimOpFactory
from ttsim.ops.tensor import SimTensor
from .device import Device

from enum import Enum, auto
from itertools import count
import numpy as np

########################################## DataType ##########################################
class DataType(Enum):
    UINT8     = auto()
    UINT16    = auto()
    UINT32    = auto()
    INT32     = auto()
    INT64     = auto()
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
                'UINT32'    : 4,
                'INT32'     : 4,
                'INT64'     : 8,
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
                'UINT32'    : np.dtype(np.uint32),
                'INT32'     : np.dtype(np.int32),
                'INT64'     : np.dtype(np.int64),
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

    @property
    def T(self):
        opname = self.name + '.transpose_op'
        optype = 'Transpose'
        perm   = [i for i in range(self.rank())]
        perm[-2], perm[-1] = perm[-1], perm[-2] #swap last 2 dims
        opinfo = {'name': opname, 'optype': optype, 'inList': [self.name], 'attrs': {'perm': perm}}
        outT   = Tensor(name=opname + '.out', op_out=[opname], device=self.device)
        opinfo['outList'] = [outT.name]

        opcls  = SimOpFactory(optype)
        opobj  = opcls(opinfo)
        pstats = opobj.get_perf_counts([self], [outT])

        self.device.add_op(opobj)

        return outT

    def view(self, *args):
        npdata = np.array(args, dtype=np.int64)
        opname = self.name + '.view_op'
        shapeT = Tensor(name=opname + '.shapeT',device=self.device, data=npdata,
                        shape=list(npdata.shape), dtype=DataType.INT64, op_in=[opname])
        optype = 'Reshape'
        opinfo = {'name': opname, 'optype': optype, 'inList': [self.name, shapeT.name]}
        outT   = Tensor(name=opname + '.out', op_out=[opname], device=self.device)
        opinfo['outList'] = [outT.name]

        opcls  = SimOpFactory(optype)
        opobj  = opcls(opinfo)
        pstats = opobj.get_perf_counts([self, shapeT], [outT])

        self.device.add_op(opobj)

        return outT

    def to(self, dt):
        self.dtype = dt.to_numpy
        return self

    def item(self):
        """ returns the Python scalar value of the tensor if the tensor has exactly one element
        (i.e., it is a 0-dimensional tensor or a scalar tensor). If the tensor has more than one
        element, calling item() will raise an error. If the tensor is empty/None item fails again!!
        """
        assert self.shape == [1], f"Tensor item() is valid only for tensor with exactly one element: {self.shape}"
        assert self.data is not None, f"Tensor item() called for missing data: {self.data}"
        return self.data[0]


def _rand(shape, dtype, device=None):
    return Tensor(shape=shape, dtype=dtype, device=device)

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

