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

#    @classmethod
#    def from_numpy(cls, numpy_dtype):
#        if hasattr(numpy_dtype, 'dtype'):
#            numpy_dtype = numpy_dtype.dtype
#
#        if hasattr(numpy_dtype, 'name'):
#            dtype_str = numpy_dtype.name
#        elif isinstance(numpy_dtype, str):
#            dtype_str = numpy_dtype
#        else:
#            dtype_str = str(numpy_dtype)
#
#        # Mapping from numpy dtype names to DataType enums
#        dtype_mapping = {
#            'uint8'  : cls.UINT8,
#            'uint16' : cls.UINT16,
#            'int32'  : cls.INT32,
#            'uint32' : cls.UINT32,
#            'int64'  : cls.INT64,
#            'float32': cls.FLOAT32,
#            'float16': cls.BFLOAT16,
#            'bool'   : cls.UINT8,  # Map bool to uint8
#        }
#
#        # Try exact match first
#        if dtype_str in dtype_mapping:
#            return dtype_mapping[dtype_str]
#
#        # Try with numpy dtype object mapping
#        numpy_dtype_mapping = {
#            np.dtype(np.uint8)  : cls.UINT8,
#            np.dtype(np.uint16) : cls.UINT16,
#            np.dtype(np.int32)  : cls.INT32,
#            np.dtype(np.uint32) : cls.UINT32,
#            np.dtype(np.int64)  : cls.INT64,
#            np.dtype(np.float32): cls.FLOAT32,
#            np.dtype(np.float16): cls.BFLOAT16,
#        }
#
#        if numpy_dtype in numpy_dtype_mapping:
#            return numpy_dtype_mapping[numpy_dtype]
#
#        # Default fallback
#        return cls.FLOAT32

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

class Shape(list):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def to_rank(self, N):
        rem = N - len(self)
        assert rem >= 0, f"Cannot convert Shape({self}) to rank {N}"
        return [1 for i in range(rem)] + [i for i in self]

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

#    def unsqueeze(self, dim: int):
#        """Unsqueeze the tensor at the specified dimension."""
#        #SK: Check this
#        if dim < 0:
#            dim += len(self.shape) + 1
#        new_shape = self.shape[:dim] + [1] + self.shape[dim:]
#        return Tensor(shape=new_shape, dtype=DataType.from_numpy(self.dtype.name))
#
#    def squeeze(self, dim: int):
#        """Squeeze the tensor at the specified dimension."""
#        #SK: Check this
#        if dim < 0:
#            dim += len(self.shape) + 1
#        if dim >= len(self.shape) or self.shape[dim] != 1:
#            print(f"Cannot squeeze dimension {dim} of shape {self.shape}")
#            return self
#        new_shape = self.shape[:dim] + self.shape[dim+1:]
#        return Tensor(shape=new_shape, dtype=DataType.from_numpy(self.dtype.name), device=self.device)


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

#def ShardTensor2dMesh(mesh_device, dims, mesh_shape):
#    # dummy implementation for ShardTensor2dMesh
#    pass

#def as_tensor(tensor_like, dtype=None, layout=None, device=None, fill_value=None, mesh_mapper=None, memory_config=None, cache_file_name=None):
#    if isinstance(tensor_like, Tensor):
#        return to_device(tensor_like, device) if device else tensor_like
#
#    if isinstance(tensor_like, np.ndarray):
#        shape = tensor_like.shape
#        if dtype is None:
#            dtype = DataType.from_numpy(tensor_like.dtype.name)
#        return Tensor(shape=shape, dtype=dtype, layout=layout, device=device, fill_value=fill_value, data=tensor_like)
#
#    raise TypeError(f"Unsupported type for as_tensor: {type(tensor_like)}")

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

def to_layout(tt_tensor_like, layout, dtype=None):
    tt_tensor_like.layout = layout
    if dtype is not None:
        tt_tensor_like.dtype = dtype
    return tt_tensor_like

def to_device(tt_tensor_like, device):
    assert device is not None, "device=None passed to to_device"

    if tt_tensor_like.device:
        old_device = tt_tensor_like.device
        if tt_tensor_like.name in old_device.tensors:
            del old_device.tensors[tt_tensor_like.name]

    tt_tensor_like.device = device
    device.add_tensor(tt_tensor_like)

    return tt_tensor_like

#def from_device(tt_tensor_like, device=None):
#    if device is not None:
#        return tt_tensor_like
#    elif (tt_tensor_like.device is not None and
#          tt_tensor_like.device != device):
        #SK: What is the case if tt_tensor_like.device != device, what about if it is already there in device
        #SK: We should add it to device in case it is not already there..
#        return tt_tensor_like
#    else:
        #SK: Don't understand this: e.g., tt_tensor_like.device == device
#        assert f"Tensor {tt_tensor_like.name} not found in device {device}"
