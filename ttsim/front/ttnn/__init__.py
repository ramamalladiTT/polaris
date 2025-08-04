#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from .device import open_device, close_device, ARCH
from .tensor import _rand, full, zeros, ones, from_torch, to_torch, to_layout, to_device, DataType
from .tensor import Layout, Shape
from .config import Conv2dConfig, WormholeComputeKernelConfig, init_device_compute_kernel_config
from .buffer import TensorMemoryLayout
from .memory import MemoryConfig
from .op     import *


float32  = DataType.FLOAT32
bfloat16 = DataType.BFLOAT16
int64    = DataType.INT64
uint32   = DataType.UINT32

ROW_MAJOR_LAYOUT = Layout.ROW_MAJOR_LAYOUT
TILE_LAYOUT      = Layout.TILE_LAYOUT

DRAM_MEMORY_CONFIG = MemoryConfig.DRAM
L1_MEMORY_CONFIG   = MemoryConfig.L1

L1_WIDTH_SHARDED_MEMORY_CONFIG = 0

#placeholders

def get_arch_name():
    return ARCH.WORMHOLE_B0.cname

class MatmulMultiCoreReuseMultiCast1DProgramConfig:

    def __init__(self, **kwargs):
        self.compute_with_storage_grid_size = kwargs.get('compute_with_storage_grid_size', (8, 8)),
        self.in0_block_w                    = kwargs.get('in0_block_w',                    1),
        self.out_subblock_h                 = kwargs.get('out_subblock_h',                 1),
        self.out_subblock_w                 = kwargs.get('out_subblock_w',                 1),
        self.per_core_M                     = kwargs.get('per_core_M',                     1),
        self.per_core_N                     = kwargs.get('per_core_N',                     1),
        self.fuse_batch                     = kwargs.get('fuse_batch',                     True),
        self.fused_activation               = kwargs.get('fused_activation',               None),
        self.mcast_in0                      = kwargs.get('mcast_in0',                      True),
        return

CoreGrid = 1
#TODO: Need to add these...
# CoreRangeSet, CoreRange
