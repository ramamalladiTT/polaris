#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from .device import open_device, close_device, ARCH
from .tensor import _rand, full, zeros, ones, from_torch, to_torch, to_layout, to_device, DataType
from .tensor import Layout, Shape
from .config import Conv2dConfig, WormholeComputeKernelConfig, init_device_compute_kernel_config
from .config import MatmulMultiCoreReuseMultiCast1DProgramConfig
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

CoreGrid = 1
#TODO: Need to add these...
# CoreRangeSet, CoreRange
