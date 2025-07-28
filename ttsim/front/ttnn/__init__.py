#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from .device import open_device, close_device
from .tensor import _rand, full, zeros, ones, from_torch, to_torch, to_layout, to_device, DataType, Layout
from .memory import MemoryConfig
from .op     import *

float32  = DataType.FLOAT32
bfloat16 = DataType.BFLOAT16
int64    = DataType.INT64

ROW_MAJOR_LAYOUT = Layout.ROW_MAJOR_LAYOUT
TILE_LAYOUT      = Layout.TILE_LAYOUT

DRAM_MEMORY_CONFIG = MemoryConfig.DRAM
L1_MEMORY_CONFIG   = MemoryConfig.L1

