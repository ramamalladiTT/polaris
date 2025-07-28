#!/usr/bin/env python
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from enum import Enum, auto

class MemoryConfig(Enum):
    DRAM = auto()
    L1   = auto()

    @classmethod
    def enumvalue(cls, s:str):
        return MemoryConfig[s.upper()]

    @property
    def cname(self)->str:
        return self.name.lower()

