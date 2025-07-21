#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from ttsim.graph import WorkloadGraph

class Device:
    def __init__(self, **kwargs):
        self.args    = kwargs
        self.tensors = {}
        self.ops     = {}
        return

    def add_tensor(self, t):
        if t.name not in self.tensors:
            self.tensors[t.name] = t

    def add_op(self, o):
        if o.name not in self.ops:
            self.ops[o.name] = o

    def get_graph(self):
        gg = WorkloadGraph('xxx')
        for _,t in self.tensors.items():
            gg.add_tensor(t)
        for _,o in self.ops.items():
            gg.add_op(o)
        gg.construct_graph()
        return gg

    def __str__(self):
        return f"(Device: {self.args})"


def open_device(**kwargs):
    return Device(**kwargs)

def close_device(device: Device):
    return

