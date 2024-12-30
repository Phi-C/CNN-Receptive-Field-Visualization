import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode


class CatchEachOp(TorchDispatchMode):

    def __init__(self, verbose=True):
        super().__init__()
        self.verbose = verbose
        self.template = None
        self.conv_layer_idx = 0

    def __torch_dispatch__(self, func, types, args=..., kwargs=None):
        if func._schema.name.startswith('aten::conv'):
            weight = args[1]
            stride = args[3]
            padding = args[4]
            k_h, k_w = weight.shape[-2:]
            if self.conv_layer_idx == 0:
                self.template = torch.ones(k_h, k_w)
            else:
                self.template += 1
                self.template = F.pad(
                    self.template,
                    (int(k_w / 2), int(k_w / 2), int(k_h / 2), int(k_h / 2)),
                    value=1)

            if self.verbose:
                print(
                    f"Kernel_Size: {weight.shape[-2:]}, Stride: {stride}, Padding: {padding}"
                )
            print(
                f"ReceptiveFileds for {self.conv_layer_idx}-th conv: {self.template}"
            )

            self.conv_layer_idx += 1

        return func(*args, **(kwargs or {}))
