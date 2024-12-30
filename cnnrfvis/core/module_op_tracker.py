import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode


class CatchEachOp(TorchDispatchMode):

    def __init__(self, verbose=True):
        super().__init__()
        self.verbose = verbose
        self.template = None
        self.prev_stride = None
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
                width = (k_w - 1) * self.prev_stride[1] + self.template.shape[1]
                height = (k_h - 1) * self.prev_stride[0] + self.template.shape[0]
                template = torch.zeros(height, width)
                for i in range(k_h):
                    for j in range(k_w):
                        start_h = i * self.prev_stride[0]
                        end_h = start_h + self.template.shape[0]
                        start_w = j * self.prev_stride[1]
                        end_w = start_w + self.template.shape[1]
                        template[start_h:end_h, start_w:end_w] += self.template
                self.template = template

            if self.verbose:
                print(
                    f"Kernel_Size: {weight.shape[-2:]}, Stride: {stride}, Padding: {padding}"
                )
            print(
                f"ReceptiveFileds for {self.conv_layer_idx}-th conv: {self.template}, {self.template.shape}"
            )

            self.conv_layer_idx += 1
            self.prev_stride = stride

        return func(*args, **(kwargs or {}))
