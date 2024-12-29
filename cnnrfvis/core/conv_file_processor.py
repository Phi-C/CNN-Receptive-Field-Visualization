import os
import sys

import numpy as np


class ConvRFVis:
    """
    Receptive Field Visualizer Based on Conv OP params
    """
    def __init__(self, file_path: str = None, layer_idx: int = None) -> None:
        self.file_path = file_path
        self.layer_idx = layer_idx
        self.template = None

    def process(self):
        """
        Get the receptive field and visualize it.
        """
        with open(self.file_path, "r") as f:
            lines = f.readlines()
        conv_layer_num = len(lines)
        assert conv_layer_num - 1 >= self.layer_idx, "layer index is out of range"

        for idx in range(self.layer_idx + 1):
            conv_info = lines[idx]
            kernel, stride, pad = conv_info.strip().split('\t')
            print(
                f"layer_idx: {idx}\tkernel: {kernel}\t stride: {stride}\tpad: {pad}"
            )
            kernel_h, kernel_w = kernel[1:-1].split(',')
            kernel_h, kernel_w = int(kernel_h), int(kernel_w)
            print(f"kernel_h: {kernel_h}, kernel_w: {kernel_w}")
            print(
                f"type(kernel_h): {type(kernel_h)}, type(kernel_w): {type(kernel_w)}"
            )
            if idx == 0:
                self.template = np.ones((kernel_h, kernel_w))
            else:
                self.template += 1
                self.template = np.pad(self.template,
                                       pad_width=((int(kernel_h / 2),
                                                   int(kernel_h / 2)),
                                                  (int(kernel_w / 2),
                                                   int(kernel_w / 2))),
                                       mode='constant',
                                       constant_values=1)

        print(f"RF: {self.template}")


def process_conv_file(file_path, layer_idx):
    print(f"conv_file: {file_path}, layer_idx = {layer_idx}")
    visualizer = ConvRFVis(file_path, layer_idx)
    visualizer.process()


if __name__ == "__main__":
    file_path = "conv.txt"
    layer_idx = 3
    process_conv_file(file_path, layer_idx)
