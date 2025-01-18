#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Chen Chen
Date created: 2025/01/10
Date last modified: 2021/10/10
Description: Save the model name and its layers to a pickle file.

Usage:
    $ python save_model_layers.py ../tests/SimpleCNN.py
"""
import sys
import os
import pickle
from collections import defaultdict
from cnnrfvis.models.SimpleCNN import get_model_rf_info

# TODO: refact
from cnnrfvis.models.SimpleStrideCNN import get_simple_stride_cnn_rf_info
from cnnrfvis.models.resnet import get_resnet18_rf_info
from cnnrfvis.models.PixelCNN import get_pixelcnn_rf_info

FUNC_DICT = {
    "SimpleCNN": get_model_rf_info,
    "SimpleStrideCNN": get_simple_stride_cnn_rf_info,
    "resnet": get_resnet18_rf_info,
    "PixelCNN": get_pixelcnn_rf_info,
}


def save_model_and_layers(model_file: str) -> None:
    meta_info = {}
    rf_info = {}
    model_layers_dict = defaultdict(list)

    model_name = model_file.split("/")[-1].split(".")[0]
    print(f"Model name: {model_name}")
    results = FUNC_DICT[model_name]()
    for k, v in results.rf_dict.items():
        model_layers_dict[model_name].append(k)

    meta_info["model_layers_dict"] = model_layers_dict
    meta_info["hw_dict"] = results.hw_dict
    meta_info["input_hw"] = [results.input_height, results.input_width]
    os.system(f"mkdir -p {model_name}")
    with open(f"meta_info.pkl", "wb") as f:
        pickle.dump(meta_info, f)
    for key, value in results.rf_dict.items():
        with open(f"{model_name}/{key}.pkl", "wb") as f:
            pickle.dump(value, f)


if __name__ == "__main__":
    # 检查是否提供了文件路径参数
    if len(sys.argv) != 2:
        print("Usage: python save_model_layers.py <model_file_path>")
        sys.exit(1)

    # 获取命令行参数
    model_file_path = sys.argv[1]

    save_model_and_layers(model_file_path)
