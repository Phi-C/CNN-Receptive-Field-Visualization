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
import subprocess
import pickle
from collections import defaultdict


def save_model_and_layers(model_file: str) -> None:
    model_layers_dict = defaultdict(list)

    model_name = model_file.split("/")[-1].split(".")[0]
    print(f"Model name: {model_name}")
    results = subprocess.run(
        ["python3", f"{model_file}"], capture_output=True, text=True
    )
    output_lines = results.stdout.splitlines()
    for line in output_lines:
        if "Operation" in line:
            layer_name = line.split("-")[-1].split(" ")[-1].strip()
            if layer_name == "aten::view":
                break
            model_layers_dict[model_name].append(layer_name)

    with open("model_layers_dict.pkl", "wb") as f:
        pickle.dump(model_layers_dict, f)


if __name__ == "__main__":
    # 检查是否提供了文件路径参数
    if len(sys.argv) != 2:
        print("Usage: python save_model_layers.py <model_file_path>")
        sys.exit(1)

    # 获取命令行参数
    model_file_path = sys.argv[1]

    save_model_and_layers(model_file_path)
