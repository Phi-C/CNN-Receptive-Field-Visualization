# CNN-Receptive-Field-Visualization
Tools: visualize CNN recptive field for better model design


# Quick Start
```shell
pip install .
cnnrfvis --conv_file tests/conv_list.txt --layer_idx 2 
```
text file must organized as following: `(kernel_size_h, kernel_size_w)  (stride_h, stride_w)  (pad_h, pad_w)` 