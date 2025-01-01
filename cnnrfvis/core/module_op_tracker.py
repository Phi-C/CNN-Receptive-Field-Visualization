import torch 
from torch.utils._python_dispatch import TorchDispatchMode
from cnnrfvis.logging import logger, setup_logging

setup_logging(log_file='app.log')

class CatchEachOp(TorchDispatchMode):
    """Catch aten pps in PyTorch, can be used as context manager

     1. CatchEachOp is a subclass of TorchDisaptchMode, ref: 
        https://pytorch.org/docs/stable/notes/extending.html#extending-all-torch-api-with-modes
     2. Here we focus on aten::conv layer for receptive field computation.
    """
    def __init__(self, verbose=True):
        super().__init__()
        self.verbose = verbose
        self.template = None
        self.prev_stride = None
        self.conv_layer_idx = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        """Intercept and process PyTorch operations

        Args:
            func: The function object of the current operation, such as `aten::add` or `aten::conv2d`
            type: The type of input tensor
            args: The input arguments of the operation
            kwargs: The input keyword arguments of the operation

        Return:
            The output result of the operation
        """
        # Find convolution operator
        if func._schema.name.startswith('aten::conv'):
            weight, stride, padding = args[1], args[3], args[4]
            kernel_height, kernel_width = weight.shape[-2:]

            # Initialize or update recptive field template
            if self.conv_layer_idx == 0:
                self.template = torch.ones(kernel_height, kernel_width)
            else:
                self._update_template(kernel_height, kernel_width)

            # Print related information
            if self.verbose:
                logger.info(
                    f"Kernel_Size: {kernel_height, kernel_width}, "
                    f"Stride: {stride}, Padding: {padding}"
                )
            logger.info(
                f"ReceptiveFields for {self.conv_layer_idx}-th conv: "
                f"{self.template}, {self.template.shape}"
            )

            # Update states
            self.conv_layer_idx += 1
            self.prev_stride = stride

        # Continue original operation
        return func(*args, **(kwargs or {}))

    def _update_template(self, kernel_height, kernel_width):
        """Update receptive field template
        
        Args:
            kernel_height: convolution kernel height
            kernel_width: convolution kernel width
        """
        new_height = (kernel_height - 1) * self.prev_stride[0] + self.template.shape[0]
        new_width = (kernel_width - 1) * self.prev_stride[1] + self.template.shape[1]
        new_template = torch.zeros(new_height, new_width)

        for i in range(kernel_height):
            for j in range(kernel_width):
                start_h = i * self.prev_stride[0]
                end_h = start_h + self.template.shape[0]
                start_w = j * self.prev_stride[1]
                end_w = start_w + self.template.shape[1]
                new_template[start_h:end_h, start_w:end_w] += self.template

        self.template = new_template