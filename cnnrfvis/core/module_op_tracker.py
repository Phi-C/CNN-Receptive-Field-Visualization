import torch
from torch.utils._python_dispatch import TorchDispatchMode
from cnnrfvis.log import logger, setup_logging
from cnnrfvis.ops.factory import HandleFactory

setup_logging(log_file="app.log")
SKIP_LIST = [
    "aten::detach",
    "aten::empty",
    "aten::slice",
    "aten::unsqueeze",
    "aten::split",
    "aten::embedding",
]


class CatchEachOp(TorchDispatchMode):
    """Catch aten pps in PyTorch, can be used as context manager

    1. CatchEachOp is a subclass of TorchDisaptchMode, ref:
       https://pytorch.org/docs/stable/notes/extending.html#extending-all-torch-api-with-modes
    2. Here we focus on aten::conv layer for receptive field computation.
    """

    def __init__(self, verbose=True):
        super().__init__()
        self.verbose = verbose
        self.flag = False
        self.layer_idx = 0
        self.rf_dict = {}
        self.hw_dict = {}
        self.input_height = 0
        self.input_width = 0
        self.handlers = HandleFactory().get_handlers()

    def __exit__(self, exc_type, exc_val, exc_tb):
        return super().__exit__(exc_type, exc_val, exc_tb)

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
        op_name = func._schema.name
        logger.info(f"Operation: {op_name}::{self.layer_idx}")
        op_name = op_name.rstrip("_")
        output = func(*args, **(kwargs or {}))

        # if op_name == "aten::slice" or op_name == "aten::unsqueeze":
        #     print(args[0].shape, output.shape)
        if op_name == "aten::convolution" or op_name == "aten::pool":
            self.flag = True

        if self.flag == False:
            return output

        # TODO: stop_op is model-dependent
        if op_name == "aten::view" or op_name == "aten::mean":
            self.flag = False
            return output

        if (
            (op_name == "aten::slice")
            and (args[1] == (len(args[0].shape) - 1))
            and (args[2] > 0 or args[3] < args[0].shape[-1])
        ):
            # remapping
            input_tensor = args[0]
            if hasattr(input_tensor, "rf_dict"):
                in_w = args[0].shape[-1]
                out_h = args[0].shape[-2]
                out_w = args[3] - args[2]
                rf_dict = {}
                for h in range(out_h):
                    for w in range(out_w):
                        rf_dict[h * out_w + w] = input_tensor.rf_dict[
                            h * in_w + w + args[2]
                        ]
                output.rf_dict = rf_dict
                self.layer_idx += 1
            return output

        if op_name in SKIP_LIST:
            input_tensor = args[0][0] if isinstance(args[0], tuple) else args[0]
            output_tensor = output[0] if isinstance(output, tuple) else output
            if hasattr(input_tensor, "rf_dict"):
                if isinstance(output_tensor, list) or isinstance(output_tensor, tuple):
                    for item in output_tensor:
                        item.rf_dict = input_tensor.rf_dict
                else:
                    output_tensor.rf_dict = input_tensor.rf_dict
            self.layer_idx += 1
            return output

        # TODO: Check
        # if self.layer_idx == 0:
        if not hasattr(args[0], "rf_dict"):
            input_tensor = args[0]
            input_height, input_width = input_tensor.shape[-2:]
            self.input_height = input_height
            self.input_width = input_width
            input_rf_dict = {}
            for i in range(input_height):
                for j in range(input_width):
                    input_rf_dict[i * input_width + j] = torch.zeros(
                        input_height, input_width
                    )
                    input_rf_dict[i * input_width + j][i, j] = 1
            input_tensor.rf_dict = input_rf_dict

        if op_name not in self.handlers:
            raise NotImplementedError(f"Operation {op_name} not supported")
        index_mapping = self.handlers[op_name].handle(args, output)
        rf_dict = self.handlers[op_name].compute_rf_tensor(args, index_mapping)

        self.rf_dict[f"{op_name}::{self.layer_idx}"] = rf_dict
        output = self._update_output(op_name, output, rf_dict)
        output_tensor = output[0] if isinstance(output, tuple) else output
        output_height, output_width = output_tensor.shape[-2:]
        self.hw_dict[f"{op_name}::{self.layer_idx}"] = (output_height, output_width)
        self.layer_idx += 1

        # Used for debug
        if self.verbose:
            middel_index = output_height // 2 * output_width + output_width // 2
            first_row, last_row, first_col, last_col = self._find_nonzero_bounds(
                output_tensor.rf_dict[middel_index]
            )
            logger.info(
                f"Layer {self.layer_idx}: {op_name}, RF: {output_tensor.rf_dict[middel_index][first_row:last_row+1, first_col:last_col+1]}, RF Shape: {output_tensor.rf_dict[middel_index][first_row:last_row+1, first_col:last_col+1].shape}"
            )
            torch.save(
                output_tensor.rf_dict[164], f"debug/{op_name}::{self.layer_idx}.pt"
            )
        return output

    def _update_output(self, op_name, output, rf_dict):
        if (
            op_name == "aten::max_pool2d_with_indices"
            or op_name == "aten::native_batch_norm"
        ):
            output[0].rf_dict = rf_dict
        else:
            output.rf_dict = rf_dict
        return output

    def _find_nonzero_bounds(self, tensor):
        if tensor.dim() != 2:
            raise ValueError("Input tensor must be 2D.")

        # 检查是否有非全零的行
        rows_nonzero = tensor.any(dim=1)
        if not rows_nonzero.any():
            first_row = None
            last_row = None
        else:
            first_row = torch.argmax(rows_nonzero.float()).item()  # 第一个非全零行
            last_row = (
                tensor.size(0) - torch.argmax(rows_nonzero.flip(0).float()).item() - 1
            )  # 最后一个非全零行

        # 检查是否有非全零的列
        cols_nonzero = tensor.any(dim=0)
        if not cols_nonzero.any():
            first_col = None
            last_col = None
        else:
            first_col = torch.argmax(cols_nonzero.float()).item()  # 第一个非全零列
            last_col = (
                tensor.size(1) - torch.argmax(cols_nonzero.flip(0).float()).item() - 1
            )  # 最后一个非全零列

        return first_row, last_row, first_col, last_col
