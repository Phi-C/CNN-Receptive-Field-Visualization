import torch 
from torch.utils._python_dispatch import TorchDispatchMode
from cnnrfvis.logging import logger, setup_logging
from cnnrfvis.ops.factory import HandleFactory

setup_logging(log_file='app.log')
SKIP_LIST = ['aten::detach']

class CatchEachOp(TorchDispatchMode):
    """Catch aten pps in PyTorch, can be used as context manager

     1. CatchEachOp is a subclass of TorchDisaptchMode, ref: 
        https://pytorch.org/docs/stable/notes/extending.html#extending-all-torch-api-with-modes
     2. Here we focus on aten::conv layer for receptive field computation.
    """
    def __init__(self, verbose=True):
        super().__init__()
        self.verbose = verbose
        self.flag = True
        self.layer_idx = 0
        self.handlers = HandleFactory().get_handlers()

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
        logger.info(f"Operation: {op_name}")
        output = func(*args, **(kwargs or {}))

        if self.flag == False:
            return output

        if op_name == 'aten::view':
            self.flag = False
            return output

        if op_name in SKIP_LIST:
            input_tensor = args[0][0] if isinstance(args[0], tuple) else args[0]
            output_tensor = output[0] if isinstance(output, tuple) else output
            output_tensor.rf_dict = input_tensor.rf_dict
            return output

        if self.layer_idx == 0:
            input_tensor = args[0]
            input_height, input_width = input_tensor.shape[-2:]
            input_rf_dict = {}
            for i in range(input_height):
                for j in range(input_width):
                    input_rf_dict[i * input_width + j] = torch.zeros(input_height, input_width)
                    input_rf_dict[i * input_width + j][i, j] = 1
            input_tensor.rf_dict = input_rf_dict

        # index_mapping = self._map_feature_map(op_name, args, output)
        if op_name not in self.handlers:
            raise NotImplementedError(f"Operation {op_name} not supported")
        index_mapping = self.handlers[op_name].handle(args, output)
        rf_dict = self._compute_rf_tensor(args, index_mapping)
        output = self._update_output(op_name, output, rf_dict)
        self.layer_idx += 1

        if self.verbose:
            output_tensor = output
            if isinstance(output, tuple):
                output_tensor = output[0]
            output_height, output_width = output_tensor.shape[-2:]
            middel_index = output_height // 2 * output_width + output_width // 2
            first_row, last_row, first_col, last_col = self._find_nonzero_bounds(output_tensor.rf_dict[middel_index])
            logger.info(f"Layer {self.layer_idx}: {op_name}, RF: {output_tensor.rf_dict[middel_index][first_row:last_row+1, first_col:last_col+1]}, RF Shape: {output_tensor.rf_dict[middel_index][first_row:last_row+1, first_col:last_col+1].shape}")
        return output

    # def _map_feature_map(self, op_name, args, output):
    #     mapping = {}

    #     if op_name.startswith('aten::conv'):
    #         input_tensor = args[0]
    #         if isinstance(input_tensor, tuple):
    #             input_tensor = input_tensor[0]
    #         weight, stride, padding = args[1], args[3], args[4]
    #         kernel_height, kernel_width = weight.shape[-2:]

    #         input_height, input_width = input_tensor.shape[-2:]
    #         output_height, output_width = output.shape[-2:]
    #         assert output_height == (input_height + 2 * padding[0] - kernel_height) // stride[0] + 1, f"Output height mismatch: {output_height}, {input_height}, {padding[0]}, {kernel_height}, {stride[0]}"
    #         assert output_width == (input_width + 2 * padding[1] - kernel_width) // stride[1] + 1, f"Output width mismatch: {output_width}, {input_width}, {padding[1]}, {kernel_width}, {stride[1]}"

    #         for h_out in range(output_height):
    #             for w_out in range(output_width):
    #                 # Start position in the padded tensor
    #                 h_padded_start = h_out * stride[0]
    #                 w_padded_start = w_out * stride[1]

    #                 # Start position in the input tensor
    #                 h_start = h_padded_start - padding[0]
    #                 w_start = w_padded_start - padding[1]

    #                 input_indices = []
    #                 for kh in range(kernel_height):
    #                     h_in = h_start + kh
    #                     if 0 <= h_in < input_height:
    #                         for kw in range(kernel_width):
    #                             w_in = w_start + kw
    #                             if 0 <= w_in < input_width:
    #                                 index = h_in * input_width + w_in
    #                                 input_indices.append(index)
    #                 output_index = h_out * output_width + w_out
    #                 mapping[output_index] = input_indices
    #         return mapping

    #     elif op_name.startswith('aten::add'):
    #         input_tensor_1 = args[0]
    #         if isinstance(input_tensor_1, tuple):
    #             input_tensor_1 = input_tensor_1[0]
    #         input_tensor_2 = args[1]
    #         if isinstance(input_tensor_2, tuple):
    #             input_tensor_2 = input_tensor_2[0]
    #         # FIXME: Add broadcast support
    #         assert input_tensor_1.shape == input_tensor_2.shape, f"Addition shape mismatch: {input_tensor_1.shape}, {input_tensor_2.shape}"
    #         input_height, input_width = input_tensor_1.shape[-2:]
    #         for h in range(input_height):
    #             for w in range(input_width):
    #                 index = h * input_width + w
    #                 mapping[index] = [index]
    #         return mapping
    #     elif op_name.startswith('aten::relu') or op_name.startswith('aten::gelu'):              # activation
    #         input_tensor = args[0]
    #         if isinstance(input_tensor, tuple):
    #             input_tensor = input_tensor[0]
    #         assert input_tensor.shape == output.shape, f"Activation shape mismatch: {input_tensor.shape}, {output.shape}"
    #         input_height, input_width = input_tensor.shape[-2:]

    #         for h in range(input_height):
    #             for w in range(input_width):
    #                 index = h * input_width + w
    #                 mapping[index] = [index]
    #         return mapping

    #     elif op_name.startswith('aten::max_pool2d'):
    #         input_tensor = args[0]
    #         if isinstance(input_tensor, tuple):
    #             input_tensor = input_tensor[0]
    #         kernel_height, kernel_width = args[1]
    #         stride_height, stride_width = args[2]
    #         input_height, input_width = input_tensor.shape[-2:]
    #         output_height, output_width = output[0].shape[-2:]
    #         assert input_height == output_height * stride_height + kernel_height - stride_height, f"MaxPool height mismatch: {input_height}, {output_height}, {stride_height}, {kernel_height}"
    #         assert input_width == output_width * stride_width + kernel_width - stride_width, f"MaxPool width mismatch: {input_width}, {output_width}, {stride_width}, {kernel_width}"

    #         for h_out in range(output_height):
    #             for w_out in range(output_width):
    #                 h_start = h_out * stride_height
    #                 w_start = w_out * stride_width
    #                 output_index = h_out * output_width + w_out
    #                 input_indices = []
    #                 for kh in range(kernel_height):
    #                     for kw in range(kernel_width):
    #                         index = (h_start + kh) * input_width + (w_start + kw)
    #                         input_indices.append(index)
    #                 mapping[output_index] = input_indices
    #         return mapping
    #     else:
    #         raise NotImplementedError(f"Operation {op_name} not supported")

    def _compute_rf_tensor(self, args, index_mapping):
        input_tensor = args[0]
        assert hasattr(input_tensor, 'rf_dict'), "Input tensor should have rf_dict"

        rf_dict = {}
        for output_index, input_indices in index_mapping.items():
            rf_tensor = torch.zeros_like(input_tensor.rf_dict[0])
            for input_index in input_indices:
                rf_tensor += input_tensor.rf_dict[input_index]
            rf_dict[output_index] = rf_tensor
        return rf_dict
    
    def _update_output(self, op_name, output, rf_dict):
        if op_name == 'aten::max_pool2d_with_indices':
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
            last_row = tensor.size(0) - torch.argmax(rows_nonzero.flip(0).float()).item() - 1  # 最后一个非全零行
        
        # 检查是否有非全零的列
        cols_nonzero = tensor.any(dim=0)
        if not cols_nonzero.any():
            first_col = None
            last_col = None
        else:
            first_col = torch.argmax(cols_nonzero.float()).item()  # 第一个非全零列
            last_col = tensor.size(1) - torch.argmax(cols_nonzero.flip(0).float()).item() - 1  # 最后一个非全零列
        
        return first_row, last_row, first_col, last_col