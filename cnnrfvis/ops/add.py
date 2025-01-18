from cnnrfvis.ops.utils import OpHandler
import torch


class AddHandler(OpHandler):
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def handle(self, args, output):
        input_tensor_1 = args[0][0] if isinstance(args[0], tuple) else args[0]
        input_tensor_2 = args[1][0] if isinstance(args[1], tuple) else args[1]

        # FIXME: Consider how to compute rf_dict when input's shape is different
        # assert (
        #     input_tensor_1.shape == input_tensor_2.shape
        # ), f"Addition shape mismatch: {input_tensor_1.shape}, {input_tensor_2.shape}"

        input_height, input_width = input_tensor_1.shape[-2:]

        mapping = {}
        for h in range(input_height):
            for w in range(input_width):
                index = h * input_width + w
                mapping[index] = [index]
        return mapping

    def compute_rf_tensor(self, args, idx_map):
        rf_dict = {}
        input_tensor_1 = args[0][0] if isinstance(args[0], tuple) else args[0]
        input_tensor_2 = args[1][0] if isinstance(args[1], tuple) else args[1]

        for input_tensor in args:
            if not hasattr(input_tensor, "rf_dict"):
                continue

            for output_index, input_indices in idx_map.items():
                rf_tensor = torch.zeros_like(input_tensor.rf_dict[0])
                for input_index in input_indices:
                    rf_tensor += input_tensor.rf_dict[input_index]
                if output_index in rf_dict.keys():
                    rf_dict[output_index] += rf_tensor
                else:
                    rf_dict[output_index] = rf_tensor

        return rf_dict
