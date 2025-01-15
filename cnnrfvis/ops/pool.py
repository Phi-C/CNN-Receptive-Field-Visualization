from cnnrfvis.ops.utils import OpHandler


class MaxPoolHandler(OpHandler):
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def handle(self, args, output):
        input_tensor = args[0][0] if isinstance(args[0], tuple) else args[0]
        kernel_height, kernel_width = args[1]
        stride_height, stride_width = args[2]
        input_height, input_width = input_tensor.shape[-2:]
        output_height, output_width = output[0].shape[-2:]
        padding_height, padding_width = 0, 0

        if len(args) >= 4:
            padding_height, padding_width = args[3]

        assert (
            output_height
            == (input_height + 2 * padding_height - kernel_height) // stride_height + 1
        ), f"MaxPool height mismatch: {input_height}, {output_height}, {stride_height}, {kernel_height}"
        assert (
            output_width
            == (input_width + 2 * padding_width - kernel_width) // stride_width + 1
        ), f"MaxPool width mismatch: {input_width}, {output_width}, {stride_width}, {kernel_width}"

        mapping = {}
        for h_out in range(output_height):
            for w_out in range(output_width):
                h_padded_start = h_out * stride_height
                w_padded_start = w_out * stride_width
                h_start = h_padded_start - padding_height
                w_start = w_padded_start - padding_width

                output_index = h_out * output_width + w_out
                input_indices = []
                for kh in range(kernel_height):
                    h_in = h_start + kh
                    if 0 <= h_in < input_height:
                        for kw in range(kernel_width):
                            w_in = w_start + kw
                            if 0 <= w_in < input_width:
                                index = (h_start + kh) * input_width + (w_start + kw)
                                input_indices.append(index)
                mapping[output_index] = input_indices
        return mapping
