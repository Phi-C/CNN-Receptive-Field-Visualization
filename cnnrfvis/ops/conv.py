from cnnrfvis.ops.utils import OpHandler


class ConvHandler(OpHandler):
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def handle(self, args, output):
        input_tensor = args[0][0] if isinstance(args[0], tuple) else args[0]
        weight, stride, padding = args[1], args[3], args[4]
        kernel_height, kernel_width = weight.shape[-2:]
        input_height, input_width = input_tensor.shape[-2:]
        output_height, output_width = output.shape[-2:]

        assert (
            output_height
            == (input_height + 2 * padding[0] - kernel_height) // stride[0] + 1
        ), f"Output height mismatch: {output_height}, {input_height}, {padding[0]}, {kernel_height}, {stride[0]}"
        assert (
            output_width
            == (input_width + 2 * padding[1] - kernel_width) // stride[1] + 1
        ), f"Output width mismatch: {output_width}, {input_width}, {padding[1]}, {kernel_width}, {stride[1]}"

        mapping = {}
        for h_out in range(output_height):
            for w_out in range(output_width):
                # Start position in the padded tensor
                h_padded_start = h_out * stride[0]
                w_padded_start = w_out * stride[1]

                # Start position in the input tensor
                h_start = h_padded_start - padding[0]
                w_start = w_padded_start - padding[1]

                input_indices = []
                for kh in range(kernel_height):
                    h_in = h_start + kh
                    if 0 <= h_in < input_height:
                        for kw in range(kernel_width):
                            w_in = w_start + kw
                            if 0 <= w_in < input_width:
                                index = h_in * input_width + w_in
                                input_indices.append(index)
                output_index = h_out * output_width + w_out
                mapping[output_index] = input_indices
        return mapping
