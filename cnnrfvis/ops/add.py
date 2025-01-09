from cnnrfvis.ops.utils import OpHandler


class AddHandler(OpHandler):
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def handle(self, args, output):
        input_tensor_1 = args[0][0] if isinstance(input_tensor_1, tuple) else args[0]
        input_tensor_2 = args[1][0] if isinstance(input_tensor_2, tuple) else args[1]

        # FIXME: Add broadcast support
        assert (
            input_tensor_1.shape == input_tensor_2.shape
        ), f"Addition shape mismatch: {input_tensor_1.shape}, {input_tensor_2.shape}"

        input_height, input_width = input_tensor_1.shape[-2:]

        mapping = {}
        for h in range(input_height):
            for w in range(input_width):
                index = h * input_width + w
                mapping[index] = [index]
        return mapping
