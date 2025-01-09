from cnnrfvis.ops.utils import OpHandler


class ActivationHandler(OpHandler):
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def handle(self, args, output):
        input_tensor = args[0][0] if isinstance(args[0], tuple) else args[0]

        assert (
            input_tensor.shape == output.shape
        ), f"Activation shape mismatch: {input_tensor.shape}, {output.shape}"
        input_height, input_width = input_tensor.shape[-2:]

        mapping = {}
        for h in range(input_height):
            for w in range(input_width):
                index = h * input_width + w
                mapping[index] = [index]

        return mapping
