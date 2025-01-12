import re
from cnnrfvis.ops.conv import ConvHandler
from cnnrfvis.ops.pool import MaxPoolHandler
from cnnrfvis.ops.add import AddHandler
from cnnrfvis.ops.activation import ActivationHandler
from cnnrfvis.ops.norm import NormlizationHandler


class HandleFactory:
    @staticmethod
    def get_handlers():
        return {
            "aten::convolution": ConvHandler(),
            "aten::max_pool2d_with_indices": MaxPoolHandler(),
            "aten::add": AddHandler(),
            "aten::relu_": ActivationHandler(),
            "aten::relu": ActivationHandler(),
            "aten::gelu": ActivationHandler(),
            "aten::sigmoid": ActivationHandler(),
            "aten::native_batch_norm": NormlizationHandler(),
            # re.compile(r'aten::(relu|relu|sigmoid)'): ActivationHandler(),
        }
