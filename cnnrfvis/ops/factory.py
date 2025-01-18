import re
from cnnrfvis.ops.conv import ConvHandler
from cnnrfvis.ops.pool import MaxPoolHandler
from cnnrfvis.ops.add import AddHandler
from cnnrfvis.ops.mul import MulHandler
from cnnrfvis.ops.activation import ActivationHandler
from cnnrfvis.ops.norm import NormlizationHandler


class HandleFactory:
    @staticmethod
    def get_handlers():
        return {
            "aten::convolution": ConvHandler(),
            "aten::max_pool2d_with_indices": MaxPoolHandler(),
            "aten::add": AddHandler(),
            # TODO: Refact, activation can be processed in the same way
            "aten::relu": ActivationHandler(),
            "aten::gelu": ActivationHandler(),
            "aten::sigmoid": ActivationHandler(),
            "aten::native_batch_norm": NormlizationHandler(),
            "aten::tanh": ActivationHandler(),
            "aten::mul": MulHandler(),
            # re.compile(r'aten::(relu|relu|sigmoid)'): ActivationHandler(),
        }
