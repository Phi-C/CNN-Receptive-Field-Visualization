"""
This module contains the abstract class for operation handlers, it defines the interface for handling operations in the module operation tracker.
"""

from abc import ABC, abstractmethod
import torch


class OpHandler(ABC):
    @abstractmethod
    def handle(self, args, output):
        """Abstract method to handle the operation"""
        pass

    def compute_rf_tensor(slef, args, idx_map):
        """ compute receptive field for each pixel on output feat map
        """
        input_tensor = args[0]
        assert hasattr(input_tensor, "rf_dict"), "Input Tensor Should Have rf_dict"

        rf_dict = {}
        for output_index, input_indices in idx_map.items():
            rf_tensor = torch.zeros_like(input_tensor.rf_dict[0])
            for input_index in input_indices:
                rf_tensor += input_tensor.rf_dict[input_index]
            rf_dict[output_index] = rf_tensor
        return rf_dict
