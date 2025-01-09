"""
This module contains the abstract class for operation handlers, it defines the interface for handling operations in the module operation tracker.
"""

from abc import ABC, abstractmethod


class OpHandler(ABC):
    @abstractmethod
    def handle(self, args, output):
        """Abstract method to handle the operation"""
        pass
