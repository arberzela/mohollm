from typing import Dict, List, Tuple, Any
from abc import ABC, abstractmethod


class ContextLimitStrategy(ABC):

    def __init__(self):
        self.max_context_configs: int = None

    @abstractmethod
    def update_context(
        self, configs: List[Any], fvals: List[Any]
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Updates the context with the given configurations and function values.

        Args:
            configs (List[Any]): The list of all configurations.
            fvals (List[Any]): The list of all function values.

        Returns:
            Tuple[List[Dict], List[Dict]]: The updated context configurations and function values.
        """
        pass
