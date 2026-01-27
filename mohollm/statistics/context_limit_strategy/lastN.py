from typing import Dict, List, Tuple, Any
from mohollm.statistics.context_limit_strategy.context_limit_strategy import (
    ContextLimitStrategy,
)


class LastN(ContextLimitStrategy):

    def __init__(self):
        super().__init__()

    def update_context(
        self, configs: List[Dict], fvals: List[Dict]
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Updates the context by keeping the last N configurations and function values.

        Args:
            configs (List[Dict]): The list of all configurations.
            fvals (List[Dict]): The list of all function values.

        Returns:
            Tuple[List[Dict], List[Dict]]: The updated context configurations and function values.
        """
        context_configs = configs[-self.max_context_configs :]
        context_fvals = fvals[-self.max_context_configs :]
        return context_configs, context_fvals
