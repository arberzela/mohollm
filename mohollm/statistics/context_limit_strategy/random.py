import numpy as np
from typing import Dict, List, Tuple, Any
from mohollm.statistics.context_limit_strategy.context_limit_strategy import (
    ContextLimitStrategy,
)


class Random(ContextLimitStrategy):
    def __init__(self):
        super().__init__()

    def update_context(
        self, configs: List[Dict], fvals: List[Dict]
    ) -> Tuple[List[Dict], List[Dict]]:
        configs_array = np.array(configs)
        fvals_array = np.array(fvals)

        num_configs_to_select = min(len(configs), self.max_context_configs)
        selected_indices = np.random.choice(
            len(configs), num_configs_to_select, replace=False
        )
        context_configs = configs_array[selected_indices]
        context_fvals = fvals_array[selected_indices]

        context_configs = context_configs.tolist()
        context_fvals = context_fvals.tolist()

        return context_configs, context_fvals
