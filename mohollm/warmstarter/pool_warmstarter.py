import logging
from typing import List, Dict
import pandas as pd

from mohollm.warmstarter.warmstarter import WARMSTARTER


logger = logging.getLogger("POOL_WARMSTARTER")


class POOL_WARMSTARTER(WARMSTARTER):
    def __init__(self):
        super().__init__()
        self.good_pool_path = (
            "./mohollm/warmstarter/pool_data/custom_poloni_good_pool.csv"
        )
        self.bad_pool_path = "./mohollm/warmstarter/pool_data/custom_poloni_bad_pool.csv"

    def generate_initialization(self) -> List[Dict]:
        if not self.initial_samples:
            return []

        beta = self.warmstarter_settings.get("beta", None)
        logger.debug(f"Found beta: {beta} for the initial configurations generation")
        if beta is None:
            raise ValueError(
            "No beta value provided in the warmstarter settings for pool warmstarter"
            )
        logger.debug(f"Using beta: {beta} for the initial configurations generation")
        n = int(self.initial_samples)
        beta = float(beta)
        beta = max(0.0, min(1.0, beta))
        n_bad = int(round(n * beta))
        n_good = n - n_bad

        try:
            good = pd.read_csv(self.good_pool_path)[["x", "y"]]
        except Exception:
            good = pd.DataFrame(columns=["x", "y"])

        try:
            bad = pd.read_csv(self.bad_pool_path)[["x", "y"]]
        except Exception:
            bad = pd.DataFrame(columns=["x", "y"])

        def sample(df, k):
            if k <= 0 or df.empty:
                return pd.DataFrame(columns=["x", "y"])
            return df.sample(n=k, replace=(k > len(df)))

        combined = pd.concat(
            [sample(good, n_good), sample(bad, n_bad)], ignore_index=True
        )
        if combined.empty:
            return []
        if len(combined) < n:
            combined = pd.concat(
                [combined, combined.sample(n=(n - len(combined)), replace=True)],
                ignore_index=True,
            )

        rows = combined.head(n).to_dict(orient="records")
        return [
            {"x": round(float(r["x"]), 3), "y": round(float(r["y"]), 3)} for r in rows
        ]
