import logging
import pandas as pd
from typing import List, Dict
from mohollm.warmstarter.warmstarter import WARMSTARTER


logger = logging.getLogger("CHECKPOINT_WARMSTARTER")


class CHECKPOINT_WARMSTARTER(WARMSTARTER):
    def __init__(
        self,
    ):
        super().__init__()
        self.file_path = "./baselines/ZDT/gpt-4o-mini-tot-warmstart-backup/observed_configs/gpt-4o-mini-tot-warmstart_zdt3_42.csv"
        self.num_samples = 20

    def generate_initialization(self) -> List[Dict]:
        initial_samples = []
        try:
            initial_samples = pd.read_csv(self.file_path).to_dict(orient="records")
        except FileNotFoundError:
            logger.error(
                f"File {self.file_path} not found. Make sure the file exists and the path is correct."
            )
        return initial_samples
