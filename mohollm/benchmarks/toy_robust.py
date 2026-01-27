import os
import torch
import logging
import numpy as np
from typing import List, Dict, Tuple
from mohollm.benchmarks.benchmark import BENCHMARK
from botorch.test_functions.multi_objective import ToyRobust

logger = logging.getLogger("ToyRobust")


class ToyRobustBenchmark(BENCHMARK):
    def __init__(self, model_name, seed):
        self.benchmark_name = "ToyRobust"
        self.n_dim = 1
        self.n_obj = 2
        self.nadir_point = [100, 100]  # Example values, adjust as needed
        self.ideal_point = [0, 0]  # Example values, adjust as needed
        self.seed = seed
        self.metrics = ["f1", "f2"]
        self.model_name = model_name
        self.problem_id = "toy_robust"
        self.problem = ToyRobust()

    def evaluate_point(self, point, **kwargs) -> Dict:
        # Convert the dictionary of values to a torch tensor
        # BoTorch expects a 2D tensor of shape (batch_size, n_dim)
        x_tensor = torch.tensor(list(point.values())).view(1, -1)

        # Use evaluate_true to get the noise-free objective values
        f_tensor = self.problem.evaluate_true(x_tensor)

        # Extract scalar values from the output tensor using .item()
        return point, {
            "F1": round(f_tensor[0, 0].item(), 3),
            "F2": round(f_tensor[0, 1].item(), 3),
        }

    def generate_initialization(self, n_points: int, **kwargs) -> List[Dict]:
        points = []
        for i in range(n_points):
            x = np.random.uniform(0, 0.7, self.n_dim)
            points.append(
                {f"x{i}": round(value, 3) for i, value in enumerate(x.tolist())}
            )
        return points

    def get_few_shot_samples(self, **kwargs) -> List[Tuple[Dict, Dict]]:
        samples = []
        for _ in range(3):
            point = self.generate_initialization(1)
            point, evaluation = self.evaluate_point(point[0])
            samples.append((point, evaluation))
        return samples

    def get_metrics_ranges(self, **kwargs) -> Dict[str, List[float]]:
        pass

    def is_valid_candidate(self, candidate) -> bool:
        if len(candidate.keys()) != self.n_dim:
            return False
        return True

    def is_valid_evaluation(self, evaluation) -> bool:
        return True
