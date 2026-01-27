import os
import torch
import logging
import numpy as np
from typing import List, Dict, Tuple
from mohollm.benchmarks.benchmark import BENCHMARK
from botorch.test_functions.multi_objective import Penicillin

logger = logging.getLogger("Penicillin")


class PenicillinBenchmark(BENCHMARK):
    def __init__(self, model_name, seed):
        self.benchmark_name = "Penicillin"
        self.n_dim = 7
        self.n_obj = 3
        self.nadir_point = [100, 100, 100]  # Example values, adjust as needed
        self.ideal_point = [0, 0, 0]  # Example values, adjust as needed
        self.seed = seed
        self.metrics = ["f1", "f2", "f3"]
        self.model_name = model_name
        self.problem_id = "penicillin"
        self.problem = Penicillin()
        self.bounds = {
            "x0": (60.0, 120.0),
            "x1": (0.05, 18.0),
            "x2": (293.0, 303.0),
            "x3": (0.05, 18.0),
            "x4": (0.01, 0.5),
            "x5": (500.0, 700.0),
            "x6": (5.0, 6.5),
        }

    def evaluate_point(self, point, **kwargs) -> Dict:
        # Convert the dictionary of values to a torch tensor
        # BoTorch expects a 2D tensor of shape (batch_size, n_dim)

        # NOTE: It is very important to use double precision (float64) here
        # to match the precision used in BoTorch otherwise points on the boundary
        # may not be evaluated correctly due to floating point precision issues :(
        x_tensor = torch.tensor(list(point.values()), dtype=torch.float64).view(1, -1)

        # Use evaluate_true to get the noise-free objective values
        f_tensor = self.problem.evaluate_true(x_tensor)

        # Extract scalar values from the output tensor using .item()
        return point, {
            "F1": round(f_tensor[0, 0].item(), 3),
            "F2": round(f_tensor[0, 1].item(), 3),
            "F3": round(f_tensor[0, 2].item(), 3),
        }

    def generate_initialization(self, n_points: int, **kwargs) -> List[Dict]:
        points = []
        for _ in range(n_points):
            # Create a dictionary for the new point
            new_point = {}
            # Iterate through the defined bounds for each variable
            for key, (lower_bound, upper_bound) in self.bounds.items():
                # Generate a random value within the specific bounds
                value = np.random.uniform(lower_bound, upper_bound)
                new_point[key] = round(value, 3)
            points.append(new_point)
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
        try:
            self.evaluate_point(candidate)
            return True
        except Exception as e:
            return False
        # if len(candidate.keys()) != self.n_dim:
        #     return False
        # return True

    def is_valid_evaluation(self, evaluation) -> bool:
        return True
