import logging
import torch
import numpy as np
from typing import List, Dict, Tuple
from mohollm.benchmarks.benchmark import BENCHMARK
from botorch.test_functions.multi_objective import CarSideImpact

logger = logging.getLogger("CarSideImpact")


class CarSideImpactBenchmark(BENCHMARK):
    def __init__(self, model_name, seed):
        self.benchmark_name = "CarSideImpact"
        self.n_dim = 7
        self.n_obj = 4
        self.seed = seed
        self.metrics = ["f1", "f2", "f3", "f4"]
        self.model_name = model_name
        self.problem_id = "car_side_impact"
        self.problem = CarSideImpact()
        self.bounds = {
            "x0": (0.5, 1.5),
            "x1": (0.45, 1.35),
            "x2": (0.5, 1.5),
            "x3": (0.5, 1.5),
            "x4": (0.875, 2.625),
            "x5": (0.4, 1.2),
            "x6": (0.4, 1.2),
        }

    def evaluate_point(self, point, **kwargs) -> Dict:
        # Convert the dictionary of values to a torch tensor
        # BoTorch expects a 2D tensor of shape (batch_size, n_dim)

        # NOTE: It is very important to use double precision (float64) here
        # to match the precision used in BoTorch otherwise points on the boundary
        # may not be evaluated correctly due to floating point precision issues :(
        x_tensor = torch.tensor(list(point.values()), dtype=torch.float64).view(1, -1)
        logger.debug(f"Evaluating point (dict): {point}")
        logger.debug(f"Evaluating point (tensor): {x_tensor}")
        f_tensor = self.problem.evaluate_true(x_tensor)
        logger.debug(f"Evaluation result: {f_tensor}")
        # Extract scalar values from the output tensor using .item()
        return point, {
            "F1": round(f_tensor[0, 0].item(), 3),
            "F2": round(f_tensor[0, 1].item(), 3),
            "F3": round(f_tensor[0, 2].item(), 3),
            "F4": round(f_tensor[0, 3].item(), 3),
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

    def is_valid_evaluation(self, evaluation) -> bool:
        return True
