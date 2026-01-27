import torch
import logging
import numpy as np
from typing import List, Dict, Tuple
from mohollm.benchmarks.benchmark import BENCHMARK
from botorch.test_functions.multi_objective import VehicleSafety

logger = logging.getLogger("VehicleSafety")


class VehicleSafetyNoiseBenchmark(BENCHMARK):
    def __init__(self, model_name, seed):
        self.benchmark_name = "VehicleSafetyNoise"
        self.n_dim = 5
        self.n_obj = 3
        self.nadir_point = [100, 100, 100]  # Example values, adjust as needed
        self.ideal_point = [0, 0, 0]  # Example values, adjust as needed
        self.seed = seed
        self.metrics = ["f1", "f2", "f3"]
        self.model_name = model_name
        self.problem_id = "vehicle_safety_noise"
        self.problem = VehicleSafety()
        self.bounds = {
            "x0": (1.0, 3.0),
            "x1": (1.0, 3.0),
            "x2": (1.0, 3.0),
            "x3": (1.0, 3.0),
            "x4": (1.0, 3.0),
        }
        # approximated by sampling 250000 samples
        self.OBJECTIVE_RANGE = {
            "F1": 38.838667,
            "F2": 5.373361,
            "F3": 0.214417,
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
        return point, self.add_noise(
            {
                "F1": round(f_tensor[0, 0].item(), 3),
                "F2": round(f_tensor[0, 1].item(), 3),
                "F3": round(f_tensor[0, 2].item(), 3),
            }
        )

    def add_noise(self, y_values):
        """
        Adds zero-mean Gaussian noise with std dev = 1% of the objective range.
        """
        y_noisy = {}
        for obj, value in y_values.items():
            # Retrieve the range for the current objective
            obj_range = self.OBJECTIVE_RANGE.get(
                obj, 1.0
            )  # Default to 1.0 if not found

            # Define standard deviation as 1% of the range
            sigma = 0.01 * obj_range

            # Generate zero-mean Gaussian noise
            noise = np.random.normal(loc=0.0, scale=sigma)

            # Add noise to the original value
            y_noisy[obj] = value + noise

        return y_noisy

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
        except Exception:
            return False

    def is_valid_evaluation(self, evaluation) -> bool:
        return True
