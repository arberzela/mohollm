from typing import List, Dict, Tuple
from mohollm.benchmarks.benchmark import BENCHMARK
from scipy.stats.qmc import Sobol
import math


class Poloni(BENCHMARK):
    """
    Poloni's Two Objective Function Benchmark for Multi-Objective Optimization.
    https://en.wikipedia.org/wiki/Test_functions_for_optimization

    Search domain: -pi <= x, y <= pi
    """

    def __init__(self, model_name: str, seed: int):
        self.benchmark_name = "Poloni"
        self.problem_id = "poloni"
        self.metrics = ["F1", "F2"]
        self.seed = seed
        self.model_name = model_name
        self.n_dim = 2
        self.n_obj = 2

    def generate_initialization(self, n_points: int, **kwargs) -> List[Dict]:
        sobol = Sobol(d=2, scramble=True, seed=self.seed)
        points = sobol.random(n=n_points)
        result = []
        for x in points:
            result.append(
                {
                    "x": round(x[0] * 2 * math.pi - math.pi, 3),
                    "y": round(x[1] * 2 * math.pi - math.pi, 3),
                }
            )
        return result

    def evaluate_point(self, point: Dict, **kwargs) -> Tuple[Dict, Dict]:
        x, y = point["x"], point["y"]

        A1 = 0.5 * math.sin(1) - 2 * math.cos(1) + math.sin(2) - 1.5 * math.cos(2)
        A2 = 1.5 * math.sin(1) - math.cos(1) + 2 * math.sin(2) - 0.5 * math.cos(2)

        B1 = 0.5 * math.sin(x) - 2 * math.cos(x) + math.sin(y) - 1.5 * math.cos(y)
        B2 = 1.5 * math.sin(x) - math.cos(x) + 2 * math.sin(y) - 0.5 * math.cos(y)

        F1 = 1 + (A1 - B1) ** 2 + (A2 - B2) ** 2
        F2 = (x + 3) ** 2 + (y + 1) ** 2

        return point, {"F1": round(F1, 3), "F2": round(F2, 3)}

    def get_few_shot_samples(self, **kwargs) -> List[Tuple[Dict, Dict]]:
        samples = []
        for _ in range(3):
            point = self.generate_initialization(1)[0]
            point, evaluation = self.evaluate_point(point)
            samples.append((point, evaluation))
        return samples

    def get_metrics_ranges(self, **kwargs) -> Dict[str, List[float]]:
        return {
            "F1": [1, 100],  # Example range for the first objective
            "F2": [0, 100],  # Example range for the second objective
        }

    def is_valid_candidate(self, candidate: Dict) -> bool:
        return "x" in candidate and "y" in candidate

    def is_valid_evaluation(self, evaluation: Dict) -> bool:
        return "F1" in evaluation and "F2" in evaluation
