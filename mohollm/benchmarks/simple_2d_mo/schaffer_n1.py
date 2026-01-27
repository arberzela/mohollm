from typing import List, Dict, Tuple
from mohollm.benchmarks.benchmark import BENCHMARK
from scipy.stats.qmc import Sobol


class SchafferN1(BENCHMARK):
    """
    Schaffer Function N. 1 Benchmark for Multi-Objective Optimization.
    https://en.wikipedia.org/wiki/Test_functions_for_optimization

    Note: With increasing A, the problem difficulty rises.

    Search domain: -A <= x <= A
    """

    def __init__(self, model_name: str, seed: int, A: int = 10):
        self.benchmark_name = "SchafferN1"
        self.problem_id = "schaffer_n1"
        self.metrics = ["F1", "F2"]
        self.seed = seed
        self.model_name = model_name
        self.A = A
        self.n_dim = 1
        self.n_obj = 2

    def generate_initialization(self, n_points: int, **kwargs) -> List[Dict]:
        sobol = Sobol(d=1, scramble=True, seed=self.seed)
        points = sobol.random(n=n_points)
        result = []
        for x in points:
            result.append({"x": round(x[0] * 2 * self.A - self.A, 3)})
        return result

    def evaluate_point(self, point: Dict, **kwargs) -> Tuple[Dict, Dict]:
        x = point["x"]
        F1 = x**2
        F2 = (x - 2) ** 2

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
            "F1": [0, self.A**2],  # Example range for the first objective
            "F2": [0, (self.A + 2) ** 2],  # Example range for the second objective
        }

    def is_valid_candidate(self, candidate: Dict) -> bool:
        return "x" in candidate

    def is_valid_evaluation(self, evaluation: Dict) -> bool:
        return "F1" in evaluation and "F2" in evaluation
