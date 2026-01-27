from typing import List, Dict, Tuple
from mohollm.benchmarks.benchmark import BENCHMARK
from scipy.stats.qmc import Sobol


class TestFunction4(BENCHMARK):
    """
    Test Function 4 Benchmark for Multi-Objective Optimization without constraints.
    https://en.wikipedia.org/wiki/Test_functions_for_optimization

    Note: For this benchmark we drop the constraints

    Search domain: -7 <= x, y <= 4
    """

    def __init__(self, model_name: str, seed: int):
        self.benchmark_name = "TestFunction4"
        self.problem_id = "test_function_4"
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
            result.append({"x": round(x[0] * 11 - 7, 3), "y": round(x[1] * 11 - 7, 3)})
        return result

    def evaluate_point(self, point: Dict, **kwargs) -> Tuple[Dict, Dict]:
        x, y = point["x"], point["y"]
        F1 = x**2 - y
        F2 = -0.5 * x - y - 1

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
            "F1": [-7, 49],  # Example range for the first objective
            "F2": [-12, 4],  # Example range for the second objective
        }

    def is_valid_candidate(self, candidate: Dict) -> bool:
        return "x" in candidate and "y" in candidate

    def is_valid_evaluation(self, evaluation: Dict) -> bool:
        return "F1" in evaluation and "F2" in evaluation
