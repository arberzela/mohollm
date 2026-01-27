from typing import List, Dict, Tuple
from mohollm.benchmarks.benchmark import BENCHMARK
from scipy.stats.qmc import Sobol


class SchafferN2(BENCHMARK):
    """
    Schaffer Function N. 2 Benchmark for Multi-Objective Optimization.
    https://en.wikipedia.org/wiki/Test_functions_for_optimization

    Note: For this benchmark we drop the constraints

    Search domain: -5 <= x <= 10
    """

    def __init__(self, model_name: str, seed: int):
        self.benchmark_name = "SchafferN2"
        self.problem_id = "schaffer_n2"
        self.metrics = ["F1", "F2"]
        self.seed = seed
        self.model_name = model_name
        self.n_dim = 1
        self.n_obj = 2

    def generate_initialization(self, n_points: int, **kwargs) -> List[Dict]:
        sobol = Sobol(d=1, scramble=True, seed=self.seed)
        points = sobol.random(n=n_points)
        result = []
        for x in points:
            result.append({"x": round(x[0] * 15 - 5, 3)})
        return result

    def evaluate_point(self, point: Dict, **kwargs) -> Tuple[Dict, Dict]:
        x = point["x"]

        if x <= 1:
            F1 = -x
        elif 1 < x <= 3:
            F1 = x - 2
        elif 3 < x <= 4:
            F1 = 4 - x
        else:
            F1 = x - 4

        F2 = (x - 5) ** 2

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
            "F1": [-5, 6],  # Example range for the first objective
            "F2": [0, 225],  # Example range for the second objective
        }

    def is_valid_candidate(self, candidate: Dict) -> bool:
        return "x" in candidate

    def is_valid_evaluation(self, evaluation: Dict) -> bool:
        return "F1" in evaluation and "F2" in evaluation
