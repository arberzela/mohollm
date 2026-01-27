from typing import List, Dict, Tuple
from mohollm.benchmarks.benchmark import BENCHMARK
from scipy.stats.qmc import Sobol


class Simple2D(BENCHMARK):
    def __init__(self, model_name: str, seed: int):
        self.benchmark_name = "Simple2D"
        self.problem_id = "simple_2d"
        self.metrics = ["F1", "F2"]
        self.seed = seed
        self.model_name = model_name

    def generate_initialization(self, n_points: int, **kwargs) -> List[Dict]:
        sobol = Sobol(d=2, scramble=True, seed=self.seed)
        points = sobol.random(n=n_points)
        result = []
        for x in points:
            result.append(
                {"x1": round(x[0] * 20 - 10, 3), "x2": round(x[1] * 20 - 10, 3)}
            )
        return result

    def evaluate_point(self, point: Dict, **kwargs) -> Tuple[Dict, Dict]:
        x1, x2 = point["x1"], point["x2"]
        F1 = x1**2 + x2**2  # First objective: minimize sum of squares
        F2 = x1**2 + x2**2  # Second objective: minimize sum of squares
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
            "F1": [0, 200],  # Example range for the first objective
            "F2": [0, 200],  # Example range for the second objective
        }

    def is_valid_candidate(self, candidate: Dict) -> bool:
        return "x1" in candidate and "x2" in candidate

    def is_valid_evaluation(self, evaluation: Dict) -> bool:
        return "F1" in evaluation and "F2" in evaluation
