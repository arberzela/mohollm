import os
import logging
import numpy as np
from typing import List, Dict, Tuple
from mohollm.benchmarks.benchmark import BENCHMARK
try:
    from pymoo.factory import get_problem
except ImportError as e:
    raise ImportError(
        "Optional dependency 'pymoo' is required for some benchmarks (kursawe).\n"
        "Install it with `pip install -e .[dev]` or `pip install pymoo` and retry."
    ) from e


logger = logging.getLogger("Kursawe")


class Kursawe(BENCHMARK):
    def __init__(self, model_name, seed):
        self.benchmark_name = "Kursawe"
        self.metrics = ["f1", "f2"]
        self.problem = get_problem("Kursawe")
        self.problem_id = "Kursawe"
        self.seed = seed
        self.model_name = model_name
        self.n_dim = 3
        self.n_obj = 2

    def generate_initialization(self, n_points: int, **kwargs) -> List[Dict]:
        points = []
        for i in range(n_points):
            x = np.random.random(self.problem.n_var)
            points.append(
                {f"x{i}": round(value, 3) for i, value in enumerate(x.tolist())}
            )
        return points

    def evaluate_point(self, point, **kwargs) -> Dict:
        f = self.problem.evaluate(np.array(list(point.values()))).tolist()
        return point, {"F1": round(float(f[0]), 3), "F2": round(float(f[1]), 3)}

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
        if len(candidate.keys()) != self.problem.n_var:
            return False
        return True

    def is_valid_evaluation(self, evaluation) -> bool:
        return True
