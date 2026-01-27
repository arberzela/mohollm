import os
import logging
import numpy as np
from typing import List, Dict, Tuple
from mohollm.benchmarks.benchmark import BENCHMARK
try:
    from pymoo.factory import get_problem
except ImportError as e:
    raise ImportError(
        "Optional dependency 'pymoo' is required for ZDT benchmarks.\n"
        "Install it with `pip install -e .[dev]` or `pip install pymoo` and retry."
    ) from e


logger = logging.getLogger("ZDT")


class ZDT(BENCHMARK):
    def __init__(self, model_name, seed, problem_id="zdt1"):
        self.benchmark_name = "ZDT"
        self.problem_id = problem_id
        self.metrics = ["f1", "f2"]
        self.problem = get_problem(problem_id)
        self.seed = seed
        self.model_name = model_name

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

    def save_progress(self, statistics: Dict) -> None:
        """
        Save progress statistics to a file after each trial. Implement this
        function if you want to save the results after each trial to a
        specific directory or file.

        Args:
            statistics (Dict): Dictionary containing benchmark statistics to be saved

        Example:
            save_progress({'accuracy': 0.95, 'loss': 0.1})
        """
        logger.debug(
            f"Saving progress {statistics}",
        )
        for key, statistic in statistics.items():
            fval_dir = f"./results/{self.problem_id.upper()}/{self.method_name}/{key}/"
            fval_filename = f"{self.model_name}_{self.problem_id}_{self.seed}.csv"
            os.makedirs(fval_dir, exist_ok=True)
            statistic.to_csv(f"{fval_dir}/{fval_filename}", index=False)
            logger.debug(f"Writing {key} to {fval_dir}/{fval_filename}")
