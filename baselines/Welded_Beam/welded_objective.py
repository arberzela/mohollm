import logging
import argparse
import numpy as np
from syne_tune import Reporter
from pymoo.problems import get_problem

report = Reporter()


def evaluate_point(problem, point):
    f = problem.evaluate(np.array(list(point.values())))
    f0 = float(f[0][0])
    f1 = float(f[0][1])
    f0 = np.clip(f0, -1e6, 1e6)
    f1 = np.clip(f1, -1e6, 1e6)
    return {"F1": round(f0, 3), "F2": round(f1, 3)}


def objective(problem: str, config: dict):
    problem = get_problem(problem)
    del config["problem"]
    point = {f"x{i}": config[f"x{i}"] for i in range(problem.n_var)}
    evaluation = evaluate_point(problem, point)
    report(**evaluation)


if __name__ == "__main__":
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", type=str)
    for i in range(30):
        parser.add_argument(f"--x{i}", type=float)
    args, _ = parser.parse_known_args()
    objective(problem=args.problem, config=vars(args))
