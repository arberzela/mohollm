import logging
import argparse
import numpy as np
from syne_tune import Reporter
from pymoo.problems import get_problem

report = Reporter()


def evaluate_point(problem, point):
    f = problem.evaluate(np.array(list(point.values()))).tolist()
    return {"f1": round(float(f[0]), 3), "f2": round(float(f[1]), 3)}


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
