import logging
import argparse
from syne_tune import Reporter
from mohollm.benchmarks.simple_2d_mo.poloni import Poloni

report = Reporter()


def evaluate_point(problem, point):
    evaluation = problem.evaluate_point(point)[1]
    return {"f1": evaluation["F1"], "f2": evaluation["F2"]}


def objective(config: dict):
    problem = Poloni(model_name="baseline", seed=42)
    point = {"x": config["x"], "y": config["y"]}
    evaluation = evaluate_point(problem, point)
    report(**evaluation)


if __name__ == "__main__":
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--x", type=float)
    parser.add_argument("--y", type=float)
    args, _ = parser.parse_known_args()
    objective(config=vars(args))
