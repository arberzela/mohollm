import logging
import argparse
import numpy as np
from syne_tune import Reporter
from pymoo.problems import get_problem
import sys


report = Reporter()


def evaluate_point(problem, point):
    f = problem.evaluate(np.array(list(point.values()))).tolist()
    return {"F1": round(float(f[0]), 3)}


def objective(problem: str, config: dict):
    problem = get_problem(problem, n_var=10)
    point = {f"x{i}": config[f"x{i}"] for i in range(problem.n_var)}
    evaluation = evaluate_point(problem, point)
    report(**evaluation)


if __name__ == "__main__":
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # Convert space-separated arguments to equals format
    modified_args = []
    i = 1  # Start from 1 to skip script name

    while i < len(sys.argv):
        current_arg = sys.argv[i]

        # Check if this is a parameter followed by a value
        if (
            current_arg.startswith("--")
            and i + 1 < len(sys.argv)
            and not sys.argv[i + 1].startswith("--")
        ):
            # Combine parameter and value with equals sign
            combined_arg = f"{current_arg}={sys.argv[i + 1]}"
            modified_args.append(combined_arg)
            i += 2  # Skip both parameter and value
        else:
            # Keep argument as is
            modified_args.append(current_arg)
            i += 1

    parser = argparse.ArgumentParser()
    for i in range(10):
        parser.add_argument(f"--x{i}", type=float)
    args, _ = parser.parse_known_args(modified_args)
    objective(problem="rastrigin", config=vars(args))
