import logging
import argparse
import numpy as np
from syne_tune import Reporter
from botorch.test_functions import VehicleSafety
import torch
import sys

report = Reporter()


def evaluate_point(problem, point):
    x_tensor = torch.tensor(list(point.values()), dtype=torch.float64).view(1, -1)
    f_tensor = VehicleSafety().evaluate_true(x_tensor)
    evaluation = {"F1": round(f_tensor[0, 0].item(), 3)}
    return evaluation


def objective(problem: str, config: dict):
    x_tensor = torch.tensor(list(config.values()), dtype=torch.float64).view(1, -1)
    f_tensor = VehicleSafety().evaluate_true(x_tensor)
    evaluation = {"F1": round(f_tensor[0, 0].item(), 3)}
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
    for i in range(5):
        parser.add_argument(f"--x{i}", type=float)
    args, _ = parser.parse_known_args(modified_args)
    objective(problem="vehicle_safety", config=vars(args))
