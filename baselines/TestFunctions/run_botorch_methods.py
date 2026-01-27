import os
import pandas as pd
import torch
from torch import tensor


from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from botorch.utils.transforms import normalize, unnormalize

from botorch.acquisition.multi_objective import (
    ExpectedHypervolumeImprovement,
    qLogExpectedHypervolumeImprovement,
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch.models import SingleTaskGP, ModelListGP
from botorch.optim import optimize_acqf
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)
from botorch.utils.sampling import draw_sobol_samples
import argparse


# Use the new benchmark registry
from benchmark_registry import get_benchmark, BENCHMARKS

dtype = torch.double
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_botorch_methods(method, problem, config_space, trials, seed):
    # Use the registry to get the problem instance, config space, and mapping function
    problem_instance, config_space, mapping_fn = get_benchmark(
        problem, seed=seed, model_name="baseline"
    )
    dim = len(config_space)

    bounds = tensor(
        [
            [v.lower for v in config_space.values()],
            [v.upper for v in config_space.values()],
        ],
        dtype=dtype,
        device=device,
    )

    NUM_INITIAL_POINTS = 5

    # --- START OF MODIFICATION: Determine batch size and iterations based on method ---
    if method == "qLogEHVI":
        q_batch_size = 4
        num_iterations = trials // q_batch_size
        print(
            f"Method '{method}' selected. Running in BATCH mode (q=4) for {num_iterations} iterations."
        )
    else:
        q_batch_size = 1
        num_iterations = trials
        print(
            f"Method '{method}' selected. Running in SEQUENTIAL mode (q=1) for {num_iterations} iterations."
        )

    total_evals = NUM_INITIAL_POINTS + (num_iterations * q_batch_size)
    print(
        f"Total evaluations will be: {NUM_INITIAL_POINTS} (initial) + {num_iterations * q_batch_size} (BO) = {total_evals}"
    )

    # Initialize training data with 5 initial points
    train_X = draw_sobol_samples(bounds=bounds, n=5, q=1).squeeze(1)

    # NEW: Evaluate points more efficiently
    train_Y_list = []
    n_obj = problem_instance.n_obj
    for x in train_X:
        point_dict = {key: val.item() for key, val in zip(config_space.keys(), x)}
        # If a mapping function exists, use it to map continuous to discrete
        if mapping_fn is not None:
            point_dict = mapping_fn(point_dict)
        # FIX: Call evaluate_point only ONCE per point
        evals = problem_instance.evaluate_point(point_dict)[1]
        # Dynamically extract all objectives
        evals_list = list(evals.values())[:n_obj]
        train_Y_list.append([float(val) for val in evals_list])
    train_Y = tensor(train_Y_list, dtype=dtype, device=device)

    acquisition_functions = {
        "EHVI": ExpectedHypervolumeImprovement,
        "qLogEHVI": qLogExpectedHypervolumeImprovement,  # TODO: If it takes to long only run this one
        # "qLogNEHVI": qLogNoisyExpectedHypervolumeImprovement,
    }

    for trial in range(num_iterations):
        print(f"Running Trial {trial + 1}/{trials}")

        train_X_normalized = normalize(train_X, bounds)

        train_Y_neg = -train_Y

        # Initialize and FIT the GP models
        models = []
        for i in range(train_Y_neg.shape[-1]):
            train_y = train_Y_neg[..., i : i + 1]
            gp = SingleTaskGP(train_X_normalized, train_y)
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)
            models.append(gp)
        model = ModelListGP(*models)

        ref_point_tensor = train_Y_neg.min(0).values
        # Convert to a LIST only when passing to the acquisition function
        # This avoids warnings about tensor vs list
        ref_point_list = ref_point_tensor.tolist()

        partitioning = FastNondominatedPartitioning(
            ref_point=ref_point_tensor, Y=train_Y_neg
        )

        # --- START OF MODIFICATION: Select acquisition function based on method ---
        if method == "EHVI":
            acqf = ExpectedHypervolumeImprovement(
                model=model, ref_point=ref_point_list, partitioning=partitioning
            )
        elif method == "qLogEHVI":
            acqf = qLogExpectedHypervolumeImprovement(
                model=model,
                ref_point=ref_point_list,
                partitioning=partitioning,
                sampler=None,
            )
        else:
            raise NotImplementedError(
                f"The method '{method}' is not supported in this script."
            )

        # Optimize acquisition function
        # FIX: Bounds for optimize_acqf must match the problem's dimension
        normalized_bounds = torch.tensor(
            [[0.0] * dim, [1.0] * dim], dtype=dtype, device=device
        )

        candidates_normalized, _ = optimize_acqf(
            acq_function=acqf,
            bounds=normalized_bounds,
            q=q_batch_size,  # Use the dynamically set batch size
            num_restarts=20,
            raw_samples=1024,
            options={"batch_limit": 5, "maxiter": num_iterations},
        )

        # The new candidate is in the normalized space, create a 1-point tensor
        new_x_normalized = candidates_normalized.detach()
        new_x = unnormalize(new_x_normalized, bounds=bounds)

        new_Y_list = []
        for point_tensor in new_x:
            point_dict = {
                key: val.item() for key, val in zip(config_space.keys(), point_tensor)
            }
            if mapping_fn is not None:
                point_dict = mapping_fn(point_dict)
            evals = problem_instance.evaluate_point(point_dict)[1]
            # Dynamically extract all objectives
            evals_list = list(evals.values())[:n_obj]
            new_Y_list.append([float(val) for val in evals_list])

        new_Y = tensor(new_Y_list, dtype=dtype, device=device)

        # Update training data with the unnormalized point
        train_X = torch.cat([train_X, new_x], dim=0)
        train_Y = torch.cat([train_Y, new_Y], dim=0)

    # Save results
    save_botorch_results(train_X, train_Y, method, problem, seed)


# FIX: Removed config_space from save function signature as it's not needed
def save_botorch_results(candidates, train_Y, method, problem, seed):
    configs = candidates.tolist()
    fvals = train_Y.tolist()

    # Map configs to discrete if mapping_fn exists
    _, config_space, mapping_fn = get_benchmark(problem)
    config_keys = list(config_space.keys())
    if mapping_fn is not None:
        mapped_configs = [
            mapping_fn({k: v for k, v in zip(config_keys, c)}) for c in configs
        ]
    else:
        mapped_configs = [dict(zip(config_keys, c)) for c in configs]

    if len(configs) != len(fvals):
        raise ValueError(
            f"Mismatch in lengths: configs ({len(configs)}) and fvals ({len(fvals)}) must be of the same length."
        )

    # Dynamically save all objectives
    n_obj = len(fvals[0]) if fvals else 0
    result = {
        "configs": mapped_configs,
    }
    for i in range(n_obj):
        result[f"F{i + 1}"] = [f[i] for f in fvals]

    results = pd.DataFrame(result)
    dir_path = (
        f"../../results/{BENCHMARKS[problem]['result_folder']}/{method}/observed_fvals/"
    )
    os.makedirs(dir_path, exist_ok=True)
    results.to_csv(f"{dir_path}/{method}_{problem}_{seed}.csv", index=False)
    print(f"\nResults saved to {dir_path}")


# --- Your main function is great, no changes needed ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["EHVI", "qEHVI", "qNEHVI", "qLogEHVI", "qLogNEHVI"],
    )
    parser.add_argument("--seed", type=int, default=31415927)
    parser.add_argument(
        "--trials",
        type=int,
        default=10,
        help="Number of trials to run AFTER initial sampling",
    )
    parser.add_argument(
        "--problem",
        type=str,
        default="",
        help="Problem ID to evaluate",
    )
    args, _ = parser.parse_known_args()
    method = args.method
    seed = args.seed
    trials = args.trials
    problem = args.problem

    # config_space is now handled by the registry in run_botorch_methods
    run_botorch_methods(method, problem, None, trials, seed)


if __name__ == "__main__":
    main()
