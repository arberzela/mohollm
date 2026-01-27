import numpy as np
import pandas as pd

from pymoo.factory import get_problem
from pymoo.algorithms.soo.nonconvex.random_search import RandomSearch
from pymoo.optimize import minimize

# Load the built-in welded beam problem.
problem = get_problem("welded_beam")

# Define termination criteria and the Random Search algorithm.
termination = ("n_eval", 100)

# Seeds to loop over.
seeds = [31415927, 6790, 42]


for seed in seeds:
    print(f"Running optimization with seed = {seed}")
    algorithm = RandomSearch()
    # Run the random search optimization with the given seed.
    res = minimize(
        problem, algorithm, termination, seed=seed, save_history=True, verbose=True
    )

    # Process the optimization history to collect candidate solutions.
    for entry in res.history:
        # Retrieve evaluated candidate solutions, objective values, and constraint evaluations.
        X = entry.opt.get("X")
        F = entry.opt.get("F")
        G = entry.opt.get("G")

        for x, f_val, g_vals in zip(X, F, G):
            data = {
                "seed": seed,
                "x1": x[0],
                "x2": x[1],
                "x3": x[2],
                "x4": x[3],
                "F1": f_val[0],
                "F2": f_val[1],
                "g1": g_vals[0],
                "g2": g_vals[1],
                "g3": g_vals[2],
                "g4": g_vals[3],
                "g5": g_vals[4],
            }

            df = pd.DataFrame(data)
            df.to_csv(f"RS_welded_beam_{seed}.csv", index=False)
            print(
                f"Results for seed {seed} saved to 'welded_beam_random_search_seed_{seed}.csv'"
            )
