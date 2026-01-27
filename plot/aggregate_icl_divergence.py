import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plot_settings import get_color
import ast  # To safely parse string representations of lists/dicts
from scipy.spatial.distance import cdist


DEFAULT_FIGSIZE = (8, 4)  # width, height in inches for all plots
FONT_SIZE = 11
AX_LABELSIZE = 12
TITLE_SIZE = 12
TICK_LABELSIZE = 10
LEGEND_FONTSIZE = 9
USE_LATEX = True

# Use LaTeX for text rendering and apply the unified sizes
plt.rcParams.update(
    {
        "text.usetex": USE_LATEX,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": FONT_SIZE,
        "axes.labelsize": AX_LABELSIZE,
        "axes.titlesize": TITLE_SIZE,
        "legend.fontsize": LEGEND_FONTSIZE,
        "xtick.labelsize": TICK_LABELSIZE,
        "ytick.labelsize": TICK_LABELSIZE,
        "figure.figsize": DEFAULT_FIGSIZE,
    }
)


MARKERS = {
    "LLAMBO": "o",
    "LLAMBO-KD": "s",
    "LLAMBO-GD": "^",
    "LLAMBO-KD-GD": "d",
    "MOHOLLM": "s",
    "LLM": "o",
    "MOHOLLM (Gemini 2.0 Flash)": "s",
    "MOHOLLM (Gemini 2.0 Flash) (Context)": "s",  # "MOHOLLM (Context)",
    "MOHOLLM (Gemini 2.0 Flash) (minimal)": "s",  # "MOHOLLM (Context)",
    "mohollm (Gemini 2.0 Flash)": "o",
}


def parse_string_to_list(s):
    """Safely parses a string like '[{...}, {...}]' into a Python list."""
    try:
        # Handle cases where the data might already be a list
        if isinstance(s, list):
            return s
        data = ast.literal_eval(s)
        if isinstance(data, dict):
            return [data]
        return data
    except (ValueError, SyntaxError):
        return []


def get_vectors_from_dicts(dict_list):
    """Converts a list of configuration dictionaries into a 2D NumPy array."""
    if not dict_list:
        return np.array([]), []

    # Use the keys from the first dictionary as the canonical order
    headers = sorted(dict_list[0].keys())

    vectors = []
    for d in dict_list:
        vectors.append([d.get(h, 0) for h in headers])

    return np.array(vectors), headers


def calculate_icl_divergence(row, plot_surrogate_model=False):
    """
    Calculates the ICL Divergence Score for a single row of the DataFrame.
    The score is the average minimum distance from each candidate to the ICL set.
    """
    if plot_surrogate_model:
        icl_configs = parse_string_to_list(row["current_icl_evaluations"])
        candidate_configs = parse_string_to_list(row["llm_surrogate_proposal"])
    else:
        icl_configs = parse_string_to_list(row["current_icl_configs"])
        candidate_configs = parse_string_to_list(row["llm_candidate_proposal"])

    if not icl_configs or not candidate_configs:
        return np.nan

    icl_vectors, icl_headers = get_vectors_from_dicts(icl_configs)
    candidate_vectors, candidate_headers = get_vectors_from_dicts(candidate_configs)

    # Ensure headers are consistent, which they should be in a real run
    if (
        not np.array_equal(icl_headers, candidate_headers)
        and icl_headers
        and candidate_headers
    ):
        print("Warning: Mismatch in decision space headers between ICL and candidates.")
        return np.nan

    if icl_vectors.size == 0 or candidate_vectors.size == 0:
        return np.nan

    # Calculate all pairwise distances (candidates vs. ICL)
    # Normalize using the min/max across BOTH the ICL and candidate vectors
    # so proposals that lie outside the current ICL bounds are scaled consistently.
    combined = np.vstack([icl_vectors, candidate_vectors])
    min_vals = np.min(combined, axis=0)
    max_vals = np.max(combined, axis=0)
    range_vals = max_vals - min_vals
    # avoid division by zero for constant-dimension cases
    range_vals[range_vals == 0] = 1.0
    icl_vectors_norm = (icl_vectors - min_vals) / range_vals
    candidate_vectors_norm = (candidate_vectors - min_vals) / range_vals
    distance_matrix = cdist(candidate_vectors_norm, icl_vectors_norm, "euclidean")
    # Unnormalized
    # distance_matrix = cdist(candidate_vectors, icl_vectors, "euclidean")

    # Find the minimum distance for each candidate to the ICL set
    min_distances = np.min(distance_matrix, axis=1)

    # The ICL Divergence Score is the average of these minimum distances
    gravity_score = np.mean(min_distances)

    return gravity_score


def compute_icl_divergence_trajectory(df):
    """
    Computes the per-iteration ICL Divergence Score for a run and returns the
    list of scores (one per iteration). This no longer computes or returns any
    cumulative-evaluation x-coordinates; alignment is done later by iteration index.
    """
    if "icl_divergence_score" not in df.columns:
        df["icl_divergence_score"] = df.apply(calculate_icl_divergence, axis=1)

    # Return the per-iteration scores (may contain NaNs which are handled during alignment)
    return df["icl_divergence_score"].tolist()


def group_runs_by_method(files, filter_str=None):
    methods = {}
    for f in files:
        # method is expected to be the third-to-last path part
        parts = f.split(os.sep)
        if len(parts) < 3:
            continue
        method = parts[-3]
        if filter_str and filter_str not in f:
            continue
        methods.setdefault(method, []).append(f)
    return methods


def load_trajectories_for_method(file_list, trials=None):
    runs = []
    for f in file_list:
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        if trials is not None:
            df = df[:trials]
        y = compute_icl_divergence_trajectory(df)
        # ensure at least one non-nan
        if y and any([not np.isnan(v) for v in y]):
            runs.append(np.array(y, dtype=float))
    return runs


def aggregate_groups(method_runs, group_fn):
    """Aggregate runs into two groups using group_fn(method)->group_name"""
    groups = {}
    for method, runs in method_runs.items():
        group = group_fn(method)
        groups.setdefault(group, []).extend(runs)
    return groups


def align_and_compute_stats(runs):
    """Given list of 1D numpy arrays (runs), align by iteration index and compute mean and std error."""
    if not runs:
        return np.array([]), np.array([]), np.array([])
    max_len = max(len(r) for r in runs)
    aligned = []
    for r in runs:
        if len(r) < max_len:
            # forward-fill then backfill for leading NaNs
            s = pd.Series(r, index=np.arange(len(r)))
            s = s.reindex(np.arange(max_len), method="ffill").bfill()
            aligned.append(s.values)
        else:
            aligned.append(r)
    arr = np.vstack(aligned)
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0)
    n = arr.shape[0]
    stderr = 1.96 * std / np.sqrt(max(1, n))
    x = np.arange(max_len)
    return x, mean, stderr


def plot_grouped_icl(groups_stats, title, outdir, filename):
    fig, ax = plt.subplots(figsize=(3.5, 3))
    colors = {
        "MOHOLLM": get_color("MOHOLLM (Gemini 2.0 Flash)", 0),
        "LLM": get_color("mohollm (Gemini 2.0 Flash)", 1),
    }
    x_min, x_max = None, None
    for i, (group, (x, mean, stderr)) in enumerate(groups_stats.items()):
        if x.size == 0:
            continue
        ax.plot(x, mean, label=group, color=colors.get(group), marker=MARKERS.get(group, "o"))
        ax.fill_between(
            x, mean - stderr, mean + stderr, color=colors.get(group), alpha=0.2
        )
        # Track the min and max x values across all groups
        if x_min is None or x[0] < x_min:
            x_min = x[0]
        if x_max is None or x[-1] > x_max:
            x_max = x[-1]
    
    # Remove left and right margins
    if x_min is not None and x_max is not None:
        ax.set_xlim(x_min, x_max)
    
    ax.set_xlabel("Function Evaluations")
    ax.set_ylabel(r"ICL Divergence")
    #ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.2)
    ax.legend()
    plt.tight_layout()
    for ext in ("png", "svg", "pdf"):
        outpath = os.path.join(outdir, f"{filename}.{ext}")
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()


def classify_method(method):
    """Return 'MOHOLLM' for the two exact MOHOLLM names, 'LLM' for the exact mohollm name, else None."""
    mohollm_names = {
        "MOHOLLM (Gemini 2.0 Flash) (Context)",
        "MOHOLLM (Gemini 2.0 Flash)",
    }
    llm_names = {"mohollm (Gemini 2.0 Flash)"}
    if method in mohollm_names:
        print(method, "MOHOLLM")
        return "MOHOLLM"
    if method in llm_names:
        print(method, "LLM")
        return "LLM"
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate ICL Divergence across benchmarks and seeds"
    )
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to results folder"
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        required=True,
        help="Comma-separated benchmarks to include",
    )
    parser.add_argument("--trials", type=int, help="Number of trials to consider")
    parser.add_argument(
        "--filter",
        type=str,
        help="Optional filter to include only files matching this substring",
    )
    parser.add_argument(
        "--title", type=str, default="Aggregate ICL Divergence", help="Plot title"
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="aggregate_icl_divergence",
        help="Output filename prefix",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./plots/icl_divergence/aggregate",
        help="Output directory",
    )
    args = parser.parse_args()

    benchmarks = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    all_files = []
    for b in benchmarks:
        search = os.path.join(args.data_path, b, "**", "*.csv")
        found = glob.glob(search, recursive=True)
        # keep only files that belong to icl_llm_proposal_trajectory
        found = [f for f in found if "icl_llm_proposal_trajectory" in f]
        all_files.extend(found)
    if args.filter:
        all_files = [f for f in all_files if args.filter in f]
    method_files = group_runs_by_method(all_files)
    method_runs = {}
    for m, files in method_files.items():
        runs = load_trajectories_for_method(files, trials=args.trials)
        if runs:
            method_runs[m] = runs

    # aggregate into MOHOLLM and LLM using the strict classifier (skip other methods)
    filtered_method_runs = {}
    for m, runs in method_runs.items():
        grp = classify_method(m)
        if grp is not None:
            filtered_method_runs[m] = runs

    groups = aggregate_groups(filtered_method_runs, classify_method)
    print("Keys: ", groups.keys())

    groups_stats = {}
    for g, runs in groups.items():
        x, mean, stderr = align_and_compute_stats(runs)
        groups_stats[g] = (x, mean, stderr)

    plot_grouped_icl(groups_stats, args.title, args.output_path, args.filename)


if __name__ == "__main__":
    main()
