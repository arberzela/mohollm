import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast  # To safely parse string representations of lists/dicts
import glob
import argparse
import os

from plot_settings import COLORS, LINE_STYLES, LABEL_MAP


# Use LaTeX for text rendering if available
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 4,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    }
)


def parse_string_to_list(s):
    """Safely parses a string like '[{...}, {...}]' into a Python list."""
    try:
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
    headers = sorted(dict_list[0].keys())
    vectors = [[d.get(h, 0) for h in headers] for d in dict_list]
    return np.array(vectors), headers


def compute_adaptability_trajectory(df, plot_surrogate_model=False):
    """
    Computes the MODIFIED Adaptability Index against the cumulative number of function evaluations.
    This version is corrected for algorithms where the ICL set is the entire history.
    It measures the movement of the historical centroid, normalized by the cloud's spread.
    """
    x_coords, y_coords = [], []

    # Pre-calculate cumulative evaluations (this part remains the same)
    df["num_evaluations"] = df["best_candidate_evaluations"].apply(
        lambda s: len(parse_string_to_list(s))
    )
    df["cumulative_evaluations"] = df["num_evaluations"].cumsum()

    # Iterate from the second row since we need a "previous" state (t) and "current" state (t+1)
    for i in range(1, len(df)):
        # --- 1. Gather data from previous (t) and current (t+1) iterations ---
        prev_row = df.iloc[i - 1]
        curr_row = df.iloc[i]

        # Use a consistent column name based on the plot_surrogate_model flag
        if plot_surrogate_model:
            col_name = "current_icl_evaluations"
        else:
            col_name = "current_icl_configs"

        # Get the full history of points from the previous and current iterations
        # I_t is the history at time t
        icl_t_configs = parse_string_to_list(prev_row[col_name])
        icl_t_vectors, _ = get_vectors_from_dicts(icl_t_configs)

        # I_{t+1} is the history at time t+1
        icl_t1_configs = parse_string_to_list(curr_row[col_name])
        icl_t1_vectors, _ = get_vectors_from_dicts(icl_t1_configs)

        # --- 2. Check if calculation is possible ---
        if icl_t_vectors.size == 0 or icl_t1_vectors.size == 0:
            continue

        # --- 3. Calculate Old and New Centroids ---
        centroid_t = np.mean(icl_t_vectors, axis=0)  # Centroid of history at time t
        centroid_t1 = np.mean(icl_t1_vectors, axis=0)  # Centroid of history at time t+1

        # --- 4. Calculate Centroid Movement (Numerator) ---
        centroid_movement = np.linalg.norm(centroid_t1 - centroid_t)

        # --- 5. Calculate Spread of the Old Point Cloud (Denominator) ---
        # The spread is the average distance of each point in I_t from its own center.
        distances_from_center = np.linalg.norm(icl_t_vectors - centroid_t, axis=1)
        spread_t = np.mean(distances_from_center)

        # --- 6. Calculate the Modified Adaptability Index ---
        if (
            spread_t < 1e-9
        ):  # Avoid division by zero if all previous points were identical
            modified_adaptability_index = np.nan
        else:
            modified_adaptability_index = centroid_movement / spread_t

        if not np.isnan(modified_adaptability_index):
            # The x-coordinate is the budget at the start of the current trial
            x_coord = prev_row["cumulative_evaluations"]
            x_coords.append(x_coord)
            y_coords.append(modified_adaptability_index)

    return (x_coords, y_coords)


def create_adaptability_over_time_plot(data, title, path, filename):
    """Create publication-ready adaptability index over time plots."""
    fig, ax = plt.subplots(figsize=(5, 3.5))
    methods = sorted(data["mean"].keys())
    x_axis = data["x_axis"]

    for i, method in enumerate(methods):
        mean_values = data["mean"][method]
        std_values = data["std"][method]
        plot_len = min(len(x_axis), len(mean_values))

        # Standard Error of the Mean (SEM) for confidence interval
        std_error = 1.96 * std_values / np.sqrt(len(mean_values))

        ax.plot(
            x_axis[:plot_len],
            mean_values[:plot_len],
            label=LABEL_MAP.get(method, method),
            color=COLORS[i % len(COLORS)],
            linestyle=LINE_STYLES[i % len(LINE_STYLES)],
            linewidth=1.2,
        )

        ax.fill_between(
            x_axis[:plot_len],
            mean_values[:plot_len] - std_error,
            mean_values[:plot_len] + std_error,
            color=COLORS[i % len(COLORS)],
            alpha=0.2,
        )

    ax.set_xlabel("Function Evaluations")
    ax.set_ylabel("Adaptability Index")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.6, linewidth=0.4)
    # Add a horizontal line at y=1 for reference

    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.2f}"))

    handles, labels = ax.get_legend_handles_labels()
    # Reorder legend to put the baseline last
    order = [h for h, l in zip(handles, labels) if "Baseline" not in l]
    order_labels = [l for l in labels if "Baseline" not in l]

    ax.legend(
        handles=order,
        labels=order_labels,
        loc="best",
        frameon=True,
        framealpha=0.95,
        edgecolor="k",
        fancybox=False,
    )

    plt.tight_layout()
    for file_type in ["svg", "pdf", "png"]:
        dir_path = f"{path}/{file_type}/{filename}.{file_type}"
        os.makedirs(os.path.dirname(dir_path), exist_ok=True)
        plt.savefig(dir_path, dpi=300, bbox_inches="tight")
    plt.close()


def extract_method_name(files):
    methods = sorted(list(set([file.split(os.sep)[-3] for file in files])))
    return methods


def group_data_by_method(files_list, file_filter):
    method_names = extract_method_name(files_list)
    method_files = {}
    for method in method_names:
        files = []
        for file in files_list:
            if method in file.split(os.sep) and (
                file_filter in file if file_filter else True
            ):
                files.append(file)
        method_files[method] = files
    return method_files


def main():
    parser = argparse.ArgumentParser(description="Plot Adaptability Index over time.")
    parser.add_argument("--benchmark", type=str, required=True, help="Benchmark name")
    parser.add_argument("--title", type=str, required=True, help="Plot title")
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to experiment data folder"
    )
    parser.add_argument(
        "--filter", type=str, help="Filter to only include files containing this string"
    )
    parser.add_argument(
        "--filename",
        default="",
        type=str,
        help="Filename of the plot (without extension)",
    )
    parser.add_argument(
        "--plot_surrogate_model",
        default=False,
        type=bool,
        help="If true plot the surrogate model ICL Divergence instead.",
    )
    args = parser.parse_args()
    plot_surrogate_model = args.plot_surrogate_model

    output_path = f"./plots/icl_adaptability/{args.benchmark}"
    if not args.filename:
        args.filename = f"{args.benchmark}_adaptability_index"

    file_names = glob.glob(f"{args.data_path}/**/*.csv", recursive=True)
    trajectory_files = [
        f for f in file_names if "icl_llm_proposal_trajectory" in f.split(os.sep)
    ]

    method_files = group_data_by_method(trajectory_files, args.filter)
    raw_trajectories = {}

    max_evals = 0
    for method, files in method_files.items():
        trajectories = []
        for file in files:
            df = pd.read_csv(file)
            if len(df) < 2:
                continue  # Need at least two rows to calculate anything
            x_coords, y_coords = compute_adaptability_trajectory(
                df, plot_surrogate_model=plot_surrogate_model
            )
            if x_coords:
                trajectories.append((x_coords, y_coords))
                if x_coords[-1] > max_evals:
                    max_evals = x_coords[-1]
        raw_trajectories[method] = trajectories

    common_x_axis = np.arange(max_evals + 1)
    mean_trajectories = {}
    std_trajectories = {}

    for method, trajectories in raw_trajectories.items():
        if not trajectories:
            continue
        aligned_trajectories = []
        for x_vals, y_vals in trajectories:
            s = pd.Series(y_vals, index=x_vals)
            s_reindexed = s.reindex(common_x_axis, method="ffill")
            s_reindexed = s_reindexed.bfill()  # Backfill first element if it's NaN
            if not s_reindexed.empty:
                aligned_trajectories.append(s_reindexed.values)

        if aligned_trajectories:
            mean_trajectories[method] = np.nanmean(aligned_trajectories, axis=0)
            std_trajectories[method] = np.nanstd(aligned_trajectories, axis=0)

    data = {
        "mean": mean_trajectories,
        "std": std_trajectories,
        "x_axis": common_x_axis,
    }
    create_adaptability_over_time_plot(data, args.title, output_path, args.filename)


if __name__ == "__main__":
    main()
