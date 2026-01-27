import argparse
import glob
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pymoo.indicators.hv import HV
from plot_settings import (
    get_color,
    LINE_STYLE_HV,
    LABEL_MAP_HV,
    HV_REFERENCE_POINTS,
    MARKER_MAP,
)
import math

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 16,
        "axes.labelsize": 16,
        "axes.titlesize": 20,
        "legend.fontsize": 20,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
    }
)


def compute_max_hv_from_pool(pool_file, reference_point):
    """
    Compute the maximum hypervolume from the good pool data.

    Args:
        pool_file: Path to the good pool CSV file
        reference_point: Reference point for HV calculation

    Returns:
        Maximum hypervolume value
    """
    df = pd.read_csv(pool_file)
    # Extract objective columns (F1, F2, ...)
    objective_cols = [col for col in df.columns if col.startswith("F")]
    objectives = df[objective_cols].to_numpy()

    ind = HV(ref_point=reference_point)
    max_hv = ind(objectives)
    return max_hv


def save_max_hv(
    benchmark, max_hv, output_file="./scripts/data_pools/max_hv_values.json"
):
    """
    Save the max HV value to a JSON file for later use.

    Args:
        benchmark: Name of the benchmark
        max_hv: Maximum hypervolume value
        output_file: Path to the output JSON file
    """
    # Load existing data if file exists
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            data = json.load(f)
    else:
        data = {}

    # Update with new value
    data[benchmark] = max_hv

    # Save back to file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved max HV for {benchmark}: {max_hv} to {output_file}")


def extract_method_name(files):
    methods = []
    for file in files:
        method = file.split("/")[-3]
        if method not in methods:
            methods.append(method)
    return methods


def group_data_by_method(observed_fvals_files, filter):
    method_names = extract_method_name(observed_fvals_files)
    method_files = {}
    for method in method_names:
        files = []
        for file in observed_fvals_files:
            if method in file.split("/") and (filter in file if filter else True):
                files.append(file)
        method_files[method] = files
    return method_files


def custom_sort_key(method_name):
    # Sort beta values in order: 0.0, 0.25, 0.5, 0.75, 1.0
    if "Beta=" in method_name:
        if "MOHOLLM" in method_name:
            beta_val = float(method_name.split("Beta=")[1].rstrip(")"))
            return (0, beta_val)  # MOHOLLM first
        elif "mohollm" in method_name:
            beta_val = float(method_name.split("Beta=")[1].rstrip(")"))
            return (1, beta_val)  # mohollm second
    return (2, method_name)


def compute_hypervolume(df, reference_point):
    objectives = df.to_numpy()
    ind = HV(ref_point=reference_point)
    hypervolume = ind(objectives)
    return hypervolume


def convert_data_to_log_hv_diff_over_time(fvals, reference_point, max_hv):
    """
    Compute log HV difference over time.

    Args:
        fvals: DataFrame with objective values
        reference_point: Reference point for HV calculation
        max_hv: Maximum achievable hypervolume

    Returns:
        Array of log HV differences
    """
    # Compute HV for each step
    hv = []
    for step in range(1, len(fvals) + 1):
        hv_val = compute_hypervolume(fvals.iloc[:step], reference_point)
        hv.append(hv_val)

    # Compute log difference from max_hv for each step
    log_diff = np.log10(np.maximum((max_hv - np.array(hv)), 1e-8))
    return log_diff


def create_log_hv_diff_plot(ax, data, title, x_lim, y_lim):
    methods = sorted(data["mean"].keys(), key=custom_sort_key)
    handles = []
    labels = []

    for i, method in enumerate(methods):
        mean_values = data["mean"][method]
        std_values = data["std"][method]
        trials = range(len(mean_values))
        std_error = 1.96 * std_values / np.sqrt(len(mean_values))

        # Use LABEL_MAP_HV for beta ablations
        label = LABEL_MAP_HV.get(method, method)

        (line,) = ax.plot(
            trials,
            mean_values,
            label=label,
            color=get_color(method, i),
            linestyle=LINE_STYLE_HV[i % len(LINE_STYLE_HV)],
            linewidth=2.5,
            marker=MARKER_MAP.get(method, None),
            markersize=5,
            markevery=5,
        )
        ax.fill_between(
            trials,
            mean_values - std_error,
            mean_values + std_error,
            color=get_color(method, i),
            alpha=0.2,
            linewidth=0,
        )
        handles.append(line)
        labels.append(label)

    ax.set_xlabel("Function Evaluations")
    ax.set_ylabel("Log Hypervolume Difference")
    ax.set_title(title, fontweight="bold")

    if x_lim and all(x != "" for x in x_lim):
        ax.set_xlim(float(x_lim[0]), float(x_lim[1]))
    if y_lim and all(y != "" for y in y_lim):
        ax.set_ylim(float(y_lim[0]), float(y_lim[1]))

    ax.grid(True, linestyle="--", alpha=0.5, linewidth=0.4)
    return handles, labels


def create_log_hv_diff_subplot(ax, data, title, x_lim, y_lim, show_ylabel=True):
    methods = sorted(data["mean"].keys(), key=custom_sort_key)
    handles = []
    labels = []

    # Define consistent colors for each method type
    color_map = {
        "MOHOLLM": "#e6194B",  # Red
        "mohollm": "#f56805",  # Orange
    }

    for i, method in enumerate(methods):
        mean_values = data["mean"][method]
        std_values = data["std"][method]
        trials = range(len(mean_values))
        std_error = 1.96 * std_values / np.sqrt(len(mean_values))

        # Determine simplified label (just method type, not beta value)
        if "MOHOLLM" in method:
            simplified_label = "MOHOLLM"
            color = color_map["MOHOLLM"]
        elif "mohollm" in method:
            simplified_label = "LLM"
            color = color_map["mohollm"]
        else:
            simplified_label = LABEL_MAP_HV.get(method, method)
            color = get_color(method, i)

        (line,) = ax.plot(
            trials,
            mean_values,
            label=simplified_label,
            color=color,
            linestyle="-",
            linewidth=2.5,
            marker=MARKER_MAP.get(method, None),
            markersize=5,
            markevery=5,
        )
        ax.fill_between(
            trials,
            mean_values - std_error,
            mean_values + std_error,
            color=color,
            alpha=0.2,
            linewidth=0,
        )
        handles.append(line)
        labels.append(simplified_label)

    ax.set_xlabel("Function Evaluations")
    if show_ylabel:
        ax.set_ylabel("Log Hypervolume Difference")
    ax.set_title(title, fontweight="bold")

    if x_lim and all(x != "" for x in x_lim):
        ax.set_xlim(float(x_lim[0]), float(x_lim[1]))

    # Set fixed y-axis limits or use provided limits
    if y_lim and all(y != "" for y in y_lim):
        ax.set_ylim(float(y_lim[0]), float(y_lim[1]))
    else:
        ax.set_ylim(0, 3)  # Default fixed y-axis range

    ax.grid(True, linestyle="--", alpha=0.5, linewidth=0.4)
    return handles, labels


def main():
    parser = argparse.ArgumentParser(
        description="Plot log HV difference for beta ablations on Poloni benchmark with subplots."
    )
    parser.add_argument("--benchmarks", nargs="+", type=str, required=True)
    parser.add_argument("--titles", nargs="+", type=str, required=True)
    parser.add_argument("--data_paths", nargs="+", type=str, required=True)
    parser.add_argument("--filters", nargs="+", type=str, default=None)
    parser.add_argument("--trials", type=int, required=True)
    parser.add_argument("--blacklists", nargs="+", type=str, default=None)
    parser.add_argument("--whitelists", nargs="+", type=str, default=None)
    parser.add_argument("--filename", type=str, default="poloni_beta_ablation_subplots")
    parser.add_argument("--x_lims", nargs="+", type=str, default=None)
    parser.add_argument("--y_lims", nargs="+", type=str, default=None)
    args = parser.parse_args()

    n = len(args.benchmarks)
    ncols = n  # All plots in one row
    nrows = 1
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4 * ncols, 4), sharex=False, sharey=True
    )

    # Ensure axes is always a flat array
    axes = np.atleast_1d(axes)
    if axes.ndim > 1:
        axes = axes.flatten()

    # Debug: Print axes type and shape
    print(f"Number of benchmarks: {n}")
    print(f"Axes type: {type(axes)}")
    print(f"Axes shape: {axes.shape if hasattr(axes, 'shape') else len(axes)}")
    print(f"First axis type: {type(axes[0]) if len(axes) > 0 else 'N/A'}")

    all_handles = None
    all_labels = None

    for idx in range(n):
        benchmark = args.benchmarks[idx]
        title = args.titles[idx]
        data_path = args.data_paths[idx]
        filter = args.filters[idx] if args.filters else None
        trials = args.trials
        blacklist = args.blacklists[idx].split(",") if args.blacklists else []
        whitelist = args.whitelists[idx].split(",") if args.whitelists else []
        x_lim = args.x_lims[idx * 2 : idx * 2 + 2] if args.x_lims else None
        y_lim = args.y_lims[idx * 2 : idx * 2 + 2] if args.y_lims else None

        reference_point = HV_REFERENCE_POINTS.get(benchmark, [62.2463, 52.57454])
        # Compute max HV from pool
        pool_file = "./scripts/data_pools/custom_poloni_good_pool.csv"
        print(f"Computing max HV from pool file: {pool_file}")
        max_hv = compute_max_hv_from_pool(pool_file, reference_point)
        print(f"Max HV: {max_hv}")

        # Save max HV for later use
        save_max_hv(benchmark, max_hv)

        file_names = glob.glob(f"{data_path}/**/*.csv", recursive=True)
        observed_fvals_files = [
            file for file in file_names if "observed_fvals" in file.split("/")
        ]
        method_files = group_data_by_method(observed_fvals_files, filter)
        method_dfs = {
            method: [pd.read_csv(file)[:trials] for file in files]
            for method, files in method_files.items()
        }

        if not method_dfs:
            print(f"Warning: No methods found for benchmark {benchmark} at {data_path}")
            continue

        first_method = next(iter(method_dfs))
        first_df = method_dfs[first_method][0]
        columns = [col for col in first_df.columns if col != "configs"]

        if len(whitelist) > 0 and whitelist[0] != "":
            filtered_dfs = {}
            for method, dfs in method_dfs.items():
                if any(entry == method for entry in whitelist):
                    filtered_dfs[method] = dfs
            method_dfs = filtered_dfs

            if not method_dfs:
                print(
                    f"Warning: No methods matched whitelist for benchmark {benchmark}"
                )
                print(f"Available methods: {list(method_files.keys())}")
                print(f"Whitelist: {whitelist}")
                continue
        elif len(blacklist) > 0 and blacklist[0] != "":
            filtered_dfs = {}
            for method, dfs in method_dfs.items():
                if not any(entry in method for entry in blacklist):
                    filtered_dfs[method] = dfs
            method_dfs = filtered_dfs

        reference_points = HV_REFERENCE_POINTS.get(benchmark)
        if not reference_points:
            reference_points = [1.0] * len(columns)

        # Compute log HV difference over time
        log_hv_diff_over_time = {}
        for method, dfs in method_dfs.items():
            method_log_hv = []
            for df in dfs:
                df = df[columns]
                method_log_hv.append(
                    convert_data_to_log_hv_diff_over_time(
                        df[:trials], reference_points, max_hv
                    )
                )
            log_hv_diff_over_time[method] = np.array(method_log_hv)

        mean_log_hv_diff = {
            method: np.mean(hv, axis=0) for method, hv in log_hv_diff_over_time.items()
        }
        std_log_hv_diff = {
            method: np.std(hv, axis=0) for method, hv in log_hv_diff_over_time.items()
        }
        data = {"mean": mean_log_hv_diff, "std": std_log_hv_diff}

        # Only show y-label on the first subplot
        show_ylabel = idx == 0
        handles, labels = create_log_hv_diff_subplot(
            axes[idx], data, title, x_lim, y_lim, show_ylabel=show_ylabel
        )

        # Collect all unique handles and labels
        if all_handles is None:
            all_handles = []
            all_labels = []

        # Add new handles/labels that aren't already in the collection
        for handle, label in zip(handles, labels):
            if label not in all_labels:
                all_handles.append(handle)
                all_labels.append(label)

    for ax in axes[n:]:
        ax.axis("off")

    # Legend configuration for single row layout
    ncols = len(all_labels)  # All legend items in one row
    fig.subplots_adjust(bottom=0.25, wspace=0.3)
    fig.legend(
        all_handles,
        all_labels,
        loc="lower center",
        ncol=ncols,
        bbox_to_anchor=(0.5, -0.12),
        frameon=True,
        framealpha=0.95,
        edgecolor="k",
        fancybox=False,
        columnspacing=1.0,
        handletextpad=0.5,
    )

    plt.tight_layout()

    # Save plot
    for file_type in ["svg", "pdf", "png"]:
        out_dir = f"./plots/beta_ablation_poloni_subplots/{file_type}/"
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(
            f"{out_dir}/{args.filename}.{file_type}", dpi=300, bbox_inches="tight"
        )

    plt.close()
    print("Done!")


if __name__ == "__main__":
    main()
