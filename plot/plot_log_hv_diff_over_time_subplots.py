import argparse
import glob
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pymoo.indicators.hv import HV
from plot_settings import (
    get_color,
    LINE_STYLE_HV,
    LABEL_MAP_HV,
    HV_REFERENCE_POINTS,
    MAX_HV,
    MARKER_MAP,
)

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
    if "MOHOLLM" in method_name:
        return (2, method_name)
    if "LLM" == method_name or "LLM" in LABEL_MAP_HV.get(method_name, method_name):
        return (1, method_name)
    return (0, method_name)


def compute_hypervolume(df, reference_point):
    objectives = df.to_numpy()
    ind = HV(ref_point=reference_point)
    hypervolume = ind(objectives)
    return hypervolume


def convert_data_to_log_hv_diff_over_time(fvals, reference_point, benchmark):
    # Compute HV for each step
    hv = []
    for step in range(1, len(fvals) + 1):
        hv_val = compute_hypervolume(fvals.iloc[:step], reference_point)
        hv.append(hv_val)
    # Compute log difference from MAX_HV for each step
    max_hv = MAX_HV.get(benchmark, 1.0)
    log_diff = np.log10(np.maximum((max_hv - np.array(hv)), 1e-8))
    return log_diff


def create_log_hv_diff_subplot(ax, data, benchmark, title, x_lim, y_lim):
    methods = sorted(data["mean"].keys(), key=custom_sort_key)
    handles = []
    labels = []
    for i, method in enumerate(methods):
        mean_values = data["mean"][method]
        std_values = data["std"][method]
        trials = range(len(mean_values))
        std_error = 1.96 * std_values / np.sqrt(len(mean_values))
        (line,) = ax.plot(
            trials,
            mean_values,
            label=LABEL_MAP_HV.get(method, method),
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
        labels.append(LABEL_MAP_HV.get(method, method))
    ax.set_xlabel("Function Evaluations")
    ax.set_ylabel("Log Hypervolume Difference")
    ax.set_title(title, fontweight="bold")
    if x_lim and all(x != "" for x in x_lim):
        ax.set_xlim(float(x_lim[0]), float(x_lim[1]))
    if y_lim and all(y != "" for y in y_lim):
        ax.set_ylim(float(y_lim[0]), float(y_lim[1]))
    ax.grid(True, linestyle="--", alpha=0.5, linewidth=0.4)
    return handles, labels


def main():
    parser = argparse.ArgumentParser(
        description="Plot log HV difference over time for multiple benchmarks as subplots."
    )
    parser.add_argument("--benchmarks", nargs="+", type=str, required=True)
    parser.add_argument("--titles", nargs="+", type=str, required=True)
    parser.add_argument("--data_paths", nargs="+", type=str, required=True)
    parser.add_argument("--filters", nargs="+", type=str, default=None)
    parser.add_argument("--trials", type=int, required=True)
    parser.add_argument("--blacklists", nargs="+", type=str, default=None)
    parser.add_argument("--filename", type=str, default=None)
    parser.add_argument("--x_lims", nargs="+", type=str, default=None)
    parser.add_argument("--y_lims", nargs="+", type=str, default=None)
    parser.add_argument("--whitelists", nargs="+", type=str, default=None)
    args = parser.parse_args()

    n = len(args.benchmarks)
    ncols = 3
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(6 * ncols, 4.5 * nrows), sharex=False
    )
    axes = axes.flatten() if n > 1 else [axes]

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

        file_names = glob.glob(f"{data_path}/**/*.csv", recursive=True)
        observed_fvals_files = [
            file for file in file_names if "observed_fvals" in file.split("/")
        ]
        method_files = group_data_by_method(observed_fvals_files, filter)
        method_dfs = {
            method: [pd.read_csv(file)[:trials] for file in files]
            for method, files in method_files.items()
        }
        # Dynamically determine objective columns
        first_method = next(iter(method_dfs))
        first_df = method_dfs[first_method][0]
        columns = [col for col in first_df.columns if col != "configs"]
        if len(whitelist) > 0 and whitelist[0] != "":
            filtered_dfs = {}
            for method, dfs in method_dfs.items():
                if any(entry == method for entry in whitelist):
                    filtered_dfs[method] = dfs
            method_dfs = filtered_dfs
        elif len(blacklist) > 0 and blacklist[0] != "":
            filtered_dfs = {}
            for method, dfs in method_dfs.items():
                if not any(entry in method for entry in blacklist):
                    filtered_dfs[method] = dfs
            method_dfs = filtered_dfs
        # Reference point for HV
        reference_points = HV_REFERENCE_POINTS.get(benchmark)
        if not reference_points:
            reference_points = [1.0] * len(columns)
        log_hv_diff_over_time = {}
        for method, dfs in method_dfs.items():
            method_log_hv = []
            for df in dfs:
                df = df[columns]
                method_log_hv.append(
                    convert_data_to_log_hv_diff_over_time(
                        df[:trials], reference_points, benchmark
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

        handles, labels = create_log_hv_diff_subplot(
            axes[idx], data, benchmark, title, x_lim, y_lim
        )
        if all_handles is None:
            all_handles = handles
            all_labels = labels
    for ax in axes[n:]:
        ax.axis("off")
    if len(all_labels) <= 3:
        ncols = len(all_labels)
    else:
        ncols = len(all_labels) / 3
    fig.subplots_adjust(bottom=0.3, wspace=0.33)
    bbox_to_anchor = (0.5, -0.1)
    if n == 3 and len(all_labels) >= 3:
        bbox_to_anchor = (0.5, -0.3)
    fig.legend(
        all_handles,
        all_labels,
        loc="outside lower center",
        ncol=ncols,
        bbox_to_anchor=bbox_to_anchor,
        frameon=True,
        framealpha=0.95,
        edgecolor="k",
        fancybox=False,
        columnspacing=1.0,
        handletextpad=0.5,
    )
    plt.tight_layout()
    for file_type in ["svg", "pdf", "png"]:
        out_dir = f"./plots/log_hypervolume_difference_over_time_subplots/{file_type}/"
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(
            f"{out_dir}/{args.filename}.{file_type}", dpi=300, bbox_inches="tight"
        )
    plt.close()


if __name__ == "__main__":
    main()
