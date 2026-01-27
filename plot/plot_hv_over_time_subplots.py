import argparse
import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from normalize_data_nb201 import normalize_data_nb201
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
        "axes.labelsize": 18,
        "axes.titlesize": 18,
        "legend.fontsize": 16,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "axes.linewidth": 1.2,
        "grid.linewidth": 0.6,
        "lines.linewidth": 2.8,
        "patch.linewidth": 1.0,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "xtick.minor.width": 0.8,
        "ytick.minor.width": 0.8,
    }
)


def compute_hypervolume(df, reference_point):
    objectives = df.to_numpy()
    ind = HV(ref_point=reference_point)
    hypervolume = ind(objectives)
    return hypervolume


def convert_data_to_hv_over_time(fvals, reference_point=[1.0, 1.0]):
    hypervolume = []
    for step in range(1, len(fvals) + 1):
        hv = compute_hypervolume(fvals.iloc[:step], reference_point)
        hypervolume.append(hv)
    return hypervolume


def normalize_data(fvals, min_max_metrics):
    for column, values in min_max_metrics.items():
        min_val = values["min"]
        max_val = values["max"]
        fvals[column] = (fvals[column] - min_val) / (max_val - min_val)
    return fvals


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
    """
    Custom sorting key. Sorts alphabetically, but ensures methods
    containing 'LLM' are placed at the end to be plotted on top.
    MOHOLLM will be rightmost in legend (last when sorted in descending order).
    """
    if "MOHOLLM" == method_name:
        return (0, method_name)  # Lowest priority, appears last when reversed
    if "LLM" == method_name or "LLM" in LABEL_MAP_HV.get(method_name, method_name):
        return (1, method_name)
    return (2, method_name)  # Highest priority, appears first when reversed


def create_hv_subplot(
    idx,
    ax,
    data,
    benchmark,
    title,
    x_lim,
    y_lim,
    legend_handles_labels=None,
    simplify_legend=False,
):
    methods = sorted(data["mean"].keys(), key=custom_sort_key, reverse=True)
    handles = []
    labels = []

    # Define color mapping for simplified legend
    color_map = {
        "MOHOLLM": "#e6194B",  # red
        "LLM": "#f56805",  # orange
    }

    for i, method in enumerate(methods):
        mean_values = data["mean"][method]
        std_values = data["std"][method]
        trials = range(len(mean_values))
        std_error = 1.96 * std_values / np.sqrt(len(mean_values))
        # std_error = 1.96 * std_values / np.sqrt(10)

        # Determine label and color based on simplify_legend flag
        if simplify_legend:
            # Simplify to just "MOHOLLM" or "LLM"
            if "MOHOLLM" in method:
                simplified_label = "MOHOLLM"
                color = color_map["MOHOLLM"]
            else:
                simplified_label = "LLM"
                color = color_map["LLM"]
        else:
            simplified_label = LABEL_MAP_HV.get(method, method)
            color = get_color(method, i)

        # Add shadow effect for depth
        ax.plot(
            trials,
            mean_values,
            color="black",
            linestyle=LINE_STYLE_HV[i % len(LINE_STYLE_HV)],
            linewidth=3.2,
            alpha=0.15,
            zorder=1,
        )
        
        # Main line with enhanced styling
        (line,) = ax.plot(
            trials,
            mean_values,
            label=simplified_label,
            color=color,
            linestyle=LINE_STYLE_HV[i % len(LINE_STYLE_HV)],
            linewidth=2.8,
            marker=MARKER_MAP.get(method, None),
            markersize=8,
            markevery=5,
            zorder=2,
            markeredgewidth=1.5,
            markeredgecolor="white",
        )
        
        # Enhanced fill with gradient-like effect
        ax.fill_between(
            trials,
            mean_values - std_error,
            mean_values + std_error,
            color=color,
            alpha=0.25,
            linewidth=0,
            zorder=0,
        )
        handles.append(line)
        labels.append(simplified_label)

    #if idx >= 1:
    ax.set_xlabel("Function Evaluations", fontweight="semibold", labelpad=8)
    if idx % 6 == 0:
        ax.set_ylabel("Hypervolume", fontweight="semibold", labelpad=8)
    
    # Enhanced title with styling
    ax.set_title(title, fontweight="bold", pad=12)
    
    # Only set axis limits if both are non-empty and convertible to float
    if x_lim and all(x != "" for x in x_lim):
        ax.set_xlim(float(x_lim[0]), float(x_lim[1]))
    if y_lim and all(y != "" for y in y_lim):
        ax.set_ylim(float(y_lim[0]), float(y_lim[1]))
    
    # Enhanced grid styling with layered effect
    ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.8, color="#CCCCCC", zorder=0)
    ax.grid(True, which="minor", linestyle=":", alpha=0.15, linewidth=0.5, color="#DDDDDD", zorder=0)
    
    # Add subtle background color for contrast
    ax.set_facecolor("#FAFAFA")
    
    # Enhanced tick formatting
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0), useMathText=True)
    ax.tick_params(axis="both", which="major", length=6, width=1.2, pad=6)
    ax.tick_params(axis="both", which="minor", length=3, width=0.8)
    
    # Remove whitespace at edges of x-axis
    ax.margins(x=0)

    ax.set_xticks([10, 20, 30, 40, 50])
    ax.set_xlim(1, 50)

    # Add subtle border
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_edgecolor("#888888")
    return handles, labels


def main():
    parser = argparse.ArgumentParser(
        description="Plot hypervolume over time for multiple benchmarks as subplots."
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        type=str,
        required=True,
        help="List of benchmark names",
    )
    parser.add_argument(
        "--titles", nargs="+", type=str, required=True, help="Titles for each subplot"
    )
    parser.add_argument(
        "--data_paths",
        nargs="+",
        type=str,
        required=True,
        help="Paths to observed fvals folders for each benchmark",
    )
    parser.add_argument(
        "--filters", nargs="+", type=str, default=None, help="Filter for each benchmark"
    )
    parser.add_argument(
        "--trials", type=int, required=True, help="Number of trials to consider"
    )
    parser.add_argument(
        "--min_trials", type=int, default=0, help="Starting point of the trials"
    )
    parser.add_argument(
        "--num_seeds", type=int, default=None, help="Number of seeds to consider"
    )
    parser.add_argument(
        "--blacklists",
        nargs="+",
        type=str,
        default=None,
        help="Blacklist for each benchmark",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default=None,
        help="Filename to store the plot",
    )
    parser.add_argument(
        "--normalization_method",
        type=str,
        default=None,
        help="Normalization method for each benchmark (default: minmax)",
    )
    parser.add_argument(
        "--nb201_device_metrics",
        nargs="+",
        type=str,
        default=None,
        help="Device metric for nb201 normalization",
    )
    parser.add_argument(
        "--x_lims",
        nargs="+",
        type=str,
        default=None,
        help="X-axis limits for each subplot (as pairs)",
    )
    parser.add_argument(
        "--y_lims",
        nargs="+",
        type=str,
        default=None,
        help="Y-axis limits for each subplot (as pairs)",
    )
    parser.add_argument(
        "--whitelists",
        nargs="+",
        type=str,
        default=None,
        help="Whitelist for each benchmark",
    )
    parser.add_argument(
        "--simplify_legend",
        action="store_true",
        help="Simplify legend to show only 'MOHOLLM' and 'LLM' with red and orange colors",
    )
    args = parser.parse_args()
    num_seeds = args.num_seeds

    n = len(args.benchmarks)
    ncols = 6
    nrows = math.ceil(n / ncols)

    # If there's only one benchmark, create a single standard figure/axes
    # (not a grid of subplots). This makes the figure behave like a normal
    # matplotlib figure and avoids layout/legend issues specific to grids.
    if n == 1:
        fig, ax = plt.subplots(1, 1, figsize=(6 * 1, 4.5 * 1), sharex=True)
        axes = [ax]
    else:
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(4.5 * ncols, 4 * nrows), sharex=True
        )
        # Normalize axes to a flat list of Axes objects
        if isinstance(axes, np.ndarray):
            axes = axes.flatten().tolist()
        else:
            axes = [axes]

    all_handles = None
    all_labels = None

    reference_points_benchmarks = {}

    for idx in range(n):
        benchmark = args.benchmarks[idx]
        title = args.titles[idx]
        data_path = args.data_paths[idx]
        filter = args.filters[idx] if args.filters else None
        trials = args.trials
        min_trials = args.min_trials
        print(min_trials)
        # filename = args.filenames[idx] if args.filenames else ""  # unused
        normalization_method = (
            args.normalization_method if args.normalization_method else "minmax"
        )
        blacklist = args.blacklists[idx].split(",") if args.blacklists else []
        whitelist = args.whitelists[idx].split(",") if args.whitelists else []
        x_lim = args.x_lims[idx * 2 : idx * 2 + 2] if args.x_lims else None
        y_lim = args.y_lims[idx * 2 : idx * 2 + 2] if args.y_lims else None
        nb201_device_metric = (
            args.nb201_device_metrics[idx] if args.nb201_device_metrics else ""
        )

        file_names = glob.glob(f"{data_path}/**/*.csv", recursive=True)
        observed_fvals_files = [
            file for file in file_names if "observed_fvals" in file.split("/")
        ]
        method_files = group_data_by_method(observed_fvals_files, filter)
        if num_seeds:
            method_dfs = {
                method: [
                    df.drop(columns=["configs"]) if "configs" in df.columns else df
                    for df in [pd.read_csv(file)[min_trials:trials] for file in files][
                        :num_seeds
                    ]
                ]
                for method, files in method_files.items()
            }
        else:
            method_dfs = {
                method: [
                    df.drop(columns=["configs"]) if "configs" in df.columns else df
                    for df in [pd.read_csv(file)[min_trials:trials] for file in files]
                ]
                for method, files in method_files.items()
            }

        # Dynamically determine objective columns
        first_method = next(iter(method_dfs))
        first_df = method_dfs[first_method][0]
        columns = [col for col in first_df.columns if col != "configs"]
        print(
            f"Number of objectives: {len(columns)}, {columns} for benchmark {benchmark}"
        )

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
        min_max_values = {
            column: {
                "min": min(
                    [
                        min([df[column].min() for df in dfs])
                        for _, dfs in method_dfs.items()
                    ]
                ),
                "max": max(
                    [
                        max([df[column].max() for df in dfs])
                        for _, dfs in method_dfs.items()
                    ]
                ),
            }
            for column in columns
        }
        # Compute reference dir for the benchmark
        reference_points = {
            column: (d.get("max") * (1.01)) for column, d in min_max_values.items()
        }
        reference_points_benchmarks[benchmark] = reference_points
        print(
            f"Reference point for {benchmark}: {reference_points} from min max values: {min_max_values}"
        )

        hypervolume_over_time = {}
        for method, dfs in method_dfs.items():
            method_hv = []
            for df in dfs:
                reference_points = [1.0] * len(columns)
                if normalization_method == "minmax":
                    df = normalize_data(df, min_max_values)
                elif normalization_method == "nb201":
                    df = normalize_data_nb201(df, nb201_device_metric)
                elif normalization_method == "reference_point":
                    reference_points = HV_REFERENCE_POINTS.get(benchmark)
                    if not reference_points:
                        print(f"No reference point available for: {benchmark}")
                        reference_points = [1.0] * len(columns)

                method_hv.append(
                    convert_data_to_hv_over_time(
                        df[:trials], reference_point=reference_points
                    )
                )
            hypervolume_over_time[method] = np.array(method_hv)
        mean_hypervolume_over_time = {
            method: np.mean(hv, axis=0) for method, hv in hypervolume_over_time.items()
        }
        std_hypervolume_over_time = {
            method: np.std(hv, axis=0) for method, hv in hypervolume_over_time.items()
        }
        data = {"mean": mean_hypervolume_over_time, "std": std_hypervolume_over_time}

        # Compute dynamic y_lim to make the plots more readable
        # Only calculate dynamic y_lim if no specific y_lim was provided
        if y_lim is None:
            # 1. Define a "burn-in" period to ignore for finding the y_min.
            #    This prevents the initial low values from stretching the y-axis.
            #    Let's ignore the first 90% of evaluations, but at least the first one.
            burn_in_period = max(1, int(0.90 * trials))

            # 2. Find the min and max hypervolume across all methods for this subplot
            all_final_values = []
            for method_hv in hypervolume_over_time.values():
                # method_hv is a 2D array (num_runs, num_evaluations)
                # We look at all values after the burn-in period
                all_final_values.extend(method_hv[:, burn_in_period:].flatten())

            if all_final_values:
                # Instead of np.min(), we take the 5th percentile
                y_min_robust = np.quantile(all_final_values, 0.03)
                # Instead of np.max(), we take the 95th percentile
                y_max_robust = np.quantile(all_final_values, 0.90)

                # Add a small padding to the new robust limits
                # Use a slightly larger padding since we've zoomed in
                padding = (y_max_robust - y_min_robust) * 0.05  # 5% padding

                if normalization_method == "reference_point":
                    final_y_max = y_max_robust + padding
                else:
                    # Ensure padding doesn't push max over 1.0
                    final_y_max = max(1, y_max_robust + padding)

                final_y_min = y_min_robust - padding

                y_lim_dynamic = [final_y_min, final_y_max]
            else:
                # Fallback in case there is no data
                y_lim_dynamic = None
        else:
            # If y_lim was provided via command line, use it
            y_lim_dynamic = y_lim

        if min_trials and min_trials > 0:
            x_lim = [min_trials, trials]
        handles, labels = create_hv_subplot(
            idx,
            axes[idx],
            data,
            benchmark,
            title,
            x_lim,
            y_lim_dynamic,
            simplify_legend=args.simplify_legend,
        )
        if all_handles is None:
            all_handles = handles
            all_labels = labels

    # Hide unused axes if any
    for ax in axes[n:]:
        ax.set_visible(False)

    # Deduplicate legend entries if simplify_legend is enabled
    if args.simplify_legend:
        unique_labels = []
        unique_handles = []
        seen_labels = set()
        for handle, label in zip(all_handles, all_labels):
            if label not in seen_labels:
                unique_labels.append(label)
                unique_handles.append(handle)
                seen_labels.add(label)
        all_handles = unique_handles
        all_labels = unique_labels

    if len(all_labels) <= 4:
        ncols = len(all_labels)
    else:
        ncols = max(4, len(all_labels) / 4)

    # Enhanced subplot spacing for better visual hierarchy
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(bottom=0.3, wspace=0.10, hspace=0.35)

    # Enhanced legend with professional styling
    legend = fig.legend(
        all_handles,
        all_labels,
        loc="outside lower center",
        ncol=8,
        bbox_to_anchor=(0.43, -0.19),
        frameon=True,
        framealpha=0.98,
        edgecolor="#666666",
        fancybox=True,
        shadow=True,
        columnspacing=1.2,
        handletextpad=0.6,
        borderpad=1.0,
        borderaxespad=0.8,
    )
    legend.get_frame().set_linewidth(1.2)
    legend.get_frame().set_facecolor("#FEFEFE")

    plt.tight_layout()
    for file_type in ["svg", "pdf", "png"]:
        dir_path = f"./plots/hypervolume_over_time_subplots/{file_type}/{args.filename}.{file_type}"
        os.makedirs(os.path.dirname(dir_path), exist_ok=True)
        plt.savefig(dir_path, dpi=300, bbox_inches="tight")

    plt.close()


if __name__ == "__main__":
    main()
