import argparse
import json
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from normalize_data_nb201 import normalize_data_nb201
from pymoo.indicators.hv import HV
from plot_settings import COLORS, LINE_STYLES, LABEL_MAP

# Use LaTeX for text rendering
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 11,
        "axes.labelsize": 14,  # Increased for better readability
        "axes.titlesize": 14,
        "legend.fontsize": 10,  # Corrected from 3 to a readable size
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    }
)


def compute_hypervolume(df, reference_point):
    """Computes the hypervolume for a set of objectives."""
    objectives = df.to_numpy()
    ind = HV(ref_point=reference_point)
    hypervolume = ind(objectives)
    return hypervolume


def convert_data_to_log_hv_diff_over_time(fvals):
    """
    Computes the Log Hypervolume Difference over time.
    The difference is calculated against a maximum hypervolume of 1.0,
    which is the theoretical maximum for normalized objectives with a [1.0, 1.0] reference point.
    """
    log_hypervolume_difference = []
    # The maximum hypervolume with a reference point of [1,1] is 1.0
    hv_max = 1.0
    for step in range(1, len(fvals) + 1):
        hv_current = compute_hypervolume(fvals.iloc[:step], [1.0, 1.0])
        hv_diff = hv_max - hv_current
        # Add a small epsilon to avoid log(0)
        log_hypervolume_difference.append(np.log10(hv_diff + 1e-9))
    return log_hypervolume_difference


def create_log_hv_diff_plot(data, benchmark, title, path, filename, x_lim, y_lim):
    """
    Create publication-ready log hypervolume difference over time plots.
    This version uses error bars instead of a shaded region for standard deviation.
    """
    # Create figure with a size suitable for journals
    fig, ax = plt.subplots(figsize=(6, 4.5))

    # Get methods and sort them for consistent plotting order
    methods = sorted(data["mean"].keys())

    for i, method in enumerate(methods):
        print(f"Plotting: {method}")
        mean_values = data["mean"][method]
        std_values = data["std"][method]
        std_error = 1.96 * std_values / np.sqrt(len(mean_values))
        evaluations = range(len(mean_values))

        # Plot the mean line
        ax.plot(
            evaluations,
            mean_values,
            label=LABEL_MAP.get(method, method),
            color=COLORS[i % len(COLORS)],
            linestyle=LINE_STYLES[i % len(LINE_STYLES)],
            linewidth=1.2,
        )

        # Add the shaded confidence interval
        ax.fill_between(
            evaluations,
            mean_values - std_error,
            mean_values + std_error,
            color=COLORS[i % len(COLORS)],
            alpha=0.2,  # Set transparency for the shaded area
        )

    # Add labels and title with LaTeX formatting, matching the example
    ax.set_xlabel("Trials")
    ax.set_ylabel("Log Hypervolume Difference")
    ax.set_title(title)

    if x_lim:
        ax.set_xlim(float(x_lim[0]), float(x_lim[1]))
    if y_lim:
        ax.set_ylim(float(y_lim[0]), float(y_lim[1]))

    # Add a grid with a subtle style
    ax.grid(True, linestyle="--", alpha=0.7, linewidth=0.5)

    # Configure tick locators and formatters for a clean look
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=6))
    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=6))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}"))

    # Place the legend below the plot with multiple columns
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=4,  # Adjust the number of columns as needed
        frameon=True,
        framealpha=0.95,
        edgecolor="black",
        fancybox=True,
        fontsize=8,
        title="Methods",
        title_fontsize=10,
    )
    plt.tight_layout(rect=[0, 0.08, 1, 1])  # Leave space for the legend

    # Save the plot in multiple formats
    for file_type in ["svg", "pdf", "png"]:
        dir_path = f"{path}/{file_type}/{filename}.{file_type}"
        os.makedirs(os.path.dirname(dir_path), exist_ok=True)
        plt.savefig(dir_path, dpi=300, bbox_inches="tight")

    plt.close()


def main():
    """
    Main function to plot log hypervolume difference over time.
    """
    parser = argparse.ArgumentParser(
        description="Plot log hypervolume difference over time."
    )
    parser.add_argument("--benchmark", type=str, required=True, help="Benchmark name")
    parser.add_argument("--title", type=str, required=True, help="Plot title")
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to observed fvals folder"
    )
    parser.add_argument(
        "--filter", type=str, help="Filter to only include files containing this string"
    )
    parser.add_argument(
        "--columns", type=str, required=True, help="Columns to read from the CSV files"
    )
    parser.add_argument("--trials", type=int, help="Number of trials to consider")
    parser.add_argument(
        "--blacklist", default="", type=str, help="Models or methods not to plot"
    )
    parser.add_argument("--filename", default="", type=str, help="Filename of the plot")
    parser.add_argument(
        "--normalization_method",
        default="minmax",
        choices=["minmax", "nb201", "none"],
        type=str,
        help="Normalization method.",
    )
    parser.add_argument(
        "--nb201_device_metric",
        default="",
        type=str,
        help="Device metric for nb201 normalization.",
    )
    parser.add_argument(
        "--x_lim", default=None, nargs=2, type=str, help="X-axis limits."
    )
    parser.add_argument(
        "--y_lim", default=None, nargs=2, type=str, help="Y-axis limits."
    )
    parser.add_argument(
        "--whitelist",
        default="",
        type=str,
        help="Only plot models or methods in this list.",
    )

    args = parser.parse_args()

    # --- Argument parsing and setup ---
    data_path = args.data_path
    benchmark = args.benchmark
    title = args.title
    filter_str = args.filter
    trials = args.trials
    filename = args.filename if args.filename else benchmark
    normalization_method = args.normalization_method
    columns = args.columns.split(",")
    blacklist = args.blacklist.split(",")
    whitelist = args.whitelist.split(",")
    output_path = f"./plots/log_hypervolume_difference_over_time/{benchmark}"
    x_lim = args.x_lim
    y_lim = args.y_lim
    nb201_device_metric = args.nb201_device_metric

    print("#" * 80)
    print("Plotting Log Hypervolume Difference over time")
    print(f"Benchmark:   {benchmark}")
    print(f"Title:       {title}")
    print(f"Data Path:   {data_path}")
    print(f"Filename:    {filename}")
    print(f"Output Path: {output_path}")
    print(f"Filter:      {filter_str}")
    print(f"Columns:     {columns}")
    print(f"Trials:      {trials}")
    print(f"Blacklist:   {blacklist}")
    print(f"Whitelist:   {whitelist}\n")

    # --- Data loading and grouping ---
    file_names = glob.glob(f"{data_path}/**/*.csv", recursive=True)
    observed_fvals_files = [f for f in file_names if "observed_fvals" in f.split("/")]
    method_files = group_data_by_method(observed_fvals_files, filter_str)
    method_dfs = {
        method: [pd.read_csv(file, usecols=columns)[:trials] for file in files]
        for method, files in method_files.items()
    }

    # --- Filtering methods based on whitelist/blacklist ---
    if whitelist and whitelist[0]:
        method_dfs = {m: dfs for m, dfs in method_dfs.items() if m in whitelist}
    elif blacklist and blacklist[0]:
        method_dfs = {
            m: dfs
            for m, dfs in method_dfs.items()
            if not any(b in m for b in blacklist)
        }

    # --- Normalization setup ---
    min_max_values = {
        col: {
            "min": min(
                min(df[col].min() for df in dfs) for _, dfs in method_dfs.items()
            ),
            "max": max(
                max(df[col].max() for df in dfs) for _, dfs in method_dfs.items()
            ),
        }
        for col in columns
    }
    print("Calculated min/max values across all datasets:", min_max_values)

    # --- Main calculation loop ---
    log_hv_diff_over_time = {}
    for method, dfs in method_dfs.items():
        method_log_diff = []
        for df in dfs:
            if normalization_method == "minmax":
                df_normalized = normalize_data(df.copy(), min_max_values)
            elif normalization_method == "nb201":
                df_normalized = normalize_data_nb201(df.copy(), nb201_device_metric)
            else:  # 'none'
                df_normalized = df.copy()

            method_log_diff.append(
                convert_data_to_log_hv_diff_over_time(df_normalized[:trials])
            )
        log_hv_diff_over_time[method] = np.array(method_log_diff)

    # --- Aggregating results (mean and std) ---
    mean_log_hv_diff = {
        m: np.mean(hv, axis=0) for m, hv in log_hv_diff_over_time.items()
    }
    std_log_hv_diff = {m: np.std(hv, axis=0) for m, hv in log_hv_diff_over_time.items()}

    data_for_plotting = {"mean": mean_log_hv_diff, "std": std_log_hv_diff}

    # --- Plotting ---
    create_log_hv_diff_plot(
        data_for_plotting, benchmark, title, output_path, filename, x_lim, y_lim
    )
    print("\nPlotting complete.")
    print("#" * 80)


def normalize_data(fvals, min_max_metrics):
    """Normalizes DataFrame columns using pre-calculated min/max values."""
    for column, values in min_max_metrics.items():
        min_val, max_val = values["min"], values["max"]
        fvals[column] = (fvals[column] - min_val) / (max_val - min_val)
    return fvals


def extract_method_name(files):
    """Extracts method names from file paths (assumes a specific path structure)."""
    methods = []
    for file in files:
        # Assumes path structure like: .../data/METHOD_NAME/seed_...
        method = file.split("/")[-3]
        if method not in methods:
            methods.append(method)
    return methods


def group_data_by_method(observed_fvals_files, file_filter):
    """Groups file paths by the extracted method name."""
    method_names = extract_method_name(observed_fvals_files)
    method_files = {}
    for method in method_names:
        files = [
            f
            for f in observed_fvals_files
            if method in f.split("/") and (file_filter in f if file_filter else True)
        ]
        if files:
            method_files[method] = files
    return method_files


if __name__ == "__main__":
    main()
