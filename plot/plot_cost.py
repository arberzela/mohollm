import argparse
import json
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pymoo.indicators.hv import HV
from plot_settings import COLORS, LINE_STYLES, LABEL_MAP

# Use LaTeX for text rendering
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 8,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    }
)


def main():
    """
    Main function to plot hypervolume over time.

    This function parses command-line arguments to get the benchmark name, plot title,
    data path, output path, and an optional filter string. It then gathers all CSV files
    from the specified data path, filters them based on the presence of "observed_fvals"
    in their names, and groups them by method name. The grouped file names are stored
    in a JSON file in the current directory.

    Command-line arguments:
    --benchmark (str): Benchmark name.
    --title (str): Plot title.
    --data_path (str): Path to the folder containing observed fvals CSV files.
    --output_path (str): Path to save the plot.
    --filter (str, optional): Filter to only include files containing this string.

    """
    parser = argparse.ArgumentParser(description="Plot hypervolume over time.")
    parser.add_argument("--benchmark", type=str, required=True, help="Benchmark name")
    parser.add_argument("--title", type=str, required=True, help="Plot title")
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to observed fvals folder"
    )
    parser.add_argument(
        "--filter", type=str, help="Filter to only include files containing this string"
    )
    parser.add_argument("--trials", type=int, help="Number of trials to consider")
    parser.add_argument(
        "--blacklist", default="", type=str, help="Models or methods not to plot"
    )
    parser.add_argument("--filename", default="", type=str, help="Filename of the plot")

    parser.add_argument(
        "--whitelist",
        default="",
        type=str,
        help="Only plot models or methods in this list",
    )

    args = parser.parse_args()

    data_path = args.data_path
    benchmark = args.benchmark
    title = args.title
    filter = args.filter
    trials = args.trials
    filename = args.filename
    blacklist = args.blacklist.split(",")
    whitelist = args.whitelist.split(",")
    output_path = f"./plots/total_cost/{benchmark}"

    # NB201 specific

    print("#" * 80)
    print("Plotting hypervolume over time")
    print(f"Benchmark:   {benchmark}")
    print(f"Title:       {title}")
    print(f"Data Path:   {data_path}")
    print(f"Filename:    {filename}")
    print(f"Output Path: {output_path}")
    print(f"Filter:      {filter}")
    print(f"Trials:      {trials}")
    print(f"Blacklist:   {blacklist}")
    print(f"Whitelist:   {whitelist}\n")

    # Gather all CSV files from folder
    file_names = glob.glob(f"{data_path}/**/*.csv", recursive=True)

    observed_fvals_files = [
        file for file in file_names if "cost_per_request" in file.split("/")
    ]

    # Group the files names into a dict of {method_name: [list of file names]}
    method_files = group_data_by_method(observed_fvals_files, filter)
    # 1. Convert to pandas DataFrames
    method_dfs = {
        method: [pd.read_csv(file)[:trials] for file in files]
        for method, files in method_files.items()
    }

    if len(whitelist) > 0 and whitelist[0] != "":  # Prioritize whitelist if provided
        filtered_dfs = {}
        for method, dfs in method_dfs.items():
            if any(entry == method for entry in whitelist):
                filtered_dfs[method] = dfs
            else:
                print(f"Filtered out method: {method} due to not being in whitelist")
        method_dfs = filtered_dfs
    elif (
        len(blacklist) > 0 and blacklist[0] != ""
    ):  # For some reason the blacklist has always the entry "" if empty
        filtered_dfs = {}
        for method, dfs in method_dfs.items():
            if not any(entry in method for entry in blacklist):
                filtered_dfs[method] = dfs
            else:
                print(f"Filtered out method: {method} due to blacklist")
        method_dfs = filtered_dfs

    create_cost_plot(method_dfs, benchmark, title, output_path, filename)


def create_cost_plot(data, benchmark, title, path, filename):
    """
    Create publication-ready hypervolume over time plots.

    Parameters:
        data (dict): Dictionary containing 'mean' and 'std' of hypervolume data
        benchmark (str): Name of the benchmark (not currently used but preserved for future use)
        title (str): Title of the plot
        path (str): Directory path to save the plot
        filename (str): Filename for saving the plot
    """
    # Create figure with appropriate size for single-column journals
    fig, ax = plt.subplots(figsize=(5, 3.5))

    # Get methods and sort them alphabetically
    methods = sorted(data.keys())
    for i, method in enumerate(methods):
        ax.plot(
            data[method][0]["total_cost"],
            label=LABEL_MAP.get(method, method),
            color=COLORS[i % len(COLORS)],
            linestyle=LINE_STYLES[i % len(LINE_STYLES)],
            linewidth=1.2,
        )

    # Add labels and title with LaTeX formatting
    ax.set_xlabel("Requests")
    ax.set_ylabel(r"Total Cost in \$ ($\times 1/10^6$)")
    ax.set_title(title)

    # Improve grid appearance
    ax.grid(True, linestyle="--", alpha=0.6, linewidth=0.4)

    # Format tick labels
    ax.xaxis.set_major_locator(plt.MaxNLocator(10))
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{(x * 1000000):.2f}:")
    )

    ax.legend(
        loc="lower right",
        frameon=True,
        framealpha=0.95,
        edgecolor="k",
        fancybox=False,
        ncol=1,
    )

    plt.tight_layout()

    for file_type in ["svg", "pdf", "png"]:
        dir_path = f"{path}/{file_type}/{filename}.{file_type}"
        os.makedirs(os.path.dirname(dir_path), exist_ok=True)
        plt.savefig(dir_path, dpi=300, bbox_inches="tight")

    plt.close()


def extract_method_name(files):
    """
    If everything is correct the methods name should be the third to last element in the path
    """
    methods = []
    for file in files:
        method = file.split("/")[-3]
        if method not in methods:
            methods.append(method)
    return methods


def group_data_by_method(observed_fvals_files, filter):
    """
    Groups a list of file paths by method names extracted from the file paths.
    Args:
        observed_fvals_files (list of str): List of file paths to be grouped.
        filter (str): Optional filter string to include only files containing this substring.
    Returns:
        dict: A dictionary where keys are method names and values are lists of file paths
              corresponding to each method.
    Example:
        observed_fvals_files = [
            "/path/to/method1/file1.txt",
            "/path/to/method2/file2.txt",
            "/path/to/method1/file3.txt"
        ]
        filter = "file"
        result = group_data_by_method(observed_fvals_files, filter)
        # result will be:
        # {
        #     "method1": ["/path/to/method1/file1.txt", "/path/to/method1/file3.txt"],
        #     "method2": ["/path/to/method2/file2.txt"]
        # }
    """

    method_names = extract_method_name(observed_fvals_files)
    method_files = {}
    for method in method_names:
        files = []
        for file in observed_fvals_files:
            # Includes file is the method name matches the file path fully (hence the split)
            if method in file.split("/") and (filter in file if filter else True):
                files.append(file)
        method_files[method] = files
    return method_files


if __name__ == "__main__":
    main()
