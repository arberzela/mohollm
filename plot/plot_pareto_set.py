import argparse
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from paretoset import paretoset
from plot_settings import COLORS, LINE_STYLES, MARKERS, LABEL_MAP

# Use LaTeX for text rendering
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 2,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    }
)


def create_pareto_set_plot(
    data, benchmark, title, path, filename, columns, x_lim, y_lim
):
    """
    Create publication-ready pareto set plot with proper COLORS, MARKERS, and formatting.

    Parameters:
    -----------
    data : dict
        Dictionary of seed -> methods -> dataframes
    benchmark : str
        Name of the benchmark
    title : str
        Plot title
    path : str
        Output path
    filename : str
        Base filename for the output
    columns : list
        List of column names to plot [x_column, y_column]
    """

    for seed, methods in data.items():
        fig, ax = plt.subplots(figsize=(5, 3.5))
        method_names = sorted(methods.keys())
        legend_elements = []

        for i, method_name in enumerate(method_names):
            color = COLORS[i % len(COLORS)]
            marker = MARKERS[i % len(MARKERS)]
            if not methods[method_name]:
                continue
            method_df = methods[method_name][0]
            # Compute Pareto front
            pareto_mask = paretoset(method_df, sense=["min", "min"])
            pareto_df = method_df[pareto_mask]
            non_pareto_df = method_df[~pareto_mask]

            # Plot non-Pareto points with lower alpha
            if len(non_pareto_df) > 0:
                ax.scatter(
                    non_pareto_df[columns[0]],
                    non_pareto_df[columns[1]],
                    color=color,
                    marker=marker,
                    s=30,
                    alpha=0.3,
                    label=f"{LABEL_MAP.get(method_name, method_name)} (non-Pareto)",
                )

            # Plot Pareto-optimal points
            if len(pareto_df) > 0:
                scatter = ax.scatter(
                    pareto_df[columns[0]],
                    pareto_df[columns[1]],
                    color=color,
                    marker=marker,
                    s=40,
                    alpha=0.9,
                    label=LABEL_MAP.get(method_name, method_name),
                )

                # Add this method to the legend
                legend_elements.append(scatter)

                # Connect Pareto-optimal points with a line if there are at least 2 points
                if len(pareto_df) > 1:
                    pareto_df_sorted = pareto_df.sort_values(columns[0])
                    ax.plot(
                        pareto_df_sorted[columns[0]],
                        pareto_df_sorted[columns[1]],
                        color=color,
                        alpha=0.9,
                        linewidth=1.5,
                        linestyle=LINE_STYLES[i % len(LINE_STYLES)],
                    )

        ax.set_xlabel(columns[0])
        ax.set_ylabel(columns[1])
        ax.set_title(f"{title}", fontweight="bold")

        if x_lim:
            ax.set_xlim(float(x_lim[0]), float(x_lim[1]))
        if y_lim:
            ax.set_ylim(float(y_lim[0]), float(y_lim[1]))

        # Improve grid appearance
        ax.grid(True, linestyle="--", alpha=0.7, linewidth=0.5, color="gray")

        # Add spines
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)

        # Add legend - sorted by method name
        print(legend_elements)
        ax.legend(
            handles=legend_elements,
            labels=method_names,
            loc="best",
            frameon=True,
            framealpha=0.95,
            edgecolor="black",
            fancybox=False,
            ncol=1,
            fontsize=6,
            title="Methods",
            title_fontsize=6,
        )

        plt.tight_layout()

        # Save in multiple formats
        for file_type in ["svg", "pdf", "png"]:
            dir_path = f"{path}/{filename}-{seed}.{file_type}"
            os.makedirs(os.path.dirname(dir_path), exist_ok=True)
            plt.savefig(dir_path, dpi=300, bbox_inches="tight")

        plt.close()


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
    parser.add_argument(
        "--columns", type=str, help="Columns to read from the CSV files"
    )
    parser.add_argument("--trials", type=int, help="Number of trials to consider")

    parser.add_argument(
        "--blacklist_methods", default="", type=str, help="Methods not to plot"
    )
    parser.add_argument(
        "--blacklist", default="", type=str, help="Models or methods not to plot"
    )
    parser.add_argument("--filename", default="", type=str, help="Filename of the plot")
    parser.add_argument(
        "--x_lim",
        default=None,
        nargs=2,
        type=str,
        help="X-axis limits for the plot",
    )
    parser.add_argument(
        "--y_lim",
        default=None,
        nargs=2,
        type=str,
        help="Y-axis limits for the plot",
    )
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
    columns = args.columns.split(",")
    blacklist = args.blacklist.split(",")
    whitelist = args.whitelist.split(",")
    x_lim = args.x_lim if args.x_lim else None
    y_lim = args.y_lim if args.y_lim else None

    output_path = f"./plots/pareto_set/{benchmark}"

    print("#" * 80)
    print("Plotting Pareto Set")
    print(f"Benchmark:   {benchmark}")
    print(f"Title:       {title}")
    print(f"Data Path:   {data_path}")
    print(f"Filename:    {filename}")
    print(f"Output Path: {output_path}")
    print(f"Filter:      {filter}")
    print(f"Columns:     {columns}")
    print(f"Trials:      {trials}")
    print(f"Blacklist:   {blacklist}\n")

    # Gather all CSV files from folder
    file_names = glob.glob(f"{data_path}/**/*.csv", recursive=True)
    observed_fvals_files = [
        file for file in file_names if "observed_fvals" in file.split("/")
    ]

    # Group the files names into a dict of {seed: {method_name: [list of file names]}}
    seed_method_files = group_data_by_seed_and_method(observed_fvals_files, filter)

    # Convert to pd dataframes for each seed
    seed_method_dfs = {}
    for seed, methods in seed_method_files.items():
        method_dfs = {
            method_name: [
                pd.read_csv(file, usecols=columns)[:trials] for file in method_files
            ]
            for method_name, method_files in methods.items()
        }
        if (
            len(whitelist) > 0 and whitelist[0] != ""
        ):  # Prioritize whitelist if provided
            filtered_dfs = {}
            for method, dfs in method_dfs.items():
                if any(entry == method for entry in whitelist):
                    filtered_dfs[method] = dfs
                else:
                    print(
                        f"Filtered out method: {method} due to not being in whitelist"
                    )
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
        seed_method_dfs[seed] = method_dfs

    create_pareto_set_plot(
        seed_method_dfs, benchmark, title, output_path, filename, columns, x_lim, y_lim
    )


def extract_method_name(files):
    """
    If everything is correct the methods name should be the second to last element in the path
    """
    methods = []
    for file in files:
        method = file.split("/")[3]
        if method not in methods:
            methods.append(method)
    return methods


def extract_seeds(files):
    """
    Extracts unique seed values from a list of file paths.

    Example:
        ./results/NB201/RS/observed_fvals/RS_1080ti_32_latency_31415927.csv
        -> returns: 31415927

    Args:
        files (list of str): A list of file paths.
    Returns:
        list of str: A list of unique seed values extracted from the file paths.
    """

    seeds = []
    for file in files:
        # Magic string splitting assuming the seed is the last thing in the path before the filetype
        seed = file.split("/")[-1].split("_")[-1].split(".")[0]
        if seed not in seeds:
            seeds.append(seed)
    return seeds


def group_data_by_seed_and_method(observed_fvals_files, filter):
    method_names = extract_method_name(observed_fvals_files)
    seeds = extract_seeds(observed_fvals_files)
    seed_method_files = {}

    for seed in seeds:
        seed_files = {}
        for method in method_names:
            files = []
            for file in observed_fvals_files:
                # Includes file is the method name matches the file path fully (hence the split)
                if (
                    method in file.split("/")
                    and seed in file
                    and (filter in file if filter else True)
                ):
                    print(method, file)
                    files.append(file)
            seed_files[method] = files
        seed_method_files[seed] = seed_files
    return seed_method_files


if __name__ == "__main__":
    main()
