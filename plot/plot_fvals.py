import argparse
import json
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from normalize_data_nb201 import normalize_data_nb201
from pymoo.indicators.hv import HV
from plot_settings import COLORS, LINE_STYLES

# Use LaTeX for text rendering
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 3,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    }
)


def create_fval_over_time_plot(
    data, benchmark, title, path, filename, x_lim, y_lim, use_log_scale
):
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
    methods = sorted(data["mean"].keys())

    for i, method in enumerate(methods):
        print(f"Plotting: {method}")
        mean_values = data["mean"][method]
        std_values = data["std"][method]
        trials = range(len(mean_values))

        # Mean hypervolume line
        ax.plot(
            trials,
            mean_values,
            label=method,
            color=COLORS[i % len(COLORS)],
            linestyle=LINE_STYLES[i % len(LINE_STYLES)],
            linewidth=1.5,
        )
        # Confidence interval TODO: Add this back
        # ax.fill_between(
        #     trials,
        #     np.array(mean_values) + np.array(std_values),
        #     np.array(mean_values) - np.array(std_values),
        #     color=COLORS[i % len(COLORS)],
        #     alpha=0.3,
        # )

    # Add labels and title with LaTeX formatting
    ax.set_xlabel("Trials")
    ax.set_ylabel("Fval")
    if use_log_scale:
        print("Using log scale for y-axis")
        ax.set_yscale("log")
    ax.set_title(title)

    if x_lim:
        ax.set_xlim(float(x_lim[0]), float(x_lim[1]))
    if y_lim:
        ax.set_ylim(float(y_lim[0]), float(y_lim[1]))

    # Improve grid appearance
    ax.grid(True, linestyle="--", alpha=0.6, linewidth=0.4)

    # Format tick labels
    ax.xaxis.set_major_locator(plt.MaxNLocator(10))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.4f}"))

    ax.legend(
        loc="upper right",
        frameon=True,
        framealpha=0.95,
        edgecolor="k",
        fancybox=False,
        ncol=1,
    )

    plt.tight_layout()

    for file_type in ["svg", "pdf", "png"]:
        dir_path = f"{path}/{filename}.{file_type}"
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
        "--blacklist", default="", type=str, help="Models or methods not to plot"
    )
    parser.add_argument("--filename", default="", type=str, help="Filename of the plot")
    parser.add_argument(
        "--normalization_method",
        default="minmax",
        choices=["minmax", "nb201", "none"],
        type=str,
        help="Normalization method to choose. Choices: minmax, nb201, none",
    )
    parser.add_argument(
        "--nb201_device_metric",
        default="",
        type=str,
        help="Device metric to use for normalization when using nb201 normalization method",
    )
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
        "--use_log_scale",
        default="False",
        type=str,
        help="Use log scale for the y-axis",
    )

    args = parser.parse_args()

    data_path = args.data_path
    benchmark = args.benchmark
    title = args.title
    filter = args.filter
    trials = args.trials
    filename = args.filename
    normalization_method = args.normalization_method
    columns = args.columns.split(",")
    blacklist = args.blacklist.split(",")
    output_path = f"./plots/fvals_over_time/{benchmark}"
    x_lim = args.x_lim if args.x_lim else None
    y_lim = args.y_lim if args.y_lim else None
    use_log_scale = args.use_log_scale == "True"

    # NB201 specific
    nb201_device_metric = args.nb201_device_metric

    print("#" * 80)
    print("Plotting hypervolume over time")
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
    # Group the files names into a dict of {method_name: [list of file names]}
    method_files = group_data_by_method(observed_fvals_files, filter)
    # 1. Convert to pandas DataFrames
    method_dfs = {
        method: [pd.read_csv(file, usecols=columns)[:trials] for file in files]
        for method, files in method_files.items()
    }

    # # Save method_dfs as JSON
    # method_dfs_serializable = {}
    # for method, dfs in method_dfs.items():
    #     method_dfs_serializable[method] = [df.max().to_dict() for df in dfs]

    # # Save to JSON file in the root directory
    # with open("method_dfs.json", "w") as f:
    #     json.dump(method_dfs_serializable, f, indent=4)

    # print(f"Saved method_dfs as JSON to {os.path.abspath('method_dfs.json')}")

    if (
        len(blacklist) > 0 and blacklist[0] != ""
    ):  # For some reason the blacklist has always the entry "" if empty
        filtered_dfs = {}
        for method, dfs in method_dfs.items():
            if not any(entry in method for entry in blacklist):
                filtered_dfs[method] = dfs
            else:
                print(f"Filtered out method: {method} due to blacklist")
        method_dfs = filtered_dfs

    # JUST FOR DEBUGGING
    # for col in columns:
    #     for _, dfs in method_dfs.items():
    #         print()
    #         print()
    #         print("Problem: ", filter)
    #         print(
    #             "min: ",
    #             [min([df[col].min() for df in dfs]) for _, dfs in method_dfs.items()],
    #         )
    #         print(
    #             "max: ",
    #             [max([df[col].max() for df in dfs]) for _, dfs in method_dfs.items()],
    #         )

    # Calculate the max and min values for each columns across all dataframes
    # 2. extract the min and max values for each column
    min_max_values = {
        column: {
            "min": min(
                [min([df[column].min() for df in dfs]) for _, dfs in method_dfs.items()]
            ),
            "max": max(
                [max([df[column].max() for df in dfs]) for _, dfs in method_dfs.items()]
            ),
        }
        for column in columns
    }
    print("Calculated min_max_values across all datasets:", min_max_values)
    # 3. Calculate the hypervolume over time for each method and each dataframe
    method_fvals = {}
    for method, dfs in method_dfs.items():
        method_fval = []
        for df in dfs:
            # 3. 1. Normalize the data for each dataframe
            if normalization_method == "minmax":
                df = normalize_data(df, min_max_values)
            elif normalization_method == "nb201":
                df = normalize_data_nb201(df, nb201_device_metric)
            method_fval.append(np.minimum.accumulate(df[:trials].values).flatten())
        method_fvals[method] = np.array(method_fval)
    # 4. Compute the mean and standard deviation of the hypervolume over time for each method
    mean_hypervolume_over_time = {
        method: np.mean(fvals, axis=0) for method, fvals in method_fvals.items()
    }
    std_hypervolume_over_time = {
        method: np.std(fvals, axis=0) for method, fvals in method_fvals.items()
    }

    data = {
        "mean": mean_hypervolume_over_time,
        "std": std_hypervolume_over_time,
    }

    create_fval_over_time_plot(
        data, benchmark, title, output_path, filename, x_lim, y_lim, use_log_scale
    )


def normalize_data(fvals, min_max_metrics):
    """
    Normalize the data in `fvals` based on the minimum and maximum values provided in `min_max_metrics`.
    Parameters:
    fvals (pd.DataFrame): A DataFrame containing the data to be normalized.
    min_max_metrics (dict): A dictionary where keys are column names and values are dictionaries with "min" and "max" keys
                            representing the minimum and maximum values for normalization.
    Returns:
    pd.DataFrame: The normalized DataFrame.
    """
    for column, values in min_max_metrics.items():
        min_val = values["min"]
        max_val = values["max"]
        fvals[column] = (fvals[column] - min_val) / (max_val - min_val)
    return fvals


def extract_method_name(files):
    """
    If everything is correct the methods name should be the second to last element in the path
    """
    methods = []

    for file in files:
        method = file.split("/")[
            4
        ]  # It is 4 for synetune as we have one more subfolder
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
