import argparse
import json
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from normalize_data_nb201 import normalize_data_nb201
from plot_settings import get_color, LINE_STYLE_HV, LABEL_MAP_HV
from plot_common import (
    PlotStyle,
    compute_hypervolume,
    convert_data_to_hv_over_time,
    normalize_data as norm_data,
    apply_axis_style,
    set_axis_limits,
    save_plot,
    group_data_by_method,
    gather_files,
    custom_sort_key,
)

# Apply default plotting style
PlotStyle.apply("default")


def create_hv_over_time_plot(data, benchmark, title, path, filename, x_lim, y_lim):
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
    fig, ax = plt.subplots(figsize=(5, 4.5))

    # Get methods and sort them using custom sort key
    methods = sorted(data["mean"].keys(), key=custom_sort_key)

    for i, method in enumerate(methods):
        print(f"Plotting: {method}")
        mean_values = data["mean"][method]
        std_values = data["std"][method]
        trials = range(len(mean_values))

        # Calculate standard error
        std_error = 1.96 * std_values / np.sqrt(len(mean_values))

        # Mean hypervolume line
        ax.plot(
            trials,
            mean_values,
            label=LABEL_MAP_HV.get(method, method),
            color=get_color(method, i),
            linestyle=LINE_STYLE_HV[i % len(LINE_STYLE_HV)],
            linewidth=1.0,
        )

        # Confidence interval using standard error
        ax.fill_between(
            trials,
            mean_values - std_error,
            mean_values + std_error,
            color=get_color(method, i),
            alpha=0.3,
            linewidth=0,
        )

    # Add labels and title with LaTeX formatting
    ax.set_xlabel("Function Evaluations")
    ax.set_ylabel("Hypervolume")
    ax.set_title(title)

    # Set axis limits using utility function
    set_axis_limits(ax, x_lim, y_lim)
    
    # Apply consistent axis styling
    apply_axis_style(ax)

    # Format tick labels
    ax.xaxis.set_major_locator(plt.MaxNLocator(10))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}"))

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        frameon=True,
        framealpha=0.95,
        edgecolor="k",
        fancybox=False,
        ncol=3,
    )

    plt.tight_layout()

    # Save plot using utility function
    save_plot(fig, path, filename, formats=["svg", "pdf", "png"])
    
    return fig


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
    normalization_method = args.normalization_method
    columns = args.columns.split(",")
    blacklist = args.blacklist.split(",")
    whitelist = args.whitelist.split(",")
    output_path = f"./plots/hypervolume_over_time/{benchmark}"
    x_lim = args.x_lim if args.x_lim else None
    y_lim = args.y_lim if args.y_lim else None

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
    print(f"Blacklist:   {blacklist}")
    print(f"Whitelist:   {whitelist}\n")

    # Gather all CSV files using utility function
    observed_fvals_files = gather_files(data_path, "observed_fvals*.csv", filter)

    # Group the files by method using utility function
    method_files = group_data_by_method(observed_fvals_files, filter)
    
    # 1. Convert to pandas DataFrames
    method_dfs = {
        method: [pd.read_csv(file, usecols=columns)[:trials] for file in files]
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
    hypervolume_over_time = {}
    for method, dfs in method_dfs.items():
        method_hv = []
        for df in dfs:
            # 3. 1. Normalize the data for each dataframe
            if normalization_method == "minmax":
                df = normalize_data(df, min_max_values)
            elif normalization_method == "nb201":
                df = normalize_data_nb201(df, nb201_device_metric)
            method_hv.append(
                convert_data_to_hv_over_time(
                    df[:trials], reference_point=[1.0] * len(columns)
                ),
            )
        print(f"Calculating HV over time for: {method}")
        hypervolume_over_time[method] = np.array(method_hv)

    # 4. Compute the mean and standard deviation of the hypervolume over time for each method
    mean_hypervolume_over_time = {
        method: np.mean(hv, axis=0) for method, hv in hypervolume_over_time.items()
    }
    std_hypervolume_over_time = {
        method: np.std(hv, axis=0) for method, hv in hypervolume_over_time.items()
    }

    data = {
        "mean": mean_hypervolume_over_time,
        "std": std_hypervolume_over_time,
    }

    create_hv_over_time_plot(
        data, benchmark, title, output_path, filename, x_lim, y_lim
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


# Note: extract_method_name and group_data_by_method are now imported from plot_common
# These functions have been moved to the common utilities module to reduce redundancy


if __name__ == "__main__":
    main()
