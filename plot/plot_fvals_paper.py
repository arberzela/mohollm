import argparse
import json
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from normalize_data_nb201 import normalize_data_nb201

# from plot_settings import COLORS, LINE_STYLES

# Use LaTeX for text rendering
plt.rcParams.update(
    {
        "text.usetex": False,
    }
)

COLORS = {
    # Our methods
    "RS": "dodgerblue",
    "REA": "sienna",
    "BORE": "darkcyan",
    "BOTorch": "olive",
    "CQR": "goldenrod",
    "TPE": "indigo",
    "Turbo": "purple",
    "LLM": "forestgreen",
    "LLM-25": "forestgreen",  # "Wuff"
    # "LLMKD": "crimson",
    "LLMKD": "black",  # We used black in the paper
    # "LLMKD-alpha-0.2": "crimson",
    # "LLMKD-alpha-0.5": "crimson",
    # "LLMKD-alpha-0.7": "crimson",
    # "LLMKD-alpha-1.0": "black",
    # # For alpha comparision
    "LLMKD-alpha-0.2": "darkcyan",
    "LLMKD-alpha-0.5": "sienna",
    "LLMKD-alpha-0.7": "purple",
    "LLMKD-alpha-1.0": "black",
    "LLMKD-1-partition-per-trial": "darkcyan",
    "LLMKD-3-partition-per-trial": "sienna",
    "LLMKD-5-partition-per-trial": "black",
    "LLMKD-7-partition-per-trial": "cornflowerblue",
    "LLMKD-k-1": "darkcyan",
    "LLMKD-k-3": "sienna",
    "LLMKD-k-5": "black",
    "LLMKD-k-7": "cornflowerblue",
    "LLMKD-k-10": "crimson",
    "LLMKD-m0-0.25*d": "darkcyan",
    "LLMKD-m0-0.5*d": "black",
    "LLMKD-m0-1*d": "cornflowerblue",
    "LLMKD-m0-1": "sienna",
    "RS + KD-Tree": "purple",
    "LLM-llama3.1-8b-llm": "#7F007F",
    "LLM-llama3.3-70b-llm": "#008B8B",
    "LLMKD-llama3.1-8b": "#9B4A24",
    "LLMKD-llama3.3-70b": "crimson",
    "LLMKD-qwen-30b": "#6495ED",
    "LLM-qwen-30b": "#FFA500",
    "LLMKD-exploitation": "darkcyan",
    "LLMKD-exploration": "sienna",
    "LLMKD-ucb1": "purple",
    "LLMKD-uniform_region_sampling": "crimson",
    "LLM (no-context)": "darkcyan",
    "LLMKD (no-context)": "sienna",
}
MARKERS = {
    # Our methods
    "RS": "x",
    "REA": "*",
    "BORE": "s",
    "BOTorch": ">",
    "CQR": "3",
    "TPE": "^",
    "Turbo": "h",
    "LLM": "x",
    "LLM-25": "x",  # "Wuff"
    "LLMKD": "o",
    "LLMKD-alpha-0.2": "o",
    "LLMKD-alpha-0.5": "o",
    "LLMKD-alpha-0.7": "o",
    "LLMKD-alpha-1.0": "o",
    "LLMKD-1-partition-per-trial": "o",
    "LLMKD-3-partition-per-trial": "o",
    "LLMKD-5-partition-per-trial": "o",
    "LLMKD-7-partition-per-trial": "o",
    "LLMKD-k-1": "o",
    "LLMKD-k-3": "o",
    "LLMKD-k-5": "o",
    "LLMKD-k-7": "o",
    "LLMKD-k-10": "o",
    "LLMKD-m0-0.25*d": "o",
    "LLMKD-m0-0.5*d": "o",
    "LLMKD-m0-1*d": "o",
    "LLMKD-m0-1": "o",
    "RS + KD-Tree": "o",
    "LLM-llama3.1-8b-llm": "o",
    "LLM-llama3.3-70b-llm": "o",
    "LLMKD-llama3.1-8b": "o",
    "LLMKD-llama3.3-70b": "o",
    "LLMKD-exploitation": "o",
    "LLMKD-exploration": "o",
    "LLMKD-ucb1": "o",
    "LLMKD-uniform_region_sampling": "o",
    "LLMKD-qwen-30b": "o",
    "LLM-qwen-30b": "o",
    "LLM (no-context)": "o",
    "LLMKD (no-context)": "o",
}
LSTYLE = {
    # Our methods
    "RS": "solid",
    "REA": "solid",
    "BORE": "solid",
    "BOTorch": "solid",
    "CQR": "solid",
    "TPE": "solid",
    "Turbo": "solid",
    "LLM": "solid",
    "LLM-25": "solid",
    "LLMKD": "solid",
    "LLMKD-alpha-0.2": "solid",
    "LLMKD-alpha-0.5": "solid",
    "LLMKD-alpha-0.7": "solid",
    "LLMKD-alpha-1.0": "solid",
    "LLMKD-1-partition-per-trial": "solid",
    "LLMKD-3-partition-per-trial": "solid",
    "LLMKD-5-partition-per-trial": "solid",
    "LLMKD-7-partition-per-trial": "solid",
    "LLMKD-k-1": "solid",
    "LLMKD-k-3": "solid",
    "LLMKD-k-5": "solid",
    "LLMKD-k-7": "solid",
    "LLMKD-k-10": "solid",
    "LLMKD-m0-0.25*d": "solid",
    "LLMKD-m0-0.5*d": "solid",
    "LLMKD-m0-1*d": "solid",
    "LLMKD-m0-1": "solid",
    "RS + KD-Tree": "solid",
    "LLM-llama3.1-8b-llm": "solid",
    "LLM-llama3.3-70b-llm": "solid",
    "LLMKD-llama3.1-8b": "solid",
    "LLMKD-llama3.3-70b": "solid",
    "LLMKD-exploitation": "solid",
    "LLMKD-exploration": "solid",
    "LLMKD-ucb1": "solid",
    "LLMKD-uniform_region_sampling": "solid",
    "LLMKD-qwen-30b": "solid",
    "LLM-qwen-30b": "solid",
    "LLM (no-context)": "solid",
    "LLMKD (no-context)": "solid",
}
LABEL = {
    # Our methods
    "RS": "RS",
    "REA": "RE",
    "BORE": "BORE",
    "BOTorch": "GP-EI",
    "CQR": "CQR",
    "TPE": "TPE",
    "Turbo": "TuRBO",
    "RS + KD-Tree": "RS + KD-Tree",
    "LLM": "LLM",
    "LLM-25": "LLM",
    "LLMKD": "HOLLM",
    "LLM (no-context)": "LLM (No Context)",
    "LLMKD (no-context)": "HOLLM (No Context)",
    "LLMKD-alpha-0.2": r"HOLLM ($\alpha_{max} = 0.2$)",
    "LLMKD-alpha-0.5": r"HOLLM ($\alpha_{max} = 0.5$)",
    "LLMKD-alpha-0.7": r"HOLLM ($\alpha_{max} = 0.7$)",
    "LLMKD-1-partition-per-trial": r"HOLLM ($K_t = 1$)",
    "LLMKD-3-partition-per-trial": r"HOLLM ($K_t = 3$)",
    "LLMKD-5-partition-per-trial": r"HOLLM ($K_t = 5$)",
    "LLMKD-7-partition-per-trial": r"HOLLM ($K_t = 7$)",
    "LLMKD-k-1": r"HOLLM ($k = 1$)",
    "LLMKD-k-3": r"HOLLM ($k = 3$)",
    "LLMKD-k-5": r"HOLLM ($k = 5$)",
    "LLMKD-k-7": r"HOLLM ($k = 7$)",
    "LLMKD-k-10": r"HOLLM ($k = 10$)",
    "LLMKD-m0-0.25*d": r"HOLLM ($m0 = 0.25 \cdot d$)",
    "LLMKD-m0-0.5*d": r"HOLLM ($m0 = 0.5 \cdot d$)",
    "LLMKD-m0-1*d": r"HOLLM ($m0 = 1\cdot d$)",
    "LLMKD-m0-1": r"HOLLM ($m0 = 1$)",
    # "LLMKD-alpha-1.0": "HOLLM",
    "LLMKD-alpha-1.0": r"HOLLM ($\alpha_{max} = 1.0$)",  # For alpha comparision
    "LLM-llama3.1-8b-llm": "LLM (Llama-3.1-8B)",
    "LLM-llama3.3-70b-llm": "LLM (Llama-3.3-70B)",
    "LLMKD-llama3.1-8b": "HOLLM (Llama-3.1-8B)",
    "LLMKD-llama3.3-70b": "HOLLM (Llama-3.3-70B)",
    "LLMKD-exploitation": "HOLLM (Exploitation)",
    "LLMKD-exploration": "HOLLM (Exploration)",
    "LLMKD-ucb1": "HOLLM (UCB1)",
    "LLMKD-uniform_region_sampling": "HOLLM (Uniform)",
    "LLMKD-qwen-30b": "HOLLM (Qwen3-30B-A3B)",
    "LLM-qwen-30b": "LLM (Qwen3-30B-A3B)",
}


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
    fig, ax = plt.subplots(figsize=(8, 6))

    # Get methods and sort them alphabetically and move everything that starts with LLM to the end
    methods = sorted(data["mean"].keys(), key=lambda x: (x.startswith("LLM"), x))
    print(methods)
    for i, method in enumerate(methods):
        print(f"Plotting: {method}")
        mean_values = (
            -1 * data["mean"][method]
        )  # Negate the mean values to flip the growth direction in the plot
        std_values = 1.96 * data["std"][method] / data["num_seeds"]  # Standard error
        trials = range(len(mean_values))

        label = LABEL[method]
        color = COLORS[method]
        marker = MARKERS[method]
        linestyle = LSTYLE[method]

        # Mean hypervolume line
        ax.plot(
            trials,
            mean_values,
            label=label,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=2,
            markevery=5,
        )
        ax.fill_between(
            trials,
            np.array(mean_values) + np.array(std_values),
            np.array(mean_values) - np.array(std_values),
            color=color,
            alpha=0.2,
        )

    # Add labels and title with LaTeX formatting
    ax.set_xlabel("Number of evaluations", fontsize=20)
    ax.set_ylabel("Function value", fontsize=20)

    if use_log_scale:
        ax.set_yscale("log")
    ax.set_title(title, fontsize=24)

    if x_lim:
        ax.set_xlim(float(x_lim[0]), float(x_lim[1]))
    if y_lim:
        ax.set_ylim(float(y_lim[0]), float(y_lim[1]))

    ax.tick_params(axis="both", which="major", labelsize=20)

    # Improve grid appearance
    ax.grid(True, alpha=0.4)

    ax.legend(loc="best", fontsize=14)
    # ax.legend(
    #     loc="upper right",
    #     frameon=True,
    #     framealpha=0.95,
    #     edgecolor="k",
    #     fancybox=False,
    #     ncol=1,
    # )

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
    parser.add_argument(
        "--whitelist",
        default="",
        type=str,
        help="Only plot models or methods in this list",
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
        default=False,
        type=bool,
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
    whitelist = args.whitelist.split(",")
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
    print(f"Blacklist:   {blacklist}")
    print(f"Whitelist:   {whitelist}\\n")

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
    print("Before: ", method_dfs.keys())
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
            if not any(entry == method for entry in blacklist):
                filtered_dfs[method] = dfs
            else:
                print(f"Filtered out method: {method} due to blacklist")
        method_dfs = filtered_dfs

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
        i = 0
        for df in dfs:
            # 3. 1. Normalize the data for each dataframe
            if normalization_method == "minmax":
                df = normalize_data(df, min_max_values)
            elif normalization_method == "nb201":
                df = normalize_data_nb201(df, nb201_device_metric)

            if method == "Turbo":
                df = df["F1"].apply(lambda x: -1 * x)
            print("In loop: ", method, " index: ", i, "dg len: ", len(df))
            i += 1
            method_fval.append(np.minimum.accumulate(df[:trials].values).flatten())
        print(method)
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
        "num_seeds": len(method_fvals),
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
