import argparse
import json
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from normalize_data_nb201 import normalize_data_nb201
from pymoo.indicators.hv import HV
from plot_settings import get_color, LINE_STYLE_HV, LABEL_MAP_HV
from scipy.stats import friedmanchisquare
from aeon.visualisation import plot_critical_difference
from scipy.stats import wilcoxon

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
    parser = argparse.ArgumentParser(description="Plot critical difference over time.")
    parser.add_argument(
        "--benchmarks",
        type=str,
        required=True,
        help="Benchmark name (seperated by comma)",
    )
    parser.add_argument("--title", type=str, required=True, help="Plot title")
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
        "--whitelist",
        default="",
        type=str,
        help="Only plot models or methods in this list",
    )

    args = parser.parse_args()

    benchmarks = args.benchmarks.split(",")
    title = args.title
    filter = args.filter
    trials = args.trials
    filename = args.filename
    normalization_method = args.normalization_method
    columns = args.columns.split(",") if args.columns else None
    blacklist = args.blacklist.split(",")
    whitelist = args.whitelist.split(",")
    output_path = "./plots/critical_difference"

    # NB201 specific
    nb201_device_metric = args.nb201_device_metric

    print("#" * 80)
    print("Plotting hypervolume over time")
    print(f"Benchmarks:   {benchmarks}")
    print(f"Title:       {title}")
    print(f"Filename:    {filename}")
    print(f"Output Path: {output_path}")
    print(f"Filter:      {filter}")
    print(f"Columns:     {columns}")
    print(f"Trials:      {trials}")
    print(f"Blacklist:   {blacklist}")
    print(f"Whitelist:   {whitelist}\n")

    all_data = {}  # All data accross all benchmarks
    for benchmark in benchmarks:
        print(f"Computing benchmark: {benchmark} ")
        # Gather all CSV files from folder
        file_names = glob.glob(f"./results/{benchmark}/**/*.csv", recursive=True)
        observed_fvals_files = [
            file for file in file_names if "observed_fvals" in file.split("/")
        ]

        # Group the files names into a dict of {method_name: [list of file names]}
        method_files = group_data_by_method(observed_fvals_files, filter)
        # 1. Convert to pandas DataFrames

        # Load all columns, then drop 'configs' if present
        method_dfs = {
            method: [
                df.drop(columns=["configs"]) if "configs" in df.columns else df
                for df in [pd.read_csv(file)[:trials] for file in files]
            ]
            for method, files in method_files.items()
        }

        # Dynamically determine objective columns
        first_method = next(iter(method_dfs))
        first_df = method_dfs[first_method][0]
        columns = [col for col in first_df.columns if col != "configs"]
        print(columns)
        print(f"Number of objectives: {len(columns)} for benchmark {benchmark}")
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

        # Calculate the max and min values for each columns across all dataframes
        # 2. extract the min and max values for each column
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

        print("Calculated min_max_values across all datasets:", min_max_values)
        # 3. Calculate the hypervolume over time for each method and each dataframe
        hypervolume_over_time = {}
        for method, dfs in method_dfs.items():
            method = LABEL_MAP_HV.get(method, method)
            method_hv = []
            for df in dfs:
                # 3. 1. Normalize the data for each dataframe
                if normalization_method == "minmax":
                    df = normalize_data(df, min_max_values)
                elif normalization_method == "nb201":
                    df = normalize_data_nb201(df, nb201_device_metric)
                # Use all columns as objectives
                method_hv.append(
                    convert_data_to_hv_over_time(
                        df[columns][:trials], reference_point=[1.0] * len(columns)
                    )
                )
            print(f"Calculating HV over time for: {method}")
            hypervolume_over_time[method] = np.array(method_hv)

        # 4. Compute the mean and standard deviation of the hypervolume over time for each method
        mean_score = {
            method: np.mean(hv, axis=0)[-1]  # We only care about the final value
            for method, hv in hypervolume_over_time.items()
        }
        # std_hypervolume_over_time = {
        #     method: np.std(hv, axis=0) for method, hv in hypervolume_over_time.items()
        # }

        # data = {
        #     "mean": mean_hypervolume_over_time,
        #     "std": std_hypervolume_over_time,
        # }

        all_data[benchmark] = mean_score

    all_data = pd.DataFrame(all_data).T  # Transpose this so each row is one benchmark
    # all_data.to_csv("test.csv")

    # create_critical_difference_diagram(all_data, output_path, filename)
    compare_mohollm_vs_llm(all_data)


def create_critical_difference_diagram(all_data, path, filename):
    # We need to maximize here as we want to maximize the hypervolume
    ranking_matrix = all_data.rank(axis=1, method="min", ascending=False)

    friedman_stat, p_value = friedmanchisquare(*ranking_matrix.T.values)
    print(f"Friedman test statistic: {friedman_stat}, p-value = {p_value}")
    scores = all_data.values
    classifiers = all_data.columns.tolist()
    plt.figure(figsize=(16, 12))
    plot_critical_difference(
        scores,
        classifiers,
        lower_better=False,
        test="nemenyi",  # or nemenyi or wilcoxon
        alpha=0.01,
        correction="bonferroni",  # or bonferroni or holm or none
    )
    # plot_critical_difference(
    #     scores,
    #     classifiers,
    #     lower_better=False,
    #     test="wilcoxon",  # or nemenyi or wilcoxon
    #     alpha=0.01,
    #     correction="holm",  # or bonferroni or holm or none
    # )
    ax = plt.gca()
    # Adjust font size and rotation of x-axis labels
    for label in ax.get_xticklabels():
        label.set_fontsize(14)
        label.set_rotation(45)
        label.set_horizontalalignment("right")

    # Increase padding between labels and axis
    ax.tick_params(axis="x", which="major", pad=20)

    # Adjust margins to provide more space for labels
    plt.subplots_adjust(bottom=0.35)

    # Optionally adjust y-axis label font size
    ax.tick_params(axis="y", labelsize=12)

    plt.tight_layout()

    for file_type in ["svg", "pdf", "png"]:
        dir_path = f"{path}/{file_type}/{filename}.{file_type}"
        os.makedirs(os.path.dirname(dir_path), exist_ok=True)
        plt.savefig(dir_path, dpi=300, bbox_inches="tight")

    plt.close()


def compare_mohollm_vs_llm(all_data):
    # Make sure the column names match exactly those in your DataFrame
    mohol = all_data["MOHOLLM"].values
    llm = all_data["LLM"].values

    # Paired Wilcoxon signed-rank test across benchmarks
    stat, p_value = wilcoxon(mohol, llm, zero_method="wilcox", alternative="two-sided")

    # Compute effect size (Vargha–Delaney A12)
    def a12(x, y):
        n1, n2 = len(x), len(y)
        wins, ties = 0, 0
        for xi in x:
            for yj in y:
                if xi > yj:
                    wins += 1
                elif xi == yj:
                    ties += 1
        return (wins + 0.5 * ties) / (n1 * n2)

    effect_size = a12(mohol, llm)

    print("Wilcoxon MOHOLLM vs LLM")
    print(f"Statistic: {stat}, p-value: {p_value:.3e}")
    print(f"Vargha–Delaney A12: {effect_size:.3f}")

    return stat, p_value, effect_size


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
        # FIXME: Replace this to ensure we have a consistent naming in the plot.
        if method == "MOHOLLM (Gemini 2.0 Flash) (Context)":
            method = "MOHOLLM (Gemini 2.0 Flash)"
        files = []
        for file in observed_fvals_files:
            # Includes file is the method name matches the file path fully (hence the split)
            if method in file.split("/") and (filter in file if filter else True):
                files.append(file)
        method_files[method] = files
    return method_files


if __name__ == "__main__":
    main()
