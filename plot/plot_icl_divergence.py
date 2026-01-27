import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast  # To safely parse string representations of lists/dicts
import glob
import argparse
import os
from plot_settings import LINE_STYLE_HV, LABEL_MAP_HV, get_color


# For efficient distance calculations
from scipy.spatial.distance import cdist

DEFAULT_FIGSIZE = (8, 4)  # width, height in inches for all plots
FONT_SIZE = 11
AX_LABELSIZE = 12
TITLE_SIZE = 12
TICK_LABELSIZE = 10
LEGEND_FONTSIZE = 9
USE_LATEX = True

MARKERS = {
    "LLAMBO": "o",
    "LLAMBO-KD": "s",
    "LLAMBO-GD": "^",
    "LLAMBO-KD-GD": "d",
    "MOHOLLM": "*",
    "LLM": "p",
    "MOHOLLM (Gemini 2.0 Flash)": "s",
    "MOHOLLM (Gemini 2.0 Flash) (Context)": "s",  # "MOHOLLM (Context)",
    "MOHOLLM (Gemini 2.0 Flash) (minimal)": "s",  # "MOHOLLM (Context)",
    "mohollm (Gemini 2.0 Flash)": "o",
}

# Use LaTeX for text rendering and apply the unified sizes
plt.rcParams.update(
    {
        "text.usetex": USE_LATEX,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": FONT_SIZE,
        "axes.labelsize": AX_LABELSIZE,
        "axes.titlesize": TITLE_SIZE,
        "legend.fontsize": LEGEND_FONTSIZE,
        "xtick.labelsize": TICK_LABELSIZE,
        "ytick.labelsize": TICK_LABELSIZE,
        "figure.figsize": DEFAULT_FIGSIZE,
    }
)


def parse_string_to_list(s):
    """Safely parses a string like '[{...}, {...}]' into a Python list."""
    try:
        # Handle cases where the data might already be a list
        if isinstance(s, list):
            return s
        data = ast.literal_eval(s)
        if isinstance(data, dict):
            return [data]
        return data
    except (ValueError, SyntaxError):
        return []


def get_vectors_from_dicts(dict_list):
    """Converts a list of configuration dictionaries into a 2D NumPy array."""
    if not dict_list:
        return np.array([]), []

    # Use the keys from the first dictionary as the canonical order
    headers = sorted(dict_list[0].keys())

    vectors = []
    for d in dict_list:
        vectors.append([d.get(h, 0) for h in headers])

    return np.array(vectors), headers


# --- Core Metric Calculation Function ---


def calculate_icl_divergence(row, plot_surrogate_model=False):
    """
    Calculates the ICL Divergence Score for a single row of the DataFrame.
    The score is the average minimum distance from each candidate to the ICL set.
    """
    if plot_surrogate_model:
        icl_configs = parse_string_to_list(row["current_icl_evaluations"])
        candidate_configs = parse_string_to_list(row["llm_surrogate_proposal"])
    else:
        icl_configs = parse_string_to_list(row["current_icl_configs"])
        candidate_configs = parse_string_to_list(row["llm_candidate_proposal"])

    if not icl_configs or not candidate_configs:
        return np.nan

    icl_vectors, icl_headers = get_vectors_from_dicts(icl_configs)
    candidate_vectors, candidate_headers = get_vectors_from_dicts(candidate_configs)

    # Ensure headers are consistent, which they should be in a real run
    if (
        not np.array_equal(icl_headers, candidate_headers)
        and icl_headers
        and candidate_headers
    ):
        print("Warning: Mismatch in decision space headers between ICL and candidates.")
        return np.nan

    if icl_vectors.size == 0 or candidate_vectors.size == 0:
        return np.nan

    # Normalzie the ICL Divergence to unit hypercube
    combined = np.vstack([icl_vectors, candidate_vectors])
    min_vals = np.min(combined, axis=0)
    max_vals = np.max(combined, axis=0)
    range_vals = max_vals - min_vals
    # Avoid division by zero for constant-dimension cases
    range_vals[range_vals == 0] = 1.0
    icl_vectors_norm = (icl_vectors - min_vals) / range_vals
    candidate_vectors_norm = (candidate_vectors - min_vals) / range_vals

    # Calculate all pairwise distances (candidates vs. ICL)
    distance_matrix = cdist(candidate_vectors_norm, icl_vectors_norm, "euclidean")

    # Find the minimum distance for each candidate to the ICL set
    min_distances = np.min(distance_matrix, axis=1)

    # The ICL Divergence Score is the average of these minimum distances
    gravity_score = np.mean(min_distances)

    return gravity_score


# --- Main Plotting Function ---


# def plot_icl_divergence_trajectory(df, ax, experiment_label=""):
#     """
#     Calculates and plots the ICL Divergence Score for each iteration.

#     Args:
#         df (pd.DataFrame): The experiment data.
#         ax (matplotlib.axes.Axes): The matplotlib axis to plot on.
#         experiment_label (str): A label for the plot line (e.g., 'GPT-4').
#     """
#     # 1. Calculate the ICL Divergence Score for each row if not already present
#     if "icl_divergence_score" not in df.columns:
#         print("Calculating ICL Divergence Scores...")
#         df["icl_divergence_score"] = df.apply(calculate_icl_divergence, axis=1)

#     # 2. Plot the trajectory
#     iterations = df.index
#     scores = df["icl_divergence_score"]

#     ax.plot(iterations, scores, marker="o", linestyle="-", label=experiment_label)

#     # 3. Add helpful labels and a title
#     ax.set_title("ICL Divergence Score Trajectory Over Iterations")
#     ax.set_xlabel("Iteration Number")
#     ax.set_ylabel("ICL Divergence Score (Avg. Min Distance)")
#     ax.grid(True, which="both", linestyle="--", linewidth=0.5)
#     ax.set_xticks(iterations)  # Ensure all iteration numbers are shown as ticks


def compute_icl_divergence_trajectory(df, plot_surrogate_model):
    """
    Computes the per-iteration ICL Divergence Score for a run and returns the
    list of scores (one per iteration). This no longer computes or returns any
    cumulative-evaluation x-coordinates; alignment is done later by iteration index.
    """
    if "icl_divergence_score" not in df.columns:
        df["icl_divergence_score"] = df.apply(
            calculate_icl_divergence, axis=1, plot_surrogate_model=plot_surrogate_model
        )

    # Return the per-iteration scores (may contain NaNs which are handled during alignment)
    return df["icl_divergence_score"].tolist()


def create_icl_divergence_over_time_plot(data, title, path, filename):
    """
    Create publication-ready ICL Divergence over time plots.

    Parameters:
        data (dict): Dictionary containing 'mean' and 'std' arrays keyed by method and
                     a common 'x_axis' numpy array of iteration indices.
        title (str): Title of the plot
        path (str): Directory path to save the plot
        filename (str): Filename for saving the plot
    """
    # Create figure with appropriate size for single-column journals
    fig, ax = plt.subplots(figsize=(5, 3.5))

    # Get methods and sort them alphabetically
    methods = sorted(data["mean"].keys())

    # The common x-axis for all methods
    x_axis = data["x_axis"]

    for i, method in enumerate(methods):
        print(f"Plotting: {method}")
        mean_values = data["mean"][method]
        std_values = data["std"][method]

        plot_len = min(len(x_axis), len(mean_values))

        # Calculate standard error
        std_error = 1.96 * std_values / np.sqrt(len(mean_values))

        # Mean hypervolume line
        ax.plot(
            x_axis[:plot_len],
            mean_values[:plot_len],
            label=LABEL_MAP_HV.get(method, method),
            color=get_color(method, i),
            marker=MARKERS.get(method, "o"),
            linestyle=LINE_STYLE_HV[i % len(LINE_STYLE_HV)],
            linewidth=1.5,
        )

        # Confidence interval using standard error
        ax.fill_between(
            x_axis[:plot_len],
            mean_values[:plot_len] - std_error,
            mean_values[:plot_len] + std_error,
            color=get_color(method, i),
            alpha=0.2,
            linewidth=0,
        )

    # Add labels and title with LaTeX formatting
    ax.set_xlabel("Function Evaluations")
    ax.set_ylabel("ICL Divergence")
    ax.set_title(title)

    # Improve grid appearance
    ax.grid(True, linestyle="--", alpha=0.2, linewidth=0.4)

    # Format tick labels
    ncol = 2
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.3),
        ncol=ncol,
        frameon=True,
        framealpha=0.95,
        edgecolor="k",
        fancybox=False,
    )

    plt.tight_layout()

    for file_type in ["svg", "pdf", "png"]:
        dir_path = f"{path}/{file_type}/{filename}.{file_type}"
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
    parser.add_argument(
        "--plot_surrogate_model",
        default=False,
        type=bool,
        help="If true plot the surrogate model ICL Divergence instead.",
    )

    args = parser.parse_args()

    data_path = args.data_path
    benchmark = args.benchmark
    plot_surrogate_model = args.plot_surrogate_model
    title = args.title
    trials = args.trials
    filter = args.filter
    filename = args.filename
    blacklist = args.blacklist.split(",")
    whitelist = args.whitelist.split(",")
    output_path = f"./plots/icl_divergence/{benchmark}"

    print("#" * 80)
    print("Plotting ICL Divergence")
    print(f"Benchmark:   {benchmark}")
    print(f"Title:       {title}")
    print(f"Data Path:   {data_path}")
    print(f"Filename:    {filename}")
    print(f"Output Path: {output_path}")
    print(f"Filter:      {filter}")
    print(f"Trials:      {trials}")
    print(f"Blacklist:   {blacklist}")
    print(f"Whitelist:   {whitelist}\n")

    file_names = glob.glob(f"{data_path}/**/*.csv", recursive=True)
    icl_llm_proposal_trajectory_files = [
        file for file in file_names if "icl_llm_proposal_trajectory" in file.split("/")
    ]

    method_files = group_data_by_method(icl_llm_proposal_trajectory_files, filter)

    method_dfs = {}
    for method, files in method_files.items():
        dfs = []
        for file in files:
            try:
                df = pd.read_csv(file)
                if trials is not None:
                    df = df[:trials]
                dfs.append(df)
            except pd.errors.EmptyDataError:
                # Skip empty files
                continue
            except Exception as e:
                print(f"Error reading {file}: {e}")
        method_dfs[method] = dfs

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

    icl_divergence_trajectory_raw = {}

    max_iters = 0
    for method, dfs in method_dfs.items():
        trajectories = []
        for df in dfs:
            y_vals = compute_icl_divergence_trajectory(
                df, plot_surrogate_model=plot_surrogate_model
            )
            # Only add if data is valid and non-empty
            if y_vals and any([not np.isnan(y) for y in y_vals]):
                trajectories.append(y_vals)
                if len(y_vals) > max_iters:
                    max_iters = len(y_vals)
        icl_divergence_trajectory_raw[method] = trajectories

    # Align by iteration index (0..max_iters-1)
    common_x_axis = np.arange(max_iters)
    mean_icl_divergence = {}
    std_icl_divergence = {}

    for method, trajectories in icl_divergence_trajectory_raw.items():
        aligned_trajectories = []
        for y_vals in trajectories:
            # Create a Series indexed by iteration number and forward-fill missing values
            s = pd.Series(y_vals, index=np.arange(len(y_vals)))
            s_reindexed = s.reindex(common_x_axis, method="ffill")
            # Back-fill any initial NaNs
            s_reindexed = s_reindexed.bfill()
            if not s_reindexed.empty:
                aligned_trajectories.append(s_reindexed.values)

        if aligned_trajectories:
            mean_icl_divergence[method] = np.mean(aligned_trajectories, axis=0)
            std_icl_divergence[method] = np.std(aligned_trajectories, axis=0)

    data = {
        "mean": mean_icl_divergence,
        "std": std_icl_divergence,
        "x_axis": common_x_axis,  # Pass the common x-axis to the plot function
    }
    create_icl_divergence_over_time_plot(data, title, output_path, filename)


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


# --- Main Execution Block ---
if __name__ == "__main__":
    main()
