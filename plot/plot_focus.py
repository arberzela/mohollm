import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
import glob
import argparse
import os
from scipy.stats import entropy
from plot_settings import get_color, LABEL_MAP_HV
from sklearn.neighbors import KernelDensity

# --- Plotting Setup ---
# Use LaTeX for text rendering if available
DEFAULT_FIGSIZE = (8, 4)  # width, height in inches for all plots
FONT_SIZE = 11
AX_LABELSIZE = 12
TITLE_SIZE = 12
TICK_LABELSIZE = 10
LEGEND_FONTSIZE = 9
USE_LATEX = True

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
    try:
        if isinstance(s, list):
            return s
        data = ast.literal_eval(s)
        return [data] if isinstance(data, dict) else data
    except (ValueError, SyntaxError):
        return []


def calculate_focus_score(series, categorical):
    """Calculates the Distributional Focus Score (0-1) for a pandas Series."""
    if series.empty:
        return np.nan
    if categorical:
        # Use Normalized Entropy: Focus = 1 - Normalized Entropy
        counts = series.value_counts()
        if len(counts) <= 1:
            return 1.0  # Max focus if only one category is present

        probs = counts / counts.sum()
        ent = entropy(probs, base=2)
        max_ent = np.log2(len(counts))

        normalized_entropy = ent / max_ent if max_ent > 0 else 0
        return 1.0 - normalized_entropy

    else:
        # TODO: Figure out if this makes sense or we should use something different
        # Use Coefficient of Variation: Focus = exp(-CV)
        # Fit KDE on series -> Distribution
        # Sample from this distribution
        # Entropy of the distribution -> https://seaborn.pydata.org/generated/seaborn.histplot.html
        # https://scikit-learn.org/stable/modules/density.html
        # std = series.std()
        # mean = series.mean()

        # if np.abs(mean) < 1e-9:
        #     return (
        #         1.0 if std < 1e-9 else 0.0
        #     )  # Max focus if all values are zero, else max diversity

        # cv = std / np.abs(mean)  # Use abs mean for robustness
        # return np.exp(-cv)
        return calculate_focus_score_continuous_kde(series)


def calculate_focus_score_continuous_kde(series, n_samples=1000, n_bins=50):
    """
    Calculates the Distributional Focus Score (0-1) for a continuous pandas Series
    using a KDE and entropy-based method.
    """
    # --- 1. Handle Edge Cases ---
    if series.empty or series.isnull().all():
        return np.nan

    # If there is no variance, the focus is maximal.
    if series.std() < 1e-9:
        return 1.0

    # --- 2. Fit KDE on the data ---
    # Scikit-learn's KDE expects a 2D array of shape (n_samples, n_features)
    data = series.to_numpy().reshape(-1, 1)

    # Bandwidth selection is crucial. 'scott' or 'silverman' are good starting points.
    # For maximum rigor, one could use GridSearchCV, but this is a good practical choice.
    kde = KernelDensity(kernel="gaussian", bandwidth="scott").fit(data)

    # --- 3. Sample from the KDE ---
    samples = kde.sample(n_samples=n_samples)

    # --- 4. Discretize samples into bins to create a PMF ---
    counts, bin_edges = np.histogram(samples, bins=n_bins, density=False)

    # If all samples fall in one bin, focus is maximal.
    if len(np.unique(counts)) <= 1 and np.sum(counts) > 0:
        return 1.0

    probs = counts / n_samples

    # --- 5. Calculate Shannon Entropy ---
    # We use base=n_bins to directly get the normalized entropy.
    # H_norm = entropy(probs, base=n_bins) is a more direct way to normalize.
    ent = entropy(probs, base=2)
    max_ent = np.log2(n_bins)

    if max_ent <= 0:
        return 1.0  # Handle case with 1 bin

    normalized_entropy = ent / max_ent

    # --- 6. Calculate the final Focus Score ---
    focus_score = 1.0 - normalized_entropy

    return focus_score


def compute_focus_trajectories(df, categorical):
    """
    Computes focus score trajectories for every feature in the LLM proposals.
    Returns a dictionary where keys are feature names and values are lists of (x, y) coordinates.
    """
    trajectories = {}

    # Data already contains correctly formatted steps; use row index as x-axis

    # Iterate over each row (optimization step)
    for i, row in df.iterrows():
        proposals_list = parse_string_to_list(row["llm_candidate_proposal"])
        if not proposals_list:
            continue

        # Convert the list of proposal dicts into a DataFrame
        proposals_df = pd.DataFrame(proposals_list)

        # The x-coordinate is the row index (step number)
        x_coord = i
        # Calculate focus score for each feature (column)
        for feature in proposals_df.columns:
            series = proposals_df[feature].dropna()
            score = calculate_focus_score(series, categorical)

            if not np.isnan(score):
                # Initialize list for the feature if it's the first time seeing it
                trajectories.setdefault(feature, []).append((x_coord, score))

    return trajectories


# --- Plotting Function ---


def create_feature_focus_plot(data, title, path, filename):
    """Creates a line plot showing the focus score evolution for each feature."""
    # Use a larger figure to accommodate the legend
    fig, ax = plt.subplots(figsize=(7, 4.5))

    features = sorted(data["mean"].keys())
    x_axis = data["x_axis"]

    # Use a colormap for distinct feature colors
    colors = plt.cm.viridis(np.linspace(0, 1, len(features)))

    for i, feature in enumerate(features):
        mean_values = data["mean"][feature]
        std_values = data["std"][feature]
        plot_len = min(len(x_axis), len(mean_values))

        std_error = 1.96 * std_values / np.sqrt(len(mean_values))

        ax.plot(
            x_axis[:plot_len],
            mean_values[:plot_len],
            label=feature.replace("_", " "),
            color=colors[i],
            linewidth=1.5,
        )

        ax.fill_between(
            x_axis[:plot_len],
            mean_values[:plot_len] - std_error,
            mean_values[:plot_len] + std_error,
            color=colors[i],
            alpha=0.15,
        )

    ax.set_xlabel("Trials")
    ax.set_ylabel("Distributional Focus Score (1 = High Focus)")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.6, linewidth=0.4)
    # ax.set_ylim(-0.05, 1.05)  # Y-axis from 0 to 1

    # Place legend below the plot (centered). Choose number of columns to keep legend compact.
    ncol = min(len(features), 6) if len(features) > 0 else 1
    ax.legend(
        title="Features",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=ncol,
        frameon=False,
    )

    # Leave room at the bottom for the legend
    plt.tight_layout(rect=[0, 0.12, 1, 1])

    for file_type in ["svg", "pdf", "png"]:
        dir_path = f"{path}/{file_type}/{filename}.{file_type}"
        os.makedirs(os.path.dirname(dir_path), exist_ok=True)
        plt.savefig(dir_path, dpi=300, bbox_inches="tight")
    plt.close()


def create_aggregated_focus_plot(data, title, path, filename):
    """Creates a line plot comparing the aggregated focus score for multiple methods."""
    fig, ax = plt.subplots(figsize=(6, 5.5))
    methods = sorted(data["mean"].keys())

    for i, method in enumerate(methods):
        mean_values = data["mean"][method]
        std_values = data["std"][method]
        # Compute local x-axis for this method from the length of the mean values
        plot_len = len(mean_values)
        std_error = 1.96 * std_values / np.sqrt(len(mean_values))
        x_axis_method = np.arange(plot_len)

        ax.plot(
            x_axis_method,
            mean_values,
            label=LABEL_MAP_HV.get(method, method),
            color=get_color(method, i),
            linestyle="-",
            linewidth=2.5,
        )
        ax.fill_between(
            x_axis_method,
            mean_values - std_error,
            mean_values + std_error,
            color=get_color(method, i),
            alpha=0.2,
            linewidth=0,
        )

    ax.set_xlabel("Trials")
    ax.set_ylabel("Aggregated Focus Score")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.6)
    # ax.set_ylim(0.5, 1.05)
    # Place legend below the plot (centered) and keep it compact
    ncol_methods = min(len(methods), 2) if len(methods) > 0 else 1
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        ncol=ncol_methods,
        frameon=True,
        framealpha=0.95,
        edgecolor="k",
        fancybox=False,
    )
    # Leave room at the bottom for the legend
    plt.tight_layout(rect=[0, 0.12, 1, 1])

    # Save to a new 'aggregated' subfolder
    for file_type in ["svg", "pdf", "png"]:
        dir_path = f"{path}/{file_type}/{filename}.{file_type}"
        os.makedirs(os.path.dirname(dir_path), exist_ok=True)
        plt.savefig(dir_path, dpi=300, bbox_inches="tight")
    plt.close()


# --- Reusable Main Structure ---


def extract_method_name(files):
    return sorted(list(set([file.split(os.sep)[-3] for file in files])))


def group_data_by_method(files_list, file_filter):
    method_names = extract_method_name(files_list)
    return {
        method: [
            file
            for file in files_list
            if method in file.split(os.sep)
            and (file_filter in file if file_filter else True)
        ]
        for method in method_names
    }


def main():
    parser = argparse.ArgumentParser(
        description="Plot Distributional Focus Score for each feature over time."
    )
    parser.add_argument(
        "--title", type=str, required=True, help="Defines the title in the plot"
    )
    parser.add_argument("--benchmark", type=str, required=True, help="Benchmark name")
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to experiment data folder"
    )
    parser.add_argument(
        "--filter", type=str, help="Filter to only include files containing this string"
    )
    parser.add_argument(
        "--filename", type=str, help="The filename how the plot should be stored"
    )
    parser.add_argument(
        "--whitelist",
        type=str,
        default=None,
        help="Comma-separated list of methods to include (only these will be plotted)",
    )
    parser.add_argument(
        "--trials", type=int, required=True, help="Number of trials (rows) to consider"
    )
    parser.add_argument(
        "--categorical",
        default=False,
        type=bool,
        help="Whether the values are categorical or continuous",
    )
    args = parser.parse_args()
    output_path_base = f"./plots/feature_focus/{args.benchmark}"
    file_names = glob.glob(f"{args.data_path}/**/*.csv", recursive=True)
    trajectory_files = [
        f for f in file_names if "icl_llm_proposal_trajectory" in f.split(os.sep)
    ]
    method_files = group_data_by_method(trajectory_files, args.filter)

    # If whitelist provided, filter method_files to only include whitelisted methods
    if args.whitelist:
        whitelist = [w.strip() for w in args.whitelist.split(",") if w.strip()]
        if whitelist:
            filtered = {}
            for method, files in method_files.items():
                if any(w == method for w in whitelist):
                    filtered[method] = files
            method_files = filtered

    # --- NEW: Dictionary to store aggregated data for the final plot ---
    aggregated_focus_data_all_methods = {"mean": {}, "std": {}}

    for method, files in method_files.items():
        print(f"--- Processing Method: {method} ---")
        # This dictionary will store raw trajectory data for each feature across all trials
        # e.g., {'learning_rate': [[(x1,y1), (x2,y2)], [(x1,y1), ...]], 'optimizer': ...}
        all_trials_trajectories = {}
        max_evals_method = 0

        for trial_file in files:
            df = pd.read_csv(trial_file)[: args.trials]
            if df.empty:
                continue
            single_trial_trajectories = compute_focus_trajectories(df, args.categorical)
            for feature, trajectory in single_trial_trajectories.items():
                if trajectory:
                    # Append the whole list of (x,y) tuples for this trial
                    all_trials_trajectories.setdefault(feature, []).append(trajectory)
                    last_x = trajectory[-1][0]
                    if last_x > max_evals_method:
                        max_evals_method = last_x

        if not all_trials_trajectories:
            print(f"No valid trajectory data for {method}. Skipping.")
            continue

        # (no global x-axis aggregation required anymore)

        common_x_axis = np.arange(max_evals_method + 1)
        mean_trajectories = {}
        std_trajectories = {}

        # This will hold the aligned data for all features for this method
        aligned_feature_data_for_aggregation = []

        for feature, trials_data in all_trials_trajectories.items():
            aligned_runs = []
            for run_data in trials_data:
                x_vals, y_vals = zip(*run_data)
                s = pd.Series(y_vals, index=x_vals)
                s_reindexed = s.reindex(common_x_axis, method="ffill").bfill()
                if not s_reindexed.empty:
                    aligned_runs.append(s_reindexed.values)

            if aligned_runs:
                mean_trajectories[feature] = np.nanmean(aligned_runs, axis=0)
                std_trajectories[feature] = np.nanstd(aligned_runs, axis=0)
                # Add the mean trajectory for this feature to our list for aggregation
                aligned_feature_data_for_aggregation.append(mean_trajectories[feature])

        # --- Individual Feature Plotting (Existing) ---
        if mean_trajectories:
            individual_data = {
                "mean": mean_trajectories,
                "std": std_trajectories,
                "x_axis": common_x_axis,
            }
            plot_title_individual = f"{args.title} ({LABEL_MAP_HV.get(method, method)})"
            filename_individual = f"{method}/{args.filename}_individual_features"
            create_feature_focus_plot(
                individual_data,
                plot_title_individual,
                output_path_base,
                filename_individual,
            )
            print(f"Individual feature plot saved for method {method}.")

        # --- NEW: Aggregation Step ---
        if aligned_feature_data_for_aggregation:
            # Calculate the mean and std *across features* for each time step
            aggregated_mean = np.nanmean(aligned_feature_data_for_aggregation, axis=0)
            aggregated_std = np.nanstd(aligned_feature_data_for_aggregation, axis=0)

            # Store it for the final combined plot
            aggregated_focus_data_all_methods["mean"][method] = aggregated_mean
            aggregated_focus_data_all_methods["std"][method] = aggregated_std

    # --- NEW: Final Aggregated Plotting ---
    if aggregated_focus_data_all_methods["mean"]:
        plot_title_aggregated = f"{args.title}"
        filename_aggregated = f"{args.filename}_aggregated_focus"
        create_aggregated_focus_plot(
            aggregated_focus_data_all_methods,
            plot_title_aggregated,
            "./plots/feature_focus/aggregated/",
            filename_aggregated,
        )
        print("Final aggregated focus plot saved.")


if __name__ == "__main__":
    main()
