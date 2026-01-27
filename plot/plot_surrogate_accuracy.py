import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast  # Crucial for safely reading cached lists from CSV
import glob
import argparse
import os
from plot_settings import COLORS_SURROGATE_ACC, LABEL_MAP_HV, get_color, MARKER_MAP
from mohollm.benchmarks.penicillin import PenicillinBenchmark
from mohollm.benchmarks.car_side_impact import CarSideImpactBenchmark
from mohollm.benchmarks.vehicle_safety import VehicleSafetyBenchmark
import json
import time  # To show performance benefits
from multiprocessing import Pool, cpu_count  # Import for parallel processing
from sklearn.metrics import r2_score  # Import for R2 calculation

# --- Helper Functions (No changes needed here) ---
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 16,
        "axes.labelsize": 16,
        "axes.titlesize": 20,
        "legend.fontsize": 20,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
    }
)
ALPHA = 0.3
# The number of data points to accumulate before starting to calculate the learning curve
LEARNING_CURVE_MIN_POINTS = 15
# Set the quantile for zooming in on the plots.
# 1.0 means use all data (no zoom).
AXIS_QUANTILE = 0.60


def parse_string_to_list(s):
    """
    Safely parses a string that looks like a Python literal into a list.
    NOTE: Using .replace("'", '"') is fast but can break if strings contain apostrophes.
    The robust filtering in the plotting function handles potential errors.
    """
    if not isinstance(s, str):
        return []
    try:
        return json.loads(s.replace("'", '"'))
    except (json.JSONDecodeError, TypeError):
        try:
            return ast.literal_eval(s)
        except (ValueError, SyntaxError):
            return []


def get_data(row, benchmark, n_obj):
    """
    Extracts data and computes ground truth. This is the slow function we want to avoid.
    """
    expected_len = n_obj  # TODO: This only works for 3 obj.
    candidate_configs = parse_string_to_list(row.llm_candidate_proposal)
    llm_surrogate_proposal = parse_string_to_list(row.llm_surrogate_proposal)
    if (
        not candidate_configs
        or not llm_surrogate_proposal
        or any([len(entry.keys()) != expected_len for entry in llm_surrogate_proposal])
    ):
        print(f"Returning None for: {llm_surrogate_proposal}")
        return None

    ground_truth = [benchmark.evaluate_point(c)[1] for c in candidate_configs]

    return {
        "candidate": [list(c.values()) for c in candidate_configs],
        "llm_proposals": [list(p.values()) for p in llm_surrogate_proposal],
        "ground_truth": [list(gt.values()) for gt in ground_truth],
    }


def main():
    # --- (Your argparse setup remains the same) ---
    parser = argparse.ArgumentParser(description="Plot surrogate model accuracy.")
    # ... (all your arguments here) ...
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
        "--n",
        type=int,
        default=None,
        help="Maximum number of random samples to use per method for plotting and metrics. Uses all data if not set.",
    )
    args = parser.parse_args()

    # --- (Your args processing remains the same) ---
    data_path = args.data_path
    benchmark_name = args.benchmark
    title = args.title
    trials = args.trials
    filter_str = args.filter
    blacklist = args.blacklist.split(",") if args.blacklist else []
    whitelist = args.whitelist.split(",") if args.whitelist else []
    output_path = f"./plots/surrogate_accuracy/{benchmark_name}"
    max_samples = args.n

    start_time = time.time()

    processed_data_cache_dir = os.path.join(output_path, "_cache")
    learning_curve_cache_file = os.path.join(output_path, "_cache_learning_curve.json")
    os.makedirs(processed_data_cache_dir, exist_ok=True)

    file_names = glob.glob(f"{data_path}/**/*.csv", recursive=True)
    icl_files = [f for f in file_names if "icl_llm_proposal_trajectory" in f]

    method_files = group_data_by_method(icl_files, filter_str)

    # Apply Whitelist/Blacklist
    if whitelist:
        method_files = {m: f for m, f in method_files.items() if m in whitelist}
    elif blacklist:
        method_files = {
            m: f for m, f in method_files.items() if not any(b in m for b in blacklist)
        }

    method_dataframes = {}

    if benchmark_name == "Penicillin":
        benchmark = PenicillinBenchmark(model_name="empty", seed=0)
        n_obj = 3
    elif benchmark_name == "VehicleSafety":
        benchmark = VehicleSafetyBenchmark(model_name="empty", seed=0)
        n_obj = 3
    elif benchmark_name == "CarSideImpact":
        benchmark = CarSideImpactBenchmark(model_name="empty", seed=0)
        n_obj = 4

    for method, files in method_files.items():
        cache_file_path = os.path.join(
            processed_data_cache_dir, f"{method}_{benchmark_name}_processed_data.csv"
        )
        df_full = None
        if os.path.exists(cache_file_path):
            print(f"Loading cached data for method: {method}")
            df_full = pd.read_csv(cache_file_path)
            for col in ["candidate", "llm_prediction", "ground_truth"]:
                df_full[col] = df_full[col].apply(ast.literal_eval)
        else:
            print(f"No cache found. Processing raw data for method: {method}...")
            all_candidates, all_predictions, all_ground_truths = [], [], []
            for file in files:
                df_raw = pd.read_csv(file, nrows=trials)
                for row in df_raw.itertuples(index=False):
                    try:
                        row_data = get_data(row, benchmark=benchmark, n_obj=n_obj)
                        if row_data:
                            all_candidates.extend(row_data["candidate"])
                            all_predictions.extend(row_data["llm_proposals"])
                            all_ground_truths.extend(row_data["ground_truth"])
                    except Exception as e:
                        print(
                            f"WARNING: Skipping a row for method '{method}' due to an error: {e}"
                        )
                        continue
            df_full = pd.DataFrame(
                {
                    "candidate": all_candidates,
                    "llm_prediction": all_predictions,
                    "ground_truth": all_ground_truths,
                }
            )
            df_full.to_csv(cache_file_path, index=False)
            print(f" -> Saved processed data to cache: {cache_file_path}")
        if max_samples is not None and len(df_full) > max_samples:
            method_dataframes[method] = df_full.sample(n=max_samples, random_state=42)
        else:
            method_dataframes[method] = df_full

    # --- START OF MODIFICATION: Add caching for Learning Curve Data ---
    learning_curve_data = {}
    if os.path.exists(learning_curve_cache_file):
        print(f"\nLoading cached learning curve data from: {learning_curve_cache_file}")
        with open(learning_curve_cache_file, "r") as f:
            learning_curve_data = json.load(f)
    else:
        print("\nNo learning curve cache found. Processing raw data...")
        num_workers = cpu_count() - 1 or 1
        tasks = [
            (method, files, trials, len(["F1", "F2", "F3"]))
            for method, files in method_files.items()
        ]
        with Pool(processes=num_workers) as pool:
            results = pool.map(process_learning_curve_data, tasks)
        learning_curve_data = {method: data for method, data in results}
        with open(learning_curve_cache_file, "w") as f:
            json.dump(learning_curve_data, f)
        print(f" -> Saved learning curve data to cache: {learning_curve_cache_file}")

    print(learning_curve_data)

    # --- END OF MODIFICATION ---

    print(f"\nData loading and processing took {time.time() - start_time:.2f} seconds.")

    # --- Plotting and Metrics Calculation ---

    objectives = [f"F{i + 1}" for i in range(n_obj)]

    # Define separate output directories for each plot type
    scatter_plot_dir = f"{output_path}/scatter_plots"
    qq_plot_dir = f"{output_path}/qq_plots"
    residual_plot_dir = f"{output_path}/residual_plots"
    metrics_dir = f"{output_path}/metrics_tables"
    learning_curve_plot_dir = f"{output_path}/learning_curves"
    aggregated_plot_dir = f"{output_path}/aggregated_plots"

    # Create all directories
    os.makedirs(scatter_plot_dir, exist_ok=True)
    os.makedirs(qq_plot_dir, exist_ok=True)
    os.makedirs(residual_plot_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(learning_curve_plot_dir, exist_ok=True)
    os.makedirs(aggregated_plot_dir, exist_ok=True)

    # 1. Create the main scatter plots and metrics table
    print("\n--- Generating Scatter Plots and Metrics ---")
    metrics_table = create_surrogate_scatter_plots(
        method_dataframes, objectives, scatter_plot_dir, title
    )

    # 2. Save the metrics tables
    save_metrics_markdown(metrics_table, metrics_dir)
    save_averaged_metrics_markdown(metrics_table, metrics_dir)

    # 3. Create the new diagnostic plots
    create_qq_plots(method_dataframes, objectives, qq_plot_dir, title)
    create_qq_plots_vertical(method_dataframes, objectives, qq_plot_dir, title)

    print("\nAnalysis complete. All plots and metrics have been saved.")


# --- (create_surrogate_scatter_plots remains the same, using the robust version) ---
def create_surrogate_scatter_plots(method_dataframes, objectives, plot_dir, title):
    from scipy.stats import kendalltau, pearsonr, spearmanr
    from sklearn.metrics import r2_score
    import matplotlib

    matplotlib.use("Agg")
    metrics_table = []

    for i, obj in enumerate(objectives):
        fig, ax = plt.subplots(figsize=(6, 4.5))
        plot_handles, plot_labels = [], []

        all_true_vals = []
        all_pred_vals = []

        for idx, (method, df) in enumerate(method_dataframes.items()):
            # ... (filtering and data extraction logic is the same) ...
            expected_len = len(objectives)
            df_filtered = df[
                (
                    df["llm_prediction"].apply(
                        lambda x: isinstance(x, list) and len(x) == expected_len
                    )
                )
                & (
                    df["ground_truth"].apply(
                        lambda x: isinstance(x, list) and len(x) == expected_len
                    )
                )
            ].copy()
            if df_filtered.empty:
                continue

            y_true = np.array(df_filtered["ground_truth"].tolist())
            y_pred = np.array(df_filtered["llm_prediction"].tolist())
            true_vals = y_true[:, i]
            pred_vals = y_pred[:, i]

            all_true_vals.extend(true_vals)
            all_pred_vals.extend(pred_vals)

            handle = ax.scatter(
                true_vals,
                pred_vals,
                alpha=ALPHA,
                label=LABEL_MAP_HV.get(method, method),
                color=COLORS_SURROGATE_ACC[idx % len(COLORS_SURROGATE_ACC)],
                s=5,
            )
            plot_handles.append(handle)
            plot_labels.append(LABEL_MAP_HV.get(method, method))

            # (Metrics calculation is the same)
            # ...
            mse = np.mean((true_vals - pred_vals) ** 2)
            std = np.std(pred_vals - true_vals)
            stderr = std / np.sqrt(len(true_vals))
            tau, tau_p = kendalltau(true_vals, pred_vals)
            pearson, pearson_p = pearsonr(true_vals, pred_vals)
            spearman, spearman_p = spearmanr(true_vals, pred_vals)
            r2 = r2_score(true_vals, pred_vals)
            metrics_table.append(
                {
                    "Method": method,
                    "Objective": obj,
                    "MSE": mse,
                    "STD": std,
                    "StdError": stderr,
                    "KendallTau": tau,
                    "KendallTau_p": tau_p,
                    "PearsonR": pearson,
                    "PearsonR_p": pearson_p,
                    "SpearmanR": spearman,
                    "SpearmanR_p": spearman_p,
                    "R2": r2,
                    "N": len(true_vals),
                }
            )

        # --- START OF MODIFICATION: Axis limit calculation with quantiles ---
        if all_true_vals and all_pred_vals:
            # Calculate quantile-based limits for X-axis
            lower_quantile = (1.0 - AXIS_QUANTILE) / 2.0
            upper_quantile = 1.0 - lower_quantile

            x_min, x_max = (
                np.quantile(all_true_vals, lower_quantile),
                np.quantile(all_true_vals, upper_quantile),
            )
            x_range = x_max - x_min if x_max > x_min else 1.0
            x_padding = x_range * 0.10
            ax.set_xlim(x_min - x_padding, x_max + x_padding)

            # Calculate quantile-based limits for Y-axis
            y_min, y_max = (
                np.quantile(all_pred_vals, lower_quantile),
                np.quantile(all_pred_vals, upper_quantile),
            )
            y_range = y_max - y_min if y_max > y_min else 1.0
            y_padding = y_range * 0.10
            ax.set_ylim(y_min - y_padding, y_max + y_padding)

            # Plot the y=x line across the visible intersection
            line_min = max(ax.get_xlim()[0], ax.get_ylim()[0])
            line_max = min(ax.get_xlim()[1], ax.get_ylim()[1])
            ax.plot(
                [line_min, line_max],
                [line_min, line_max],
                "k--",
                lw=1.5,
                label="Perfect Prediction",
            )

            plot_handles.append(plt.Line2D([0], [0], color="k", lw=1.5, linestyle="--"))
            plot_labels.append("Perfect Prediction")
        # --- END OF MODIFICATION ---

        # ... (rest of the plotting code is the same) ...
        ax.set_xlabel(f"Ground Truth {obj}")
        ax.set_ylabel(f"Prediction {obj}")
        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(
            handles=plot_handles,
            labels=plot_labels,
            loc="best",
            frameon=True,
            framealpha=0.95,
            edgecolor="k",
            fancybox=False,
            ncol=1,
        )
        plt.tight_layout()
        for file_type in ["svg", "pdf", "png"]:
            dir_path = f"{plot_dir}/{file_type}/{obj}_scatter.{file_type}"
            os.makedirs(os.path.dirname(dir_path), exist_ok=True)
            plt.savefig(dir_path, dpi=300, bbox_inches="tight")
        plt.close()

    return metrics_table


def create_qq_plots(method_dataframes, objectives, plot_dir, title):
    """
    Generates and saves Quantile-Quantile (Q-Q) plots to compare the distribution
    of predicted values against the ground truth values.
    """
    import matplotlib

    matplotlib.use("Agg")  # For headless environments
    print("\n--- Generating Q-Q Plots ---")
    # Create a single figure with 1 x N subplots
    n_plots = len(objectives)
    if n_plots == 0:
        return

    fig, axs = plt.subplots(1, n_plots, figsize=(4 * n_plots, 5.5), squeeze=False)
    axs = axs[0]

    # We'll collect one color per method for the shared legend at the bottom
    # (we create legend proxies with larger markers so the legend markers are bigger
    # than the on-figure scatter points)
    legend_colors = {}

    # Store data for individual plots
    individual_plot_data = {}

    for i, obj in enumerate(objectives):
        ax = axs[i]
        all_plot_vals = []
        obj_data = []

        for idx, (method, df) in enumerate(method_dataframes.items()):
            expected_len = len(objectives)
            df_filtered = df[
                (
                    df["llm_prediction"].apply(
                        lambda x: isinstance(x, list) and len(x) == expected_len
                    )
                )
                & (
                    df["ground_truth"].apply(
                        lambda x: isinstance(x, list) and len(x) == expected_len
                    )
                )
            ].copy()

            if df_filtered.empty:
                continue

            # --- Q-Q Plot Core Logic ---
            true_quantiles = np.sort(
                np.array(df_filtered["ground_truth"].tolist())[:, i]
            )
            pred_quantiles = np.sort(
                np.array(df_filtered["llm_prediction"].tolist())[:, i]
            )

            all_plot_vals.extend(true_quantiles)
            all_plot_vals.extend(pred_quantiles)

            ax.scatter(
                true_quantiles,
                pred_quantiles,
                # alpha=ALPHA,
                label=LABEL_MAP_HV.get(method, method),
                color=get_color(method, idx),
                # color=COLORS_SURROGATE_ACC[idx % len(COLORS_SURROGATE_ACC)],
                s=5,
            )

            # register the color for the shared legend (only the first occurrence matters)
            label = LABEL_MAP_HV.get(method, method)
            if label not in legend_colors:
                # legend_colors[label] = COLORS_SURROGATE_ACC[
                #     idx % len(COLORS_SURROGATE_ACC)
                # ]
                legend_colors[label] = get_color(method, idx)

            # Store data for individual plots
            obj_data.append(
                {
                    "method": method,
                    "label": label,
                    "color": get_color(method, idx),
                    "true_quantiles": true_quantiles,
                    "pred_quantiles": pred_quantiles,
                }
            )

        # For Q-Q plots, square axes are essential for interpretation
        if all_plot_vals:
            lower_quantile = (1.0 - AXIS_QUANTILE) / 2.0
            upper_quantile = 1.0 - lower_quantile

            plot_min, plot_max = (
                np.quantile(all_plot_vals, lower_quantile),
                np.quantile(all_plot_vals, upper_quantile),
            )
            data_range = plot_max - plot_min if plot_max > plot_min else 1.0
            padding = data_range * 0.10
            final_min, final_max = plot_min - padding, plot_max + padding

            ax.set_xlim(final_min, final_max)
            ax.set_ylim(final_min, final_max)
            ax.plot([final_min, final_max], [final_min, final_max], "k--", lw=1.5)

            # Store limits for individual plots
            individual_plot_data[obj] = {
                "data": obj_data,
                "limits": (final_min, final_max),
                "all_plot_vals": all_plot_vals,
            }

        if i == 0:
            ax.set_ylabel(f"Prediction Quantiles")        
        ax.set_title(f"{obj}")
        ax.grid(True, linestyle="--", alpha=0.6)

        # For Vehicle Safety, remove extreme off-diagonal values for objectives 1 and 2
        # (Mass and Acceleration) to improve plot readability
        if "VehicleSafety" in title and i in [1, 2]:
            limit = 11 if i == 1 else 0.2
            limit_low = 7 if i == 1 else 0.05
            ax.set_xlim(left=limit_low, right=limit)
            ax.set_ylim(bottom=limit_low, top=limit)
        elif "CarSideImpact" in title and i in [1]:
            limit = 4.5
            limit_low = 3.5
            ax.set_xlim(left=limit_low, right=limit)
            ax.set_ylim(bottom=limit_low, top=limit)


    # Create proxies (larger markers) for the legend using the collected colors
    handles = []
    labels = []
    for label, color in legend_colors.items():
        proxy = plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=color,
            markeredgecolor=color,
            markersize=8,
        )
        handles.append(proxy)
        labels.append(label)

    # Add a proxy for perfect distribution match to the legend
    proxy_line = plt.Line2D([0], [0], color="k", lw=1.5, linestyle="--")
    handles.append(proxy_line)
    #labels.append("Perfect Distribution Match")

    ncol = min(max(1, len(labels)), 6)
    # Add global title above subplots
    fig.suptitle(title, y=0.9, fontsize=26)
    fig.supxlabel("Ground Truth Quantiles", y=0.2, fontsize=18)
    # Adjust layout to make room for the legend and title
    plt.tight_layout(rect=[0, 0.14, 1, 0.95])
    fig.legend(
        handles,
        labels,
        fontsize=18,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.08),
        ncol=ncol,
        frameon=True,
        framealpha=0.95,
        edgecolor="k",
    )

    # Save the combined figure
    for file_type in ["svg", "pdf", "png"]:
        dir_path = f"{plot_dir}/{file_type}"
        os.makedirs(dir_path, exist_ok=True)
        out_path = os.path.join(dir_path, f"qq_plots_{title}.{file_type}")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Create individual plots for each objective
    print("\n--- Generating Individual Q-Q Plots ---")
    for obj, plot_info in individual_plot_data.items():
        fig, ax = plt.subplots(figsize=(6, 4.5))
        plot_handles = []
        plot_labels = []

        for method_data in plot_info["data"]:
            ax.scatter(
                method_data["true_quantiles"],
                method_data["pred_quantiles"],
                label=method_data["label"],
                color=method_data["color"],
                s=5,
            )

            proxy = plt.Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markerfacecolor=method_data["color"],
                markeredgecolor=method_data["color"],
                markersize=8,
            )
            plot_handles.append(proxy)
            plot_labels.append(method_data["label"])

        # Set limits and draw diagonal line
        final_min, final_max = plot_info["limits"]
        ax.set_xlim(final_min, final_max)
        ax.set_ylim(final_min, final_max)
        ax.plot([final_min, final_max], [final_min, final_max], "k--", lw=1.5)

        # Add diagonal line to legend
        proxy_line = plt.Line2D([0], [0], color="k", lw=1.5, linestyle="--")
        plot_handles.append(proxy_line)
        plot_labels.append("Perfect Distribution Match")

        ax.set_xlabel(f"Ground Truth Quantiles ({obj})")
        ax.set_ylabel(f"Prediction Quantiles ({obj})")
        ax.set_title(f"{title}{obj}")
        ax.grid(True, linestyle="--", alpha=0.6)

        # Add legend at the bottom
        ncol = min(max(1, len(plot_labels)), 6)
        plt.tight_layout(rect=[0, 0.12, 1, 1])
        ax.legend(
            handles=plot_handles,
            labels=plot_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.25),
            ncol=2,
            frameon=True,
            framealpha=0.95,
            edgecolor="k",
        )

        # Save individual plot
        for file_type in ["svg", "pdf", "png"]:
            dir_path = f"{plot_dir}/{file_type}"
            os.makedirs(dir_path, exist_ok=True)
            out_path = os.path.join(dir_path, f"{obj}_qq_plot.{file_type}")
            plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()


def create_qq_plots_vertical(method_dataframes, objectives, plot_dir, title):
    """
    Generates and saves Quantile-Quantile (Q-Q) plots to compare the distribution
    of predicted values against the ground truth values.
    """
    import matplotlib

    matplotlib.use("Agg")  # For headless environments
    print("\n--- Generating Q-Q Plots ---")
    
    # Pick only the first 2 objectives
    n_plots = min(2, len(objectives))
    if n_plots == 0:
        return
    
    selected_objectives = [objectives[0], objectives[2]]

    # Create a single figure with 2 subplots stacked vertically (2 rows, 1 column)
    # Figure size is 4x4 inches
    fig, axs = plt.subplots(n_plots, 1, figsize=(4, 4), squeeze=False)
    axs = axs.flatten()

    # Store data for individual plots
    individual_plot_data = {}

    for i, obj in enumerate(selected_objectives):
        ax = axs[i]
        all_plot_vals = []
        obj_data = []

        for idx, (method, df) in enumerate(method_dataframes.items()):
            expected_len = len(objectives)
            df_filtered = df[
                (
                    df["llm_prediction"].apply(
                        lambda x: isinstance(x, list) and len(x) == expected_len
                    )
                )
                & (
                    df["ground_truth"].apply(
                        lambda x: isinstance(x, list) and len(x) == expected_len
                    )
                )
            ].copy()

            if df_filtered.empty:
                continue

            # Get the index for this objective in the original objectives list
            obj_idx = objectives.index(obj)
            
            # --- Q-Q Plot Core Logic ---
            true_quantiles = np.sort(
                np.array(df_filtered["ground_truth"].tolist())[:, obj_idx]
            )
            pred_quantiles = np.sort(
                np.array(df_filtered["llm_prediction"].tolist())[:, obj_idx]
            )

            all_plot_vals.extend(true_quantiles)
            all_plot_vals.extend(pred_quantiles)

            ax.scatter(
                true_quantiles,
                pred_quantiles,
                label=LABEL_MAP_HV.get(method, method),
                color=get_color(method, idx),
                marker=MARKER_MAP[method],
                s=6,  # Increased marker size
            )

            # Store data for individual plots
            obj_data.append(
                {
                    "method": method,
                    "label": LABEL_MAP_HV.get(method, method),
                    "color": get_color(method, idx),
                    "true_quantiles": true_quantiles,
                    "pred_quantiles": pred_quantiles,
                }
            )

        # For Q-Q plots, square axes are essential for interpretation
        if all_plot_vals:
            lower_quantile = (1.0 - AXIS_QUANTILE) / 2.0
            upper_quantile = 1.0 - lower_quantile

            plot_min, plot_max = (
                np.quantile(all_plot_vals, lower_quantile),
                np.quantile(all_plot_vals, upper_quantile),
            )
            data_range = plot_max - plot_min if plot_max > plot_min else 1.0
            padding = data_range * 0.10
            final_min, final_max = plot_min - padding, plot_max + padding

            ax.set_xlim(final_min, final_max)
            ax.set_ylim(final_min, final_max)
            ax.plot([final_min, final_max], [final_min, final_max], "k--", lw=1.5)

            # Store limits for individual plots
            individual_plot_data[obj] = {
                "data": obj_data,
                "limits": (final_min, final_max),
                "all_plot_vals": all_plot_vals,
            }

        # Increased font sizes for labels
        if i != 0:
            ax.set_xlabel(f"Ground Truth Quantiles", fontsize=11)
        # No title as requested
        ax.grid(True, linestyle="--", alpha=0.6)
        
        # Increase tick label font size
        ax.tick_params(axis='both', which='major', labelsize=6)
        ax.set_yticklabels([])

        
        # Remove the last xtick from the right side
        xticks = ax.get_xticks()
        if len(xticks) > 1:
            ax.set_xticks(xticks[:-2])

    # No legend, no global title
    # Reduce padding between subplots and remove right padding
    plt.tight_layout(pad=0.1, h_pad=0.1, w_pad=0.2)
    plt.subplots_adjust(left=0.09)

    plt.subplots_adjust(right=0.98)
    fig.supylabel("Prediction Quantiles", fontsize=11)

    # Save the combined figure
    for file_type in ["svg", "pdf", "png"]:
        dir_path = f"{plot_dir}/{file_type}"
        os.makedirs(dir_path, exist_ok=True)
        out_path = os.path.join(dir_path, f"qq_plots_vertical.{file_type}")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


# --- (save_metrics_markdown remains the same) ---
def save_metrics_markdown(metrics_table, plot_dir):
    """
    Saves the detailed metrics table, including correlation p-values, to a Markdown file.
    """
    md_path = os.path.join(plot_dir, "surrogate_accuracy_metrics.md")
    with open(md_path, "w") as f:
        f.write("# Surrogate Accuracy Metrics (Per Objective)\n\n")

        # --- START OF MODIFICATION: Add p-value columns to the header ---
        f.write(
            "| Method | Objective | R2 | MSE | STD | KendallTau | KendallTau_p | PearsonR | PearsonR_p | SpearmanR | SpearmanR_p | N |\n"
        )
        f.write("|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        # --- END OF MODIFICATION ---

        for row in sorted(metrics_table, key=lambda x: (x["Method"], x["Objective"])):
            # --- START OF MODIFICATION: Add p-value data to the row ---
            # P-values are formatted in scientific notation (e.g., 1.23e-45) for readability
            f.write(
                f"| {row['Method']} | {row['Objective']} | {row['R2']:.3f} | {row['MSE']:.2f} | {row['STD']:.2f} | "
                f"{row['KendallTau']:.3f} | {row['KendallTau_p']:.2e} | "
                f"{row['PearsonR']:.3f} | {row['PearsonR_p']:.2e} | "
                f"{row['SpearmanR']:.3f} | {row['SpearmanR_p']:.2e} | "
                f"{row['N']} |\n"
            )


# --- NEW FUNCTION: Save Averaged Metrics ---
def save_averaged_metrics_markdown(metrics_table, plot_dir):
    """Calculates and saves the average metrics across all objectives for each method."""
    if not metrics_table:
        return

    metrics_df = pd.DataFrame(metrics_table)
    avg_metrics = metrics_df.drop(columns=["Objective"]).groupby("Method").mean()
    total_n = metrics_df.groupby("Method")["N"].sum()
    avg_metrics["N"] = total_n

    md_path = os.path.join(plot_dir, "surrogate_accuracy_metrics_averaged.md")
    with open(md_path, "w") as f:
        f.write("# Surrogate Accuracy Metrics (Averaged Across Objectives)\n\n")
        # --- START OF MODIFICATION ---
        f.write(
            "| Method | Avg. R2 | Avg. MSE | Avg. STD | Avg. KendallTau | Avg. PearsonR | Avg. SpearmanR | Total N |\n"
        )
        f.write("|---|---|---|---|---|---|---|---|\n")
        for method, row in avg_metrics.sort_index().iterrows():
            f.write(
                f"| {method} | {row['R2']:.3f} | {row['MSE']:.2f} | {row['STD']:.2f} | {row['KendallTau']:.3f} | {row['PearsonR']:.3f} | {row['SpearmanR']:.3f} | {int(row['N'])} |\n"
            )


# --- (group_data_by_method and other helpers remain the same) ---
def group_data_by_method(file_list, filter_str):
    method_files = {}
    for file in file_list:
        if filter_str and filter_str not in file:
            continue
        try:
            method = file.split(os.sep)[-3]
            if method not in method_files:
                method_files[method] = []
            method_files[method].append(file)
        except IndexError:
            pass
    return method_files


# --- Main Execution Block ---
if __name__ == "__main__":
    main()
