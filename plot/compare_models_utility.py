import argparse
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Line2D previously used for grouped legends; grouping removed so no import needed
from pymoo.indicators.hv import HV
from plot_settings import get_color, LABEL_MAP_HV

# --- CONSISTENT PLOTTING STYLE CONSTANTS ---
# Central place to tune figure/layout fonts and sizes so every plot looks the same.
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


def apply_axis_style(ax):
    """Apply consistent axis styling: label sizes, title size, tick label sizes.

    This helper centralizes the per-axis styling so all plots look uniform.
    """
    try:
        ax.title.set_size(TITLE_SIZE)
    except Exception:
        pass
    try:
        ax.xaxis.label.set_size(AX_LABELSIZE)
        ax.yaxis.label.set_size(AX_LABELSIZE)
    except Exception:
        pass
    try:
        ax.tick_params(axis="both", which="major", labelsize=TICK_LABELSIZE)
    except Exception:
        pass


def compute_hypervolume(df, reference_point):
    """Compute hypervolume for a dataframe of objective columns using pymoo.HV.

    df: pandas.DataFrame with objective columns (already normalized if needed)
    reference_point: list-like reference point length == n_objectives
    """
    objectives = df.to_numpy()
    ind = HV(ref_point=reference_point)
    return float(ind(objectives))


def convert_data_to_hv_over_time(fvals, reference_point=[1.0, 1.0]):
    """Convert a dataframe of fvals into a list of hypervolume values over time.

    At step t we compute HV of first t rows.
    """
    hypervolume = []
    for step in range(1, len(fvals) + 1):
        hv = compute_hypervolume(fvals.iloc[:step], reference_point)
        hypervolume.append(hv)
    return np.array(hypervolume)


def normalize_data(fvals, min_max_metrics):
    """Min-max normalize the objective columns in-place and return the df.

    min_max_metrics: dict mapping column -> {"min":..., "max":...}
    """
    for column, values in min_max_metrics.items():
        min_val = values["min"]
        max_val = values["max"]
        # avoid division by zero
        denom = max_val - min_val if max_val != min_val else 1.0
        fvals[column] = (fvals[column] - min_val) / denom
    return fvals


def main():
    """
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
    parser.add_argument("--benchmarks", type=str, required=True, help="Benchmark names")
    parser.add_argument("--title", type=str, required=True, help="Plot title")
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
    benchmarks = args.benchmarks.split(",")
    title = args.title
    filter = args.filter
    trials = args.trials
    filename = args.filename
    blacklist = args.blacklist.split(",") if args.blacklist else []
    whitelist = args.whitelist.split(",") if args.whitelist else []

    output_path = "./plots/utility_models/"
    os.makedirs(output_path, exist_ok=True)

    print("#" * 80)
    print("Plotting utility")
    print(f"Benchmarks:   {benchmarks}")
    print(f"Title:       {title}")
    print(f"Filename:    {filename}")
    print(f"Output Path: {output_path}")
    print(f"Filter:      {filter}")
    print(f"Trials:      {trials}")
    print(f"Blacklist:   {blacklist}")
    print(f"Whitelist:   {whitelist}\n")

    aggregated_data = {
        "token_usage_per_request": {},
        "time_taken_per_trials": {},
        "cost_per_request": {},
        "filtering_stats": {},
    }

    # Keep aggregation also per-benchmark so we can produce a table with one
    # row per benchmark (and per method) later.
    aggregated_by_benchmark = {}

    base_data_path = "./results"

    for benchmark in benchmarks:
        print(benchmark)
        for data_folder in [
            "token_usage_per_request",
            "time_taken_per_trials",
            "cost_per_request",
            "filtering_stats",
        ]:
            search_path = os.path.join(base_data_path, benchmark, "**", "*.csv")
            file_names = glob.glob(search_path, recursive=True)

            observed_fvals_files = [
                file for file in file_names if data_folder in file.split(os.sep)
            ]
            method_files = group_data_by_method(observed_fvals_files, filter)

            method_dfs = {}
            for method, files in method_files.items():
                dfs = []
                for file in files:
                    try:
                        df = pd.read_csv(file)
                        dfs.append(df)
                    except pd.errors.EmptyDataError as e:
                        print(f"EmptyDataError while reading file: {file} -- {e}")
                    except Exception as e:
                        print(f"Error while reading file: {file} -- {e}")
                method_dfs[method] = dfs

            if whitelist and whitelist[0] != "":
                filtered_dfs = {
                    m: dfs
                    for m, dfs in method_dfs.items()
                    if any(entry == m for entry in whitelist)
                }
                method_dfs = filtered_dfs
            elif blacklist and blacklist[0] != "":
                filtered_dfs = {
                    m: dfs
                    for m, dfs in method_dfs.items()
                    if not any(entry in m for entry in blacklist)
                }
                method_dfs = filtered_dfs

            for method, dfs in method_dfs.items():
                if method not in aggregated_data[data_folder]:
                    aggregated_data[data_folder][method] = []
                aggregated_data[data_folder][method].extend(dfs)
            # store per-benchmark grouping
            aggregated_by_benchmark.setdefault(benchmark, {})[data_folder] = method_dfs

    # --- Compute hypervolume per-trial from observed_fvals and create HV vs Cost/Time plots per benchmark
    token_data_agg = process_tokens_per_trial(
        aggregated_data["token_usage_per_request"], trials
    )
    time_data = process_time_taken_per_trials(aggregated_data["time_taken_per_trials"])
    cost_data_agg = process_cost(aggregated_data["cost_per_request"], trials)
    requests_data = process_requests_per_trial(
        aggregated_data["token_usage_per_request"], trials
    )

    create_operational_costs_dashboard(
        time_data,
        token_data_agg,
        cost_data_agg,
        requests_data,
        output_path,
        filename,
        trials,
    )
    create_summary_table(aggregated_by_benchmark, trials, output_path)
    # Process and plot filtering stats (rejection rates)
    filtering_data = process_filtering_stats(aggregated_data["filtering_stats"], trials)
    filtering_plot_dir = os.path.join(output_path, "filtering_stats_plots")
    os.makedirs(filtering_plot_dir, exist_ok=True)
    # Grouped plot: two subplots (MOHOLLM vs LLM) with mean ± std shading per rejection reason
    # Create a rejection rate plot that shows each method independently
    create_rejection_rate_plot(filtering_data, filtering_plot_dir, filename)
    # Summary table for filtering stats
    create_filtering_summary_table(filtering_data, output_path)
    # Also write requests per trial plot (cumulative requests) if we have requests data
    create_requests_per_trial_plot(requests_data, output_path, filename)


# --- (process_tokens_per_trial and process_cost are unchanged) ---
def process_tokens_per_trial(data, trials):
    aggregated = {}
    for method, dfs in data.items():
        method_aggregated_seeds = []
        for df in dfs:
            if "trial" in df.columns:
                grouped = df.groupby("trial", as_index=False)[
                    ["prompt_tokens", "completion_tokens", "total_tokens"]
                ].sum()
                if trials is not None:
                    grouped = grouped[grouped["trial"] < trials]
                method_aggregated_seeds.append(grouped)
        if method_aggregated_seeds:
            aggregated[method] = method_aggregated_seeds
    return aggregated


def process_time_taken_per_trials(data):
    processed = {}
    for method, dfs in data.items():
        method_seeds = []
        for df in dfs:
            method_seeds.append(df)
        if method_seeds:
            processed[method] = method_seeds
    return processed


def process_cost(data, trials):
    aggregated = {}
    for method, dfs in data.items():
        method_aggregated_seeds = []
        for df in dfs:
            if "trial" in df.columns:
                grouped = df.groupby("trial", as_index=False)[
                    ["prompt_cost", "completion_cost", "total_cost"]
                ].sum()
                if trials is not None:
                    grouped = grouped[grouped["trial"] < trials]
                method_aggregated_seeds.append(grouped)
        if method_aggregated_seeds:
            aggregated[method] = method_aggregated_seeds
    return aggregated


# --- START OF NEW FUNCTION ---
def process_requests_per_trial(data, trials):
    """Counts the number of API requests per trial for each seed."""
    processed = {}
    for method, dfs in data.items():
        seed_counts = []
        for df in dfs:
            if "trial" in df.columns:
                # Count occurrences of each trial number
                counts = df["trial"].value_counts().sort_index()
                if trials is not None:
                    counts = counts[counts.index < trials]
                seed_counts.append(counts)
        if seed_counts:
            processed[method] = seed_counts
    return processed


# --- START OF NEW FUNCTIONS ---


def process_filtering_stats(data, trials):
    """Aggregates filtering stats by trial for each seed."""
    processed = {}
    for method, dfs in data.items():
        seed_stats = []
        for df in dfs:
            if "trial" in df.columns:
                # Sum up stats for each trial (since a trial can have multiple entries)
                agg_df = df.groupby("trial").sum()
                if trials is not None:
                    agg_df = agg_df[agg_df.index < trials]
                seed_stats.append(agg_df)
        if seed_stats:
            processed[method] = seed_stats
    return processed


def create_rejection_rate_plot(filtering_data, path, filename):
    """Creates a plot of the rejection rate per trial."""
    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    methods = sorted(filtering_data.keys())

    for i, method in enumerate(methods):
        color = get_color(method, i)
        seed_curves = []
        for df_seed in filtering_data[method]:
            # Calculate rejection rate, handle division by zero
            total_rejected = df_seed[
                ["rejected_duplicates", "rejected_observed", "rejected_region"]
            ].sum(axis=1)
            rejection_rate = (
                total_rejected / df_seed["num_proposed_candidates"]
            ).fillna(0) * 100
            seed_curves.append(rejection_rate)

        if not seed_curves:
            continue

        # Align data to a common trial index for averaging
        max_trial = max(s.index.max() for s in seed_curves if not s.empty)
        common_index = pd.RangeIndex(start=0, stop=max_trial + 1)
        aligned_seeds = [s.reindex(common_index, fill_value=0) for s in seed_curves]

        curves = np.array([s.values for s in aligned_seeds])
        mean_rate = np.mean(curves, axis=0)

        ax.plot(common_index, mean_rate, label=method, color=color, lw=2)

        if len(aligned_seeds) > 1:
            n = len(aligned_seeds)
            std_err = 1.96 * np.std(curves, axis=0, ddof=1) / np.sqrt(n)
            ax.fill_between(
                common_index,
                mean_rate - std_err,
                mean_rate + std_err,
                color=color,
                alpha=0.2,
                linewidth=0,
            )

    ax.set_xlabel("Trial")
    ax.set_ylabel("Rejection Rate (%) (Mean ± Std. Error)")
    ax.set_title("LLM Proposal Rejection Rate vs. Trial")
    ax.grid(True, linestyle="--", alpha=0.6)
    # Apply consistent axis styling
    apply_axis_style(ax)
    ax.legend(loc="upper left")
    plt.tight_layout()

    for file_type in ["svg", "pdf", "png"]:
        dir_path = f"{path}/{file_type}/{filename}_rejection_rate.{file_type}"
        os.makedirs(os.path.dirname(dir_path), exist_ok=True)
        plt.savefig(dir_path, dpi=300, bbox_inches="tight")
    plt.close()


# Grouped MOHOLLM/LLM plotting removed. Each method is plotted and evaluated independently.


def create_filtering_summary_table(filtering_data, path):
    """Creates a Markdown table summarizing total rejection statistics."""
    table_data = []
    methods = sorted(filtering_data.keys())

    for method in methods:
        seed_dfs = filtering_data[method]
        if not seed_dfs:
            continue

        # Sum stats across all trials for each seed, then average across seeds
        total_proposed = np.mean(
            [df["num_proposed_candidates"].sum() for df in seed_dfs]
        )
        total_dupes = np.mean([df["rejected_duplicates"].sum() for df in seed_dfs])
        total_observed = np.mean([df["rejected_observed"].sum() for df in seed_dfs])
        total_region = np.mean([df["rejected_region"].sum() for df in seed_dfs])

        total_rejected = total_dupes + total_observed + total_region

        # Calculate rates
        total_rejection_rate = (
            (total_rejected / total_proposed * 100) if total_proposed > 0 else 0
        )
        duplicate_rate = (
            (total_dupes / total_proposed * 100) if total_proposed > 0 else 0
        )
        observed_rate = (
            (total_observed / total_proposed * 100) if total_proposed > 0 else 0
        )
        region_rate = (total_region / total_proposed * 100) if total_proposed > 0 else 0

        table_data.append(
            {
                "Method": LABEL_MAP_HV.get(method, method),
                "Avg. Total Proposed": total_proposed,
                "Total Rejection Rate (%)": total_rejection_rate,
                "Duplicate Rate (%)": duplicate_rate,
                "Observed Rate (%)": observed_rate,
                "Out of Region Rate (%)": region_rate,
            }
        )

    df_summary = pd.DataFrame(table_data)

    md_path = os.path.join(path, "filtering_stats_summary.md")
    with open(md_path, "w") as f:
        f.write("# Summary of LLM Proposal Rejection Statistics\n\n")
        f.write(
            df_summary.to_markdown(
                index=False, floatfmt=(".1f", ".2f", ".2f", ".2f", ".2f")
            )
        )

    print(f"\nFiltering summary table saved to {md_path}")


# --- END OF NEW FUNCTIONS ---


# --- (create_operational_costs_dashboard is unchanged) ---
def create_operational_costs_dashboard(
    time_data, token_data, cost_data, requests_data, path, filename, trials
):
    # Create 2x2 subplots: Time | Tokens
    #                      Requests | Cost
    fig, axes = plt.subplots(2, 2, figsize=DEFAULT_FIGSIZE, sharex="col")
    ax1 = axes[0, 0]  # Time
    ax2 = axes[0, 1]  # Tokens
    ax3 = axes[1, 0]  # Requests
    ax4 = axes[1, 1]  # Cost

    # collect all methods that might appear in any dataset
    methods = sorted(
        set(
            list(time_data.keys())
            + list(token_data.keys())
            + list(cost_data.keys())
            + (list(requests_data.keys()) if requests_data else []),
        )
    )

    for i, method in enumerate(methods):
        color = get_color(method, i)
        if method in time_data:
            time_curves = [
                df["trial_total_time"].cumsum().values[:trials]
                for df in time_data[method]
            ]
            pd.DataFrame(time_curves).to_csv("time_curves.csv")

            time_curves_trunc = np.array([c for c in time_curves])
            mean_time = np.mean(time_curves_trunc, axis=0)
            std_error = 1.96 * np.std(time_curves, axis=0) / np.sqrt(len(mean_time))
            ax1.plot(
                range(trials),
                mean_time,
                label=LABEL_MAP_HV.get(method, method),
                color=color,
                lw=2,
            )
            ax1.fill_between(
                range(trials),
                mean_time - std_error,
                mean_time + std_error,
                color=color,
                alpha=0.2,
                linewidth=0,
            )
        if method in token_data:
            token_curves = [
                df["total_tokens"].cumsum().values[:trials] for df in token_data[method]
            ]
            token_curves_trunc = np.array([c for c in token_curves])
            mean_tokens = np.mean(token_curves_trunc, axis=0)
            std_error = (
                1.96 * np.std(token_curves_trunc, axis=0) / np.sqrt(len(mean_tokens))
            )
            ax2.plot(
                range(trials),
                mean_tokens,
                label=LABEL_MAP_HV.get(method, method),
                color=color,
                lw=2,
            )
            ax2.fill_between(
                range(trials),
                mean_tokens - std_error,
                mean_tokens + std_error,
                color=color,
                alpha=0.2,
                linewidth=0,
            )
            # Use scientific notation for large token counts and place exponent on top
            ax2.ticklabel_format(style="sci", axis="y", scilimits=(6, 6))
            try:
                # Place the exponent (offset text) above the axis
                ax2.yaxis.get_offset_text().set_x(0.5)
                ax2.yaxis.set_offset_position("left")
            except Exception:
                pass
        # Plot cumulative requests in their own subplot (ax3)
        if requests_data and method in requests_data:
            seed_counts = requests_data[method]
            # Align seeds to common index
            max_trial = max(s.index.max() for s in seed_counts if not s.empty)
            common_index = pd.RangeIndex(start=0, stop=max_trial + 1)
            aligned_seeds = [s.reindex(common_index, fill_value=0) for s in seed_counts]
            curves = np.array([s.values for s in aligned_seeds])
            # Convert to cumulative per-seed
            cum_curves = np.cumsum(curves, axis=1)[:, :trials]
            mean_requests_cum = np.mean(cum_curves, axis=0)
            std_requests = np.std(cum_curves, axis=0)
            ax3.plot(
                range(trials),
                mean_requests_cum,
                label=LABEL_MAP_HV.get(method, method) + " requests",
                color=color,
                linestyle="-",
                lw=1.8,
            )
            req_ci = 1.96 * std_requests / np.sqrt(max(1, cum_curves.shape[0]))
            ax3.fill_between(
                range(trials),
                mean_requests_cum - req_ci,
                mean_requests_cum + req_ci,
                color=color,
                alpha=0.12,
                linewidth=0,
            )
        if method in cost_data:
            cost_curves = [
                df["total_cost"].cumsum().values[:trials] for df in cost_data[method]
            ]
            cost_curves_trunc = np.array([c for c in cost_curves])
            mean_cost = np.mean(cost_curves_trunc, axis=0)
            std_error = (
                1.96 * np.std(cost_curves_trunc, axis=0) / np.sqrt(len(mean_cost))
            )
            ax4.plot(
                range(trials),
                mean_cost,
                label=LABEL_MAP_HV.get(method, method),
                color=color,
                lw=2,
            )
            ax4.fill_between(
                range(trials),
                mean_cost - std_error,
                mean_cost + std_error,
                color=color,
                alpha=0.2,
                linewidth=0,
            )
    ax1.set_ylabel(r"Time (s)")
    # Show tokens in units of 10^6 to avoid large tick labels
    ax2.set_ylabel(r"Total Tokens")

    ax3.set_ylabel("Requests")
    ax3.set_xlabel("Trials")

    ax4.set_ylabel(r"Cost (\$)")
    ax4.set_xlabel("Trials")
    # fig.suptitle(
    #     "Cumulative Operational Costs vs. Trials", fontsize=14, fontweight="bold"
    # )
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax3.grid(True, linestyle="--", alpha=0.6)
    ax4.grid(True, linestyle="--", alpha=0.6)
    # Apply consistent axis styling to all subplots
    for ax in [ax1, ax2, ax3, ax4]:
        apply_axis_style(ax)
    ax1.legend(loc="upper left", ncol=2)
    plt.tight_layout()
    for file_type in ["svg", "pdf", "png"]:
        dir_path = f"{path}/{file_type}/{filename}_dashboard.{file_type}"
        os.makedirs(os.path.dirname(dir_path), exist_ok=True)
        plt.savefig(dir_path, dpi=300, bbox_inches="tight")
    plt.close()


# --- START OF NEW PLOTTING FUNCTION ---
def create_requests_per_trial_plot(requests_data, path, filename):
    """Creates a plot showing the number of API requests per trial."""
    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    methods = sorted(requests_data.keys())

    for i, method in enumerate(methods):
        color = get_color(method, i)
        seed_counts_list = requests_data[method]
        if not seed_counts_list:
            continue

        # Align data from different seeds to a common trial index
        max_trial = max(s.index.max() for s in seed_counts_list if not s.empty)
        common_index = pd.RangeIndex(start=0, stop=max_trial + 1)
        aligned_seeds = [
            s.reindex(common_index, fill_value=0) for s in seed_counts_list
        ]

        curves = np.array([s.values for s in aligned_seeds])
        mean_requests = np.mean(curves, axis=0)

        ax.plot(common_index, mean_requests, label=method, color=color, lw=2)

        if len(aligned_seeds) > 1:
            n = len(aligned_seeds)
            std_err = 1.96 * np.std(curves, axis=0, ddof=1) / np.sqrt(n)
            ax.fill_between(
                common_index,
                mean_requests - std_err,
                mean_requests + std_err,
                color=color,
                alpha=0.2,
                linewidth=0,
            )

    ax.set_xlabel("Trial")
    ax.set_ylabel("Number of API Requests (Mean ± Std. Error)")
    ax.set_title("API Requests per Trial")
    ax.grid(True, linestyle="--", alpha=0.6)
    apply_axis_style(ax)
    ax.legend(loc="upper right")
    plt.tight_layout()

    for file_type in ["svg", "pdf", "png"]:
        dir_path = f"{path}/{file_type}/{filename}_requests_per_trial.{file_type}"
        os.makedirs(os.path.dirname(dir_path), exist_ok=True)
        plt.savefig(dir_path, dpi=300, bbox_inches="tight")
    plt.close()


def _align_and_pad_series(series_list, pad_value=0.0):
    """Align a list of 1D numpy arrays (or lists) to the same length by padding with pad_value."""
    max_len = max(len(s) for s in series_list)
    aligned = [
        np.pad(s, (0, max_len - len(s)), constant_values=pad_value) for s in series_list
    ]
    return np.array(aligned)


def create_summary_table(aggregated_by_benchmark, trials, path):
    """Creates and saves two Markdown tables (MOHOLLM and LLM) summarizing costs and requests.

    aggregated_by_benchmark: dict mapping benchmark -> data_folder -> method -> list[df]
    trials: int or None, used to cap per-trial calculations
    path: output directory for the markdown files
    """
    rows = []

    def safe_mean(lst):
        return float(np.mean(lst)) if lst else 0.0

    # helpers
    def avg_time_per_trial(dfs):
        vals = []
        for df in dfs:
            if df is None or df.empty:
                continue
            if "trial_total_time" in df.columns:
                # Some CSVs contain a 'trial' column, others only list trial_total_time per row.
                # Support both formats: when 'trial' exists, group by it; otherwise treat each
                # row as one trial (use the row index as implicit trial number).
                if "trial" in df.columns:
                    g = df.groupby("trial", as_index=False)["trial_total_time"].sum()
                else:
                    g = (
                        df[["trial_total_time"]]
                        .reset_index()
                        .rename(columns={"index": "trial"})
                    )
                if trials is not None:
                    g = g[g["trial"] < trials]
                if len(g) > 0:
                    vals.append(g["trial_total_time"].sum() / len(g))
        return safe_mean(vals)

    def avg_tokens_per_trial(dfs):
        prompt_vals = []
        completion_vals = []
        total_vals = []
        for df in dfs:
            if df is None or df.empty:
                continue
            # Accept two possible formats: (a) a dataframe with a 'trial' column
            # or (b) a dataframe where each row corresponds to a trial and the
            # token columns exist.
            if all(
                col in df.columns
                for col in ("prompt_tokens", "completion_tokens", "total_tokens")
            ):
                if "trial" in df.columns:
                    g = df.groupby("trial", as_index=False)[
                        ["prompt_tokens", "completion_tokens", "total_tokens"]
                    ].sum()
                else:
                    g = (
                        df[["prompt_tokens", "completion_tokens", "total_tokens"]]
                        .reset_index()
                        .rename(columns={"index": "trial"})
                    )
                if trials is not None:
                    g = g[g["trial"] < trials]
                if len(g) > 0:
                    prompt_vals.append(g["prompt_tokens"].sum() / len(g))
                    completion_vals.append(g["completion_tokens"].sum() / len(g))
                    total_vals.append(g["total_tokens"].sum() / len(g))
        return safe_mean(prompt_vals), safe_mean(completion_vals), safe_mean(total_vals)

    def avg_requests_per_trial(dfs):
        vals = []
        for df in dfs:
            if df is None or df.empty:
                continue
            if "trial" in df.columns:
                counts = df["trial"].value_counts().sort_index()
                if trials is not None:
                    counts = counts[counts.index < trials]
                denom = (
                    trials
                    if trials is not None
                    else (counts.index.max() + 1 if not counts.empty else 0)
                )
                if denom > 0:
                    vals.append(counts.sum() / denom)
            else:
                # If trial column is missing, assume each row represents one request
                # and distribute them evenly across `trials` if provided, otherwise
                # treat the number of rows as the number of trials.
                denom = trials if trials is not None else len(df)
                if denom > 0:
                    vals.append(len(df) / float(denom))
        return safe_mean(vals)

    def avg_cost_per_trial(dfs):
        vals = []
        for df in dfs:
            if df is None or df.empty:
                continue
            if "trial" in df.columns and "total_cost" in df.columns:
                g = df.groupby("trial", as_index=False)["total_cost"].sum()
                if trials is not None:
                    g = g[g["trial"] < trials]
                if len(g) > 0:
                    vals.append(g["total_cost"].sum() / len(g))
        return safe_mean(vals)

    def avg_total_cost_sum(dfs):
        # average of total_cost summed across trials per seed
        vals = []
        for df in dfs:
            if df is None or df.empty:
                continue
            if "total_cost" in df.columns:
                if "trial" in df.columns and trials is not None:
                    df2 = df[df["trial"] < trials]
                else:
                    df2 = df
                vals.append(df2["total_cost"].sum())
        return safe_mean(vals)

    def avg_total_time_sum(dfs):
        # average of trial_total_time summed across trials per seed
        vals = []
        for df in dfs:
            if df is None or df.empty:
                continue
            if "trial_total_time" in df.columns:
                if "trial" in df.columns and trials is not None:
                    df2 = df[df["trial"] < trials]
                else:
                    df2 = df
                vals.append(df2["trial_total_time"].sum())
        return safe_mean(vals)

    def avg_total_tokens_sum(dfs):
        # average of total_tokens summed across trials per seed
        vals = []
        for df in dfs:
            if df is None or df.empty:
                continue
            if "total_tokens" in df.columns:
                if "trial" in df.columns and trials is not None:
                    df2 = df[df["trial"] < trials]
                else:
                    df2 = df
                vals.append(df2["total_tokens"].sum())
        return safe_mean(vals)

    # --- STATISTICS HELPERS (return mean, std) ---
    def stats_time_per_trial(dfs):
        vals = []
        for df in dfs:
            if df is None or df.empty:
                continue
            if "trial_total_time" in df.columns:
                if "trial" in df.columns:
                    g = df.groupby("trial", as_index=False)["trial_total_time"].sum()
                else:
                    g = (
                        df[["trial_total_time"]]
                        .reset_index()
                        .rename(columns={"index": "trial"})
                    )
                if trials is not None:
                    g = g[g["trial"] < trials]
                if len(g) > 0:
                    vals.append(g["trial_total_time"].sum() / len(g))
        if not vals:
            return (0.0, 0.0)
        return (
            float(np.mean(vals)),
            float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        )

    def stats_tokens_per_trial(dfs):
        p_vals = []
        c_vals = []
        t_vals = []
        for df in dfs:
            if df is None or df.empty:
                continue
            if all(
                col in df.columns
                for col in ("prompt_tokens", "completion_tokens", "total_tokens")
            ):
                if "trial" in df.columns:
                    g = df.groupby("trial", as_index=False)[
                        ["prompt_tokens", "completion_tokens", "total_tokens"]
                    ].sum()
                else:
                    g = (
                        df[["prompt_tokens", "completion_tokens", "total_tokens"]]
                        .reset_index()
                        .rename(columns={"index": "trial"})
                    )
                if trials is not None:
                    g = g[g["trial"] < trials]
                if len(g) > 0:
                    p_vals.append(g["prompt_tokens"].sum() / len(g))
                    c_vals.append(g["completion_tokens"].sum() / len(g))
                    t_vals.append(g["total_tokens"].sum() / len(g))

        def _ms(vals):
            if not vals:
                return (0.0, 0.0)
            return (
                float(np.mean(vals)),
                float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            )

        return (_ms(p_vals), _ms(c_vals), _ms(t_vals))

    def stats_requests_per_trial(dfs):
        vals = []
        for df in dfs:
            if df is None or df.empty:
                continue
            if "trial" in df.columns:
                counts = df["trial"].value_counts().sort_index()
                if trials is not None:
                    counts = counts[counts.index < trials]
                denom = (
                    trials
                    if trials is not None
                    else (counts.index.max() + 1 if not counts.empty else 0)
                )
                if denom > 0:
                    vals.append(counts.sum() / denom)
            else:
                denom = trials if trials is not None else len(df)
                if denom > 0:
                    vals.append(len(df) / float(denom))
        if not vals:
            return (0.0, 0.0)
        return (
            float(np.mean(vals)),
            float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        )

    def stats_cost_per_trial(dfs):
        vals = []
        for df in dfs:
            if df is None or df.empty:
                continue
            if "trial" in df.columns and "total_cost" in df.columns:
                g = df.groupby("trial", as_index=False)["total_cost"].sum()
                if trials is not None:
                    g = g[g["trial"] < trials]
                if len(g) > 0:
                    vals.append(g["total_cost"].sum() / len(g))
        if not vals:
            return (0.0, 0.0)
        return (
            float(np.mean(vals)),
            float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        )

    def stats_total_cost_sum(dfs):
        vals = []
        for df in dfs:
            if df is None or df.empty:
                continue
            if "total_cost" in df.columns:
                if "trial" in df.columns and trials is not None:
                    df2 = df[df["trial"] < trials]
                else:
                    df2 = df
                vals.append(df2["total_cost"].sum())
        if not vals:
            return (0.0, 0.0)
        return (
            float(np.mean(vals)),
            float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        )

    def stats_total_time_sum(dfs):
        vals = []
        for df in dfs:
            if df is None or df.empty:
                continue
            if "trial_total_time" in df.columns:
                if "trial" in df.columns and trials is not None:
                    df2 = df[df["trial"] < trials]
                else:
                    df2 = df
                vals.append(df2["trial_total_time"].sum())
        if not vals:
            return (0.0, 0.0)
        return (
            float(np.mean(vals)),
            float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        )

    def stats_total_tokens_sum(dfs):
        vals = []
        for df in dfs:
            if df is None or df.empty:
                continue
            if "total_tokens" in df.columns:
                if "trial" in df.columns and trials is not None:
                    df2 = df[df["trial"] < trials]
                else:
                    df2 = df
                vals.append(df2["total_tokens"].sum())
        if not vals:
            return (0.0, 0.0)
        return (
            float(np.mean(vals)),
            float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        )

    method_rows = {}
    # iterate benchmarks
    for benchmark in sorted(aggregated_by_benchmark.keys()):
        bench_data = aggregated_by_benchmark[benchmark]
        token_dict = bench_data.get("token_usage_per_request", {}) or {}
        time_dict = bench_data.get("time_taken_per_trials", {}) or {}
        cost_dict = bench_data.get("cost_per_request", {}) or {}

        methods = sorted(
            set(
                list(token_dict.keys())
                + list(time_dict.keys())
                + list(cost_dict.keys())
            )
        )
        for method in methods:
            time_mean, time_std = stats_time_per_trial(time_dict.get(method, []))
            (p_mean, p_std), (c_mean, c_std), (tot_mean, tot_std) = (
                stats_tokens_per_trial(token_dict.get(method, []))
            )
            req_mean, req_std = stats_requests_per_trial(token_dict.get(method, []))
            cost_mean, cost_std = stats_cost_per_trial(cost_dict.get(method, []))
            cost_sum_mean, cost_sum_std = stats_total_cost_sum(
                cost_dict.get(method, [])
            )
            total_time_mean, total_time_std = stats_total_time_sum(
                time_dict.get(method, [])
            )
            total_tokens_sum_mean, total_tokens_sum_std = stats_total_tokens_sum(
                token_dict.get(method, [])
            )

            row = {
                "Benchmark": benchmark,
                "Method": LABEL_MAP_HV.get(method, method),
                # numeric means (for later numeric aggregation if needed)
                "Avg. Time / Trial (s)_mean": time_mean,
                "Avg. Time / Trial (s)_std": time_std,
                "Avg. Prompt Tokens / Trial_mean": p_mean,
                "Avg. Prompt Tokens / Trial_std": p_std,
                "Avg. Completion Tokens / Trial_mean": c_mean,
                "Avg. Completion Tokens / Trial_std": c_std,
                "Avg. Total Tokens / Trial_mean": tot_mean,
                "Avg. Total Tokens / Trial_std": tot_std,
                "Avg. Total Tokens (sum)_mean": total_tokens_sum_mean,
                "Avg. Total Tokens (sum)_std": total_tokens_sum_std,
                "Avg. Requests / Trial_mean": req_mean,
                "Avg. Requests / Trial_std": req_std,
                "Avg. Total Cost / Trial ($)_mean": cost_mean,
                "Avg. Total Cost / Trial ($)_std": cost_std,
                "Avg. Total Cost ($)_mean": cost_sum_mean,
                "Avg. Total Cost ($)_std": cost_sum_std,
                "Avg. Total Time (s)_mean": total_time_mean,
                "Avg. Total Time (s)_std": total_time_std,
                # formatted display strings
                "Avg. Time / Trial (s)": f"{time_mean:.2f} ± {time_std:.2f}",
                "Avg. Prompt Tokens / Trial": f"{int(round(p_mean))} ± {int(round(p_std))}",
                "Avg. Completion Tokens / Trial": f"{int(round(c_mean))} ± {int(round(c_std))}",
                "Avg. Total Tokens / Trial": f"{int(round(tot_mean))} ± {int(round(tot_std))}",
                "Avg. Total Tokens (sum)": f"{int(round(total_tokens_sum_mean))} ± {int(round(total_tokens_sum_std))}",
                "Avg. Requests / Trial": f"{int(round(req_mean))} ± {req_std:.1f}",
                "Avg. Total Cost / Trial ($)": f"{(cost_mean / 1e-3):.3f} ± {(cost_std / 1e-3):.3f}",
                "Avg. Total Cost ($)": f"{cost_sum_mean:.5f} ± {cost_sum_std:.5f}",
                "Avg. Total Time (s)": f"{total_time_mean:.2f} ± {total_time_std:.2f}",
            }
            rows.append(row)
            method_rows.setdefault(method, []).append(row)

    # compute per-method average rows using the numeric *_mean and *_std fields
    summary_rows = []
    for method, vals in method_rows.items():
        dfm = pd.DataFrame(vals)

        # list of (mean_col, std_col, display_format, display_key)
        cols_spec = [
            (
                "Avg. Time / Trial (s)_mean",
                "Avg. Time / Trial (s)_std",
                "{:.2f} ± {:.2f}",
                "Avg. Time / Trial (s)",
            ),
            (
                "Avg. Prompt Tokens / Trial_mean",
                "Avg. Prompt Tokens / Trial_std",
                "{:.0f} ± {:.0f}",
                "Avg. Prompt Tokens / Trial",
            ),
            (
                "Avg. Completion Tokens / Trial_mean",
                "Avg. Completion Tokens / Trial_std",
                "{:.0f} ± {:.0f}",
                "Avg. Completion Tokens / Trial",
            ),
            (
                "Avg. Total Tokens / Trial_mean",
                "Avg. Total Tokens / Trial_std",
                "{:.0f} ± {:.0f}",
                "Avg. Total Tokens / Trial",
            ),
            (
                "Avg. Total Tokens (sum)_mean",
                "Avg. Total Tokens (sum)_std",
                "{:.0f} ± {:.0f}",
                "Avg. Total Tokens (sum)",
            ),
            (
                "Avg. Requests / Trial_mean",
                "Avg. Requests / Trial_std",
                "{:.0f} ± {:.1f}",
                "Avg. Requests / Trial",
            ),
            (
                "Avg. Total Cost / Trial ($)_mean",
                "Avg. Total Cost / Trial ($)_std",
                "{:.3f} ± {:.3f}",
                "Avg. Total Cost / Trial ($)",
            ),
            (
                "Avg. Total Cost ($)_mean",
                "Avg. Total Cost ($)_std",
                "{:.5f} ± {:.5f}",
                "Avg. Total Cost ($)",
            ),
            (
                "Avg. Total Time (s)_mean",
                "Avg. Total Time (s)_std",
                "{:.2f} ± {:.2f}",
                "Avg. Total Time (s)",
            ),
        ]

        summary = {"Benchmark": "Average", "Method": LABEL_MAP_HV.get(method, method)}
        for mean_col, std_col, fmt, display_key in cols_spec:
            m = 0.0
            s = 0.0
            if mean_col in dfm.columns:
                m_vals = pd.to_numeric(dfm[mean_col], errors="coerce").dropna()
                if not m_vals.empty:
                    m = float(m_vals.mean())
            if std_col in dfm.columns:
                s_vals = pd.to_numeric(dfm[std_col], errors="coerce").dropna()
                if not s_vals.empty:
                    s = float(s_vals.mean())

            # For cost per trial we store the raw dollars mean; display formatting will
            # scale to x10^-3 later in writers. Keep display string consistent here.
            summary[display_key] = fmt.format(m, s)

        # append summary row
        summary_rows.append(summary)

    # Write a single combined operational costs file (no MOHOLLM/LLM grouping)
    def write_md(rows_list, filename, title):
        df_out = pd.DataFrame(rows_list)
        # Remove the Method column from the output table per user request
        # if "Method" in df_out.columns:
        #     df_out = df_out.drop(columns=["Method"])

        # Avg. Requests: if numeric, round to integer for display; if already
        # a formatted string (mean ± std), leave as-is.
        if "Avg. Requests / Trial" in df_out.columns:
            col = df_out["Avg. Requests / Trial"]
            if pd.api.types.is_numeric_dtype(col):
                df_out["Avg. Requests / Trial"] = col.round(0).astype(int)

        # Scale Avg. Total Cost / Trial to units of 10^-3 and rename the column.
        # If the column is numeric, create the scaled numeric column. If it's
        # already a formatted string (mean ± std), assume it's already scaled and
        # rename the column to reflect x10^-3 units.
        if "Avg. Total Cost / Trial ($)" in df_out.columns:
            col = df_out["Avg. Total Cost / Trial ($)"]
            if pd.api.types.is_numeric_dtype(col):
                scaled = col.fillna(0.0) / 1e-3
                df_out["Avg. Total Cost / Trial ($ x10^-3)"] = scaled
                df_out = df_out.drop(columns=["Avg. Total Cost / Trial ($)"])
            else:
                # rename the column to indicate it's already in x10^-3
                df_out = df_out.rename(
                    columns={
                        "Avg. Total Cost / Trial ($)": "Avg. Total Cost / Trial ($ x10^-3)"
                    }
                )

        # Build display column order
        # Include Method column so the operational costs table shows which method each row refers to
        cols = [
            "Method",
            "Benchmark",
            "Avg. Time / Trial (s)",
            "Avg. Prompt Tokens / Trial",
            "Avg. Completion Tokens / Trial",
            "Avg. Total Tokens / Trial",
            "Avg. Requests / Trial",
        ]
        # Place the summed total tokens just before the per-trial cost column as requested
        if "Avg. Total Tokens (sum)" in df_out.columns:
            cols.append("Avg. Total Tokens (sum)")
        if "Avg. Total Cost / Trial ($ x10^-3)" in df_out.columns:
            cols.append("Avg. Total Cost / Trial ($ x10^-3)")
        if "Avg. Total Cost ($)" in df_out.columns:
            cols.append("Avg. Total Cost ($)")
        if "Avg. Total Time (s)" in df_out.columns:
            cols.append("Avg. Total Time (s)")

        df_display = df_out[cols]

        out_path = os.path.join(path, filename)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            f.write(f"# {title}\n\n")
            if not df_display.empty:
                # The table now contains pre-formatted 'mean ± std' strings for
                # these columns, so render without floatfmt.
                f.write(df_display.to_markdown(index=False))
            else:
                f.write("No data available\n")
        print(f"Saved {out_path}")

    write_md(rows, "operational_costs.md", "Operational Costs - All Methods")

    # Additionally write per-metric files into per-method subfolders
    def write_metric_files(rows_list, method_key):
        # rows_list: list of dict rows
        base_dir = os.path.join(path, method_key)
        os.makedirs(base_dir, exist_ok=True)
        df = pd.DataFrame(rows_list)
        if df.empty:
            return
        # # drop Method column for the per-metric outputs as well
        # if "Method" in df.columns:
        #     df = df.drop(columns=["Method"])

        # Ensure cost per trial column is scaled and renamed if present. If the
        # column contains formatted strings (mean ± std) assume it's already
        # scaled and just rename it to the x10^-3 header.
        if "Avg. Total Cost / Trial ($)" in df.columns:
            col = df["Avg. Total Cost / Trial ($)"]
            if pd.api.types.is_numeric_dtype(col):
                df["Avg. Total Cost / Trial ($ x10^-3)"] = col.fillna(0.0) / 1e-3
                df = df.drop(columns=["Avg. Total Cost / Trial ($)"])
            else:
                df = df.rename(
                    columns={
                        "Avg. Total Cost / Trial ($)": "Avg. Total Cost / Trial ($ x10^-3)"
                    }
                )

        # Requests formatting (integer) if numeric; leave formatted strings untouched
        if "Avg. Requests / Trial" in df.columns:
            col = df["Avg. Requests / Trial"]
            if pd.api.types.is_numeric_dtype(col):
                df["Avg. Requests / Trial"] = col.round(0).astype(int)

        # Money file: include per-trial scaled and total cost
        money_cols = [
            c
            for c in [
                "Benchmark",
                "Avg. Total Cost / Trial ($ x10^-3)",
                "Avg. Total Cost ($)",
            ]
            if c in df.columns
        ]
        if money_cols:
            out = df[money_cols]
            out_path = os.path.join(base_dir, "money.md")
            with open(out_path, "w") as f:
                f.write("# Money - Operational Costs\n\n")
                f.write(out.to_markdown(index=False))

        # Tokens file
        token_cols = [
            c
            for c in [
                "Benchmark",
                "Avg. Prompt Tokens / Trial",
                "Avg. Completion Tokens / Trial",
                "Avg. Total Tokens / Trial",
                "Avg. Total Tokens (sum)",
            ]
            if c in df.columns
        ]
        if token_cols:
            out = df[token_cols]
            out_path = os.path.join(base_dir, "tokens.md")
            with open(out_path, "w") as f:
                f.write("# Token Usage\n\n")
                f.write(out.to_markdown(index=False))

        # Time file - include requests column here as requested
        time_cols = [
            c
            for c in [
                "Benchmark",
                "Avg. Time / Trial (s)",
                "Avg. Requests / Trial",
                "Avg. Total Time (s)",
            ]
            if c in df.columns
        ]
        if time_cols:
            out = df[time_cols]
            out_path = os.path.join(base_dir, "time.md")
            with open(out_path, "w") as f:
                f.write("# Time Usage\n\n")
                f.write(out.to_markdown(index=False))

        # Requests file
        # req_cols = [
        #     c for c in ["Benchmark", "Avg. Requests / Trial"] if c in df.columns
        # ]
        # if req_cols:
        #     out = df[req_cols]
        #     out_path = os.path.join(base_dir, "requests.md")
        #     with open(out_path, "w") as f:
        #         f.write("# Requests\n\n")
        #         f.write(out.to_markdown(index=False, floatfmt=(".0f",)))

    # write per-method metric files
    for method, method_rows_list in method_rows.items():
        write_metric_files(method_rows_list, method)

    # Write a simple method averages file (one row per method) using the
    # precomputed `summary_rows` which already contains formatted display strings.
    df_summary = pd.DataFrame(summary_rows)
    avg_path = os.path.join(path, "method_averages.md")
    os.makedirs(os.path.dirname(avg_path), exist_ok=True)
    with open(avg_path, "w") as f:
        f.write("# Average Values per Method (mean ± std)\n\n")
        if not df_summary.empty:
            f.write(df_summary.to_markdown(index=False))
        else:
            f.write("No data available")
    print(f"Saved {avg_path}")


# --- END OF MODIFIED TABLE FUNCTION ---


# --- (extract_method_name and group_data_by_method are unchanged) ---
def extract_method_name(files):
    methods = []
    for file in files:
        method = file.split(os.sep)[-3]
        if method not in methods:
            methods.append(method)
    return methods


def group_data_by_method(observed_fvals_files, filter):
    method_names = extract_method_name(observed_fvals_files)
    method_files = {}
    for method in method_names:
        files = []
        for file in observed_fvals_files:
            if method in file.split(os.sep) and (filter in file if filter else True):
                files.append(file)
        method_files[method] = files
    return method_files


if __name__ == "__main__":
    main()
