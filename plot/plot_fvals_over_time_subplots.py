import argparse
import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 18,
        "axes.labelsize": 18,
        "axes.titlesize": 24,
        "legend.fontsize": 20,
        "xtick.labelsize": 22,
        "ytick.labelsize": 22,
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
    "LLMKD-exploitation": "HOLLM (exploitation)",
    "LLMKD-exploration": "HOLLM (exploration)",
    "LLMKD-ucb1": "HOLLM (UCB1)",
    "LLMKD-uniform_region_sampling": "HOLLM (uniform)",
    "LLMKD-qwen-30b": "HOLLM (Qwen3-30B-A3B)",
    "LLM-qwen-30b": "LLM (Qwen3-30B-A3B)",
}


def normalize_data(fvals, min_max_metrics):
    for column, values in min_max_metrics.items():
        min_val = values["min"]
        max_val = values["max"]
        fvals[column] = (fvals[column] - min_val) / (max_val - min_val)
    return fvals


def extract_method_name(files):
    methods = []
    for file in files:
        method = file.split("/")[-3]
        if method not in methods:
            methods.append(method)
    return methods


def group_data_by_method(observed_fvals_files, filter):
    method_names = extract_method_name(observed_fvals_files)
    method_files = {}
    for method in method_names:
        files = []
        for file in observed_fvals_files:
            if method in file.split("/") and (filter in file if filter else True):
                files.append(file)
        method_files[method] = files
    return method_files


def create_fval_subplot(ax, data, title, x_lim, y_lim, first, use_log_scale=False):
    """Draw fvals-over-time on the provided Axes and return handles/labels for legend."""

    def custom_sort_key(method_name):
        """
        Custom key for sorting methods in the legend.
        Returns a tuple (priority, label) where:
        - Priority 0 is for standard methods (sorted alphabetically).
        - Priority 1 is for LLM.
        - Priority 2 is for HOLLM (making it last).
        """
        label = LABEL.get(method_name, method_name)
        if label.startswith(
            "HOLLM"
        ):  # Use startswith to catch variations like HOLLM (Llama-3.1-8B)
            return (2, label)
        if label.startswith("LLM"):
            return (1, label)
        return (0, label)

    methods = sorted(data["mean"].keys(), key=custom_sort_key)

    # methods = sorted(data["mean"].keys())
    handles = []
    labels = []

    for i, method in enumerate(methods):
        mean_values = data["mean"][method]
        std_values = data["std"][method]
        # Follow same formula as plot_fvals_paper: std scaled by num_seeds
        std_error = 1.96 * std_values / data.get("num_seeds", 1)
        trials = range(len(mean_values))

        label = LABEL.get(method, method)
        color = COLORS.get(method)
        marker = MARKERS.get(method, None)
        linestyle = LSTYLE.get(method, "solid")

        (line,) = ax.plot(
            trials,
            -1 * np.array(mean_values),
            label=label,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=2,
            markevery=5,
        )
        ax.fill_between(
            trials,
            -1 * (np.array(mean_values) + np.array(std_error)),
            -1 * (np.array(mean_values) - np.array(std_error)),
            color=color,
            alpha=0.2,
        )
        handles.append(line)
        labels.append(label)

    ax.set_xlabel("Number of evaluations", fontsize=30)
    ax.set_xticks([0, 10, 20, 30, 40, 50])
    if first:
        ax.set_ylabel("Function value", fontsize=30)
    if use_log_scale:
        ax.set_yscale("log")
    ax.set_title(title, fontweight="bold", fontsize=32)
    if x_lim and all(x != "" for x in x_lim):
        ax.set_xlim(float(x_lim[0]), float(x_lim[1]))
    if y_lim and all(y != "" for y in y_lim):
        ax.set_ylim(float(y_lim[0]), float(y_lim[1]))
    ax.grid(True, linestyle="--", alpha=0.5, linewidth=0.4)
    return handles, labels


def main():
    parser = argparse.ArgumentParser(
        description="Plot function values over time for multiple benchmarks as subplots."
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        type=str,
        required=True,
        help="List of benchmark names",
    )
    parser.add_argument(
        "--titles", nargs="+", type=str, required=True, help="Titles for each subplot"
    )
    parser.add_argument(
        "--data_paths",
        nargs="+",
        type=str,
        required=True,
        help="Paths to fvals folders for each benchmark",
    )
    parser.add_argument(
        "--filters", nargs="+", type=str, default=None, help="Filter for each benchmark"
    )
    parser.add_argument(
        "--trials", type=int, required=True, help="Number of trials to consider"
    )
    parser.add_argument(
        "--num_seeds", type=int, default=None, help="Number of seeds to consider"
    )
    parser.add_argument(
        "--blacklists",
        nargs="+",
        type=str,
        default=None,
        help="Blacklist for each benchmark",
    )
    parser.add_argument(
        "--filename", type=str, default=None, help="Filename to store the plot"
    )
    parser.add_argument(
        "--x_lims",
        nargs="+",
        type=str,
        default=None,
        help="X-axis limits for each subplot (as pairs)",
    )
    parser.add_argument(
        "--y_lims",
        nargs="+",
        type=str,
        default=None,
        help="Y-axis limits for each subplot (as pairs)",
    )
    parser.add_argument(
        "--whitelists",
        nargs="+",
        type=str,
        default=None,
        help="Whitelist for each benchmark",
    )
    parser.add_argument(
        "--use_log_scale", default=False, type=bool, help="Use log scale for the y-axis"
    )

    args = parser.parse_args()

    n = len(args.benchmarks)
    ncols = n
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(6 * ncols, 4.5 * nrows), sharex=False
    )
    axes = axes.flatten() if n > 1 else [axes]

    all_handles = None
    all_labels = None
    first = True
    for idx in range(n):
        benchmark = args.benchmarks[idx]
        title = args.titles[idx]
        data_path = args.data_paths[idx]
        filter = args.filters[idx] if args.filters else None
        trials = args.trials
        blacklist = args.blacklists[idx].split(",") if args.blacklists else []
        whitelist = args.whitelists[idx].split(",") if args.whitelists else []
        x_lim = args.x_lims if args.x_lims else None
        y_lim = args.y_lims if args.y_lims else None

        file_names = glob.glob(f"{data_path}/**/*.csv", recursive=True)
        observed_fvals_files = [
            file for file in file_names if "observed_fvals" in file.split("/")
        ]
        method_files = group_data_by_method(observed_fvals_files, filter)

        method_dfs = {
            method: [pd.read_csv(file, usecols=["F1"])[:trials] for file in files]
            for method, files in method_files.items()
        }

        # Apply whitelist/blacklist filtering
        if len(whitelist) > 0 and whitelist[0] != "":
            filtered_dfs = {}
            for method, dfs in method_dfs.items():
                if any(entry == method for entry in whitelist):
                    filtered_dfs[method] = dfs
            method_dfs = filtered_dfs
        elif len(blacklist) > 0 and blacklist[0] != "":
            filtered_dfs = {}
            for method, dfs in method_dfs.items():
                if not any(entry in method for entry in blacklist):
                    filtered_dfs[method] = dfs
            method_dfs = filtered_dfs

        if not method_dfs:
            print(f"No methods found for benchmark {benchmark}. Skipping.")
            continue

        # Compute per-method fvals over time
        method_fvals = {}
        for method, dfs in method_dfs.items():
            method_fval = []
            for df in dfs:
                method_fval.append(np.minimum.accumulate(df[:trials].values).flatten())
                print(np.minimum.accumulate(df[:trials].values).flatten())
            print(method)
            method_fvals[method] = np.array(method_fval)

        mean_fvals = {
            method: np.mean(fvals, axis=0) for method, fvals in method_fvals.items()
        }
        std_fvals = {
            method: np.std(fvals, axis=0) for method, fvals in method_fvals.items()
        }

        data = {
            "mean": mean_fvals,
            "std": std_fvals,
            "num_seeds": len(method_fvals),
        }

        y_lim_to_use = y_lim  # Default to user-provided limit
        if y_lim_to_use is None and method_fvals:
            # 1. Define a "burn-in" period to ignore early, volatile values.
            #    We'll focus on the final 10% of the evaluations.
            burn_in_period = max(1, int(0.99 * trials))

            # 2. Find the min/max function values across all methods after the burn-in.
            all_final_values = []
            for fvals_per_seed in method_fvals.values():
                # The plot uses -1 * fvals, so we must do the same for limit calculation
                relevant_values = -1 * fvals_per_seed[:, burn_in_period:]
                all_final_values.extend(relevant_values.flatten())

            if all_final_values:
                # 3. Use percentiles for robust limits to avoid outliers skewing the view.
                y_min_robust = np.quantile(all_final_values, 0.03)  # 3rd percentile
                y_max_robust = np.quantile(all_final_values, 0.90)  # 90th percentile

                # 4. Add a small padding so plot lines don't touch the axes.
                padding = (y_max_robust - y_min_robust) * 0.05  # 5% padding
                final_y_min = y_min_robust - padding
                final_y_max = y_max_robust + padding

                y_lim_to_use = [final_y_min, final_y_max]

        if benchmark in ["rastrigin", "levy"]:
            print("please")

        if benchmark == "rastrigin":
            y_lim_to_use = [-150, -25]
        if benchmark == "levy":
            y_lim_to_use = [-75, -10]

        handles, labels = create_fval_subplot(
            axes[idx],
            data,
            title,
            x_lim,
            y_lim_to_use,
            first,
            use_log_scale=args.use_log_scale,
        )
        if all_handles is None:
            all_handles = handles
            all_labels = labels
        first = False  # Disables the y-label for the the other plots

    # Hide unused axes
    for ax in axes[n:]:
        ax.set_visible(False)

    fig.subplots_adjust(bottom=0.3, wspace=0.33)

    fig.legend(
        all_handles,
        all_labels,
        loc="outside lower center",
        ncol=8,
        bbox_to_anchor=(0.5, -0.2),
        frameon=True,
        framealpha=0.95,
        edgecolor="k",
        fancybox=False,
        columnspacing=1.0,
        handletextpad=0.5,
        fontsize=30,
    )

    plt.tight_layout()
    for file_type in ["svg", "pdf", "png"]:
        dir_path = (
            f"./plots/fvals_over_time_subplots/{file_type}/{args.filename}.{file_type}"
        )
        os.makedirs(os.path.dirname(dir_path), exist_ok=True)
        plt.savefig(dir_path, dpi=300, bbox_inches="tight")

    plt.close()


if __name__ == "__main__":
    main()
