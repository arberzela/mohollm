import argparse
import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


plt.rcParams.update(
    {
        # Use matplotlib's mathtext renderer (no external LaTeX required)
        "text.usetex": True,
        "mathtext.fontset": "cm",
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "legend.fontsize": 10,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    }
)


def group_data_by_method(observed_fvals_files, filter):
    """
    Groups observed_fvals CSV files by the method name inferred from their path.
    This mirrors the helper from plot_fvals_paper so calling semantics match.
    """

    def extract_method_name(files):
        methods = []
        for file in files:
            method = file.split("/")[-3]
            if method not in methods:
                methods.append(method)
        return methods

    method_names = extract_method_name(observed_fvals_files)
    method_files = {}
    for method in method_names:
        files = []
        for file in observed_fvals_files:
            if method in file.split("/") and (filter in file if filter else True):
                files.append(file)
        method_files[method] = files
    return method_files


def normalize_data(fvals, min_max_metrics):
    for column, values in min_max_metrics.items():
        min_val = values["min"]
        max_val = values["max"]
        fvals[column] = (fvals[column] - min_val) / (max_val - min_val)
    return fvals


def main():
    parser = argparse.ArgumentParser(
        description="Bar plot comparing methods using averaged final fvals"
    )
    parser.add_argument("--benchmark", type=str, required=True, help="Benchmark name")
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to observed fvals folder"
    )
    parser.add_argument(
        "--filter", type=str, default=None, help="Filter substring for files to include"
    )
    parser.add_argument(
        "--columns",
        type=str,
        default=None,
        help="Columns to read from the CSV files (comma separated)",
    )
    parser.add_argument(
        "--trials", type=int, default=None, help="Number of trials to consider"
    )
    parser.add_argument(
        "--blacklist", default="", type=str, help="Models or methods not to plot"
    )
    parser.add_argument(
        "--whitelist",
        default="",
        type=str,
        help="Only plot models or methods in this list",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="bar_plot",
        help="Output filename (without extension)",
    )

    parser.add_argument("--title", type=str, default="Model comparison")
    parser.add_argument("--ylabel", type=str, default="Function value")

    args = parser.parse_args()

    data_path = args.data_path
    benchmark = args.benchmark
    filter = args.filter
    trials = args.trials
    filename = args.filename
    columns = args.columns.split(",") if args.columns else None
    blacklist = args.blacklist.split(",")
    whitelist = args.whitelist.split(",")

    print(f"Plotting bar comparison for benchmark={benchmark} from {data_path}")

    # Gather all CSV files from folder
    file_names = glob.glob(f"{data_path}/**/*.csv", recursive=True)
    observed_fvals_files = [
        file for file in file_names if "observed_fvals" in file.split("/")
    ]

    method_files = group_data_by_method(observed_fvals_files, filter)

    # Read dataframes grouped by method
    method_dfs = {
        method: [pd.read_csv(file, usecols=columns)[:trials] for file in files]
        for method, files in method_files.items()
    }

    # Apply whitelist/blacklist
    if len(whitelist) > 0 and whitelist[0] != "":
        filtered_dfs = {}
        for method, dfs in method_dfs.items():
            if any(entry == method for entry in whitelist):
                filtered_dfs[method] = dfs
            else:
                print(f"Filtered out method: {method} due to not being in whitelist")
        method_dfs = filtered_dfs
    elif len(blacklist) > 0 and blacklist[0] != "":
        filtered_dfs = {}
        for method, dfs in method_dfs.items():
            if not any(entry == method for entry in blacklist):
                filtered_dfs[method] = dfs
            else:
                print(f"Filtered out method: {method} due to blacklist")
        method_dfs = filtered_dfs

    if len(method_dfs) == 0:
        raise ValueError(
            "No methods found after applying whitelist/blacklist and filters"
        )

    # For each method, compute per-run scalar = all-time minimum of F1
    method_scalars = {}
    for method, dfs in sorted(method_dfs.items(), key=lambda x: x[0]):
        run_scalars = []
        for df in dfs:
            # If trials is provided, consider only first `trials` rows when computing the all-time min
            if trials is not None and len(df) >= trials:
                df_considered = df.iloc[:trials]
            else:
                df_considered = df

            # The F1 column is expected to always be present; use its minimum over time
            if "F1" not in df_considered.columns:
                raise ValueError(
                    f"F1 column not found for method {method} in one of the files"
                )

            f1_series = pd.to_numeric(df_considered["F1"], errors="coerce")
            if f1_series.dropna().empty:
                raise ValueError(
                    f"No numeric values in F1 column for method {method} in one of the files"
                )

            # Compute the all-time minimum of F1 for this run
            scalar = float(f1_series.min())
            run_scalars.append(scalar)

        method_scalars[method] = run_scalars

    # Group by base LLM and category (LLM vs LLMKD)
    grouped = {}
    for method, scalars in method_scalars.items():
        if method in [
            "LLMKD",
            "LLMKD-llama3.1-8b",
            "LLMKD-llama3.3-70b",
        ]:
            is_kd = True
        else:
            is_kd = False

        base = ""
        if method == "LLMKD-llama3.3-70b" or method == "LLM-llama3.3-70b-llm":
            base = "Llama-3.3-70B"
        if method == "LLMKD-llama3.1-8b" or method == "LLM-llama3.1-8b-llm":
            base = "Llama-3.1-8B"
        if method == "LLM" or method == "LLMKD":
            base = "Gemini-2.0-Flash"

        print(base)
        if base == "":
            base = method

        if base not in grouped:
            grouped[base] = {"LLM": [], "LLMKD": []}

        if is_kd:
            grouped[base]["LLMKD"].extend(scalars)
        else:
            grouped[base]["LLM"].extend(scalars)

    # mean_grouped = {}
    # for base, vals in grouped.items():
    #     llm_vals = np.array(vals["LLM"])
    #     kd_vals = np.array(vals["LLMKD"])
    #     mean_grouped[base] = {
    #         "LLM": np.nan if llm_vals.size == 0 else float(np.mean(llm_vals)),
    #         "LLMKD": np.nan if kd_vals.size == 0 else float(np.mean(kd_vals)),
    #     }
    # # Create a DataFrame where rows are ['LLM','LLMKD'] and columns are bases,
    # # then save the mean values to CSV.
    # data = pd.DataFrame(mean_grouped)
    # data.to_csv("data_bar_comparison_plot_means.csv")
    # print("Saved mean table:\n", data)

    # Prepare plotting arrays: for each base LLM, two bars (LLM, LLMKD)
    bases = sorted(grouped.keys())
    llm_means = []
    llm_stds = []
    kd_means = []
    kd_stds = []
    for i, base in enumerate(bases):
        llm_vals = np.array(grouped[base]["LLM"])
        kd_vals = np.array(grouped[base]["LLMKD"])

        llm_means.append(np.nan if llm_vals.size == 0 else float(np.mean(llm_vals)))
        llm_stds.append(0.0 if llm_vals.size == 0 else float(np.std(llm_vals)))

        kd_means.append(np.nan if kd_vals.size == 0 else float(np.mean(kd_vals)))
        kd_stds.append(0.0 if kd_vals.size == 0 else float(np.std(kd_vals)))

    # Plotting grouped bars: for each base LLM show two bars (LLM, LLMKD)
    x = np.arange(len(bases))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6, 4.5))

    llm_means_arr = np.array(llm_means, dtype=float)
    llm_stds_arr = np.array(llm_stds, dtype=float)
    kd_means_arr = np.array(kd_means, dtype=float)
    kd_stds_arr = np.array(kd_stds, dtype=float)

    llm_present = ~np.isnan(llm_means_arr)
    kd_present = ~np.isnan(kd_means_arr)

    # Use consistent colors: green for LLM, red for LLMKD
    llm_color = "tab:green"
    kd_color = "tab:red"

    # Plot LLM bars (left)
    for i in range(len(bases)):
        if llm_present[i]:
            ax.bar(
                x[i] - width / 2,
                llm_means_arr[i],
                width,
                yerr=llm_stds_arr[i],
                capsize=5,
                color=llm_color,
                alpha=0.9,
                label="LLM" if i == 0 else "",
            )
        else:
            ax.bar(x[i] - width / 2, 0, width, color=llm_color, alpha=0.0)

    # Plot LLMKD bars (right)
    for i in range(len(bases)):
        if kd_present[i]:
            ax.bar(
                x[i] + width / 2,
                kd_means_arr[i],
                width,
                yerr=kd_stds_arr[i],
                capsize=5,
                color=kd_color,
                alpha=0.9,
                label="LLMKD" if i == 0 else "",
            )
        else:
            ax.bar(x[i] + width / 2, 0, width, color=kd_color, alpha=0.0)

    ax.set_xticks(x)
    ax.set_xticklabels(bases, rotation=25, ha="right")
    ax.set_ylabel(args.ylabel)
    ax.set_ylim(1600, 1680)
    ax.grid(True, alpha=0.3, linewidth=0.4)
    # ax.set_title(args.title)
    ax.legend()

    plt.tight_layout()

    # Save in multiple formats to the plots folder
    for ext in ["png", "svg", "pdf"]:
        out_dir = f"./plots/bar/{ext}"
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{filename}.{ext}")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
