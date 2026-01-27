import argparse
import glob
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plot_settings import COLORS, LABEL_MAP_HV, MARKERS
from eaf import EmpiricalAttainmentFuncPlot, get_empirical_attainment_surface

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


def custom_sort_key(method_name):
    if "MOHOLLM" in method_name:
        return (2, method_name)
    if "LLM" == method_name or "LLM" in LABEL_MAP_HV.get(method_name, method_name):
        return (1, method_name)
    return (0, method_name)


def create_eaf_subplot(ax, data, benchmark, title, columns, x_lim, y_lim):
    methods = sorted(data.keys(), key=custom_sort_key)
    eaf = EmpiricalAttainmentFuncPlot()
    handles = []
    labels = []
    for i, method in enumerate(methods):
        dfs = data[method]
        if not dfs:
            continue
        arr = np.array([df[columns].to_numpy() for df in dfs])
        levels = [1, 2, 3]
        surfs = get_empirical_attainment_surface(costs=arr, levels=levels)
        plot_elem = eaf.plot_surface_with_band(
            ax,
            color=COLORS.get(method, f"C{i}"),
            label=LABEL_MAP_HV.get(method, method),
            surfs=surfs,
        )
        handles.append(plot_elem)
        labels.append(LABEL_MAP_HV.get(method, method))
    ax.set_xlabel(columns[0])
    ax.set_ylabel(columns[1])
    ax.set_title(title, fontweight="bold")
    if x_lim and all(x != "" for x in x_lim):
        ax.set_xlim(float(x_lim[0]), float(x_lim[1]))
    if y_lim and all(y != "" for y in y_lim):
        ax.set_ylim(float(y_lim[0]), float(y_lim[1]))
    ax.grid(True, linestyle="--", alpha=0.5, linewidth=0.4)
    return handles, labels


def main():
    parser = argparse.ArgumentParser(
        description="Plot empirical attainment for multiple benchmarks as subplots."
    )
    parser.add_argument("--benchmarks", nargs="+", type=str, required=True)
    parser.add_argument("--titles", nargs="+", type=str, required=True)
    parser.add_argument("--data_paths", nargs="+", type=str, required=True)
    parser.add_argument("--columns", nargs="+", type=str, required=True)
    parser.add_argument("--filters", nargs="+", type=str, default=None)
    parser.add_argument("--trials", type=int, required=True)
    parser.add_argument("--blacklists", nargs="+", type=str, default=None)
    parser.add_argument("--filename", type=str, default=None)
    parser.add_argument("--x_lims", nargs="+", type=str, default=None)
    parser.add_argument("--y_lims", nargs="+", type=str, default=None)
    parser.add_argument("--whitelists", nargs="+", type=str, default=None)
    args = parser.parse_args()

    n = len(args.benchmarks)
    ncols = 3
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4 * ncols, 4 * nrows), sharex=False
    )
    axes = axes.flatten() if n > 1 else [axes]

    all_handles = None
    all_labels = None

    for idx in range(n):
        benchmark = args.benchmarks[idx]
        title = args.titles[idx]
        data_path = args.data_paths[idx]
        columns = args.columns[idx].split(",")
        filter = args.filters[idx] if args.filters else None
        trials = args.trials
        blacklist = args.blacklists[idx].split(",") if args.blacklists else []
        whitelist = args.whitelists[idx].split(",") if args.whitelists else []
        x_lim = args.x_lims[idx * 2 : idx * 2 + 2] if args.x_lims else None
        y_lim = args.y_lims[idx * 2 : idx * 2 + 2] if args.y_lims else None

        file_names = glob.glob(f"{data_path}/**/*.csv", recursive=True)
        observed_fvals_files = [
            file for file in file_names if "observed_fvals" in file.split("/")
        ]
        method_files = group_data_by_method(observed_fvals_files, filter)
        method_dfs = {
            method: [pd.read_csv(file, usecols=columns)[:trials] for file in files]
            for method, files in method_files.items()
        }
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
        handles, labels = create_eaf_subplot(
            axes[idx], method_dfs, benchmark, title, columns, x_lim, y_lim
        )
        if all_handles is None:
            all_handles = handles
            all_labels = labels

    for ax in axes[n:]:
        ax.axis("off")

    if len(all_labels) <= 3:
        ncols = len(all_labels)
    else:
        ncols = len(all_labels) / 3

    fig.subplots_adjust(bottom=0.3, wspace=0.33)
    bbox_to_anchor = (
        0.5,
        -0.1,
    )  # This has to be adjusted for more plot -0.03 for synthetic benchmarks and -0.1 for real world
    fig.legend(
        all_handles,
        all_labels,
        loc="outside lower center",
        ncol=len(all_labels),
        bbox_to_anchor=bbox_to_anchor,
        frameon=True,
        framealpha=0.95,
        edgecolor="k",
        fancybox=False,
        columnspacing=1.0,
        handletextpad=0.5,
    )
    plt.tight_layout()
    for file_type in ["svg", "pdf", "png"]:
        out_dir = f"./plots/pareto_set_empirical_attainment_subplots/{file_type}/"
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(
            f"{out_dir}/{args.filename}.{file_type}", dpi=300, bbox_inches="tight"
        )
    plt.close()


if __name__ == "__main__":
    main()
