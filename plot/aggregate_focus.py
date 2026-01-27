import os
import glob
import argparse
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import entropy
from sklearn.neighbors import KernelDensity
from plot_settings import get_color, LABEL_MAP_HV

DEFAULT_FIGSIZE = (8, 4)
FONT_SIZE = 11
AX_LABELSIZE = 12
TITLE_SIZE = 12
TICK_LABELSIZE = 10
LEGEND_FONTSIZE = 9
USE_LATEX = True

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


def calculate_focus_score_continuous_kde(series, n_samples=1000, n_bins=50):
    if series.empty or series.isnull().all():
        return np.nan
    if series.std() < 1e-9:
        return 1.0
    data = series.to_numpy().reshape(-1, 1)
    kde = KernelDensity(kernel="gaussian", bandwidth="scott").fit(data)
    samples = kde.sample(n_samples=n_samples)
    counts, _ = np.histogram(samples, bins=n_bins, density=False)
    if len(np.unique(counts)) <= 1 and np.sum(counts) > 0:
        return 1.0
    probs = counts / n_samples
    ent = entropy(probs, base=2)
    max_ent = np.log2(n_bins)
    if max_ent <= 0:
        return 1.0
    normalized_entropy = ent / max_ent
    return 1.0 - normalized_entropy


def calculate_focus_score(series, categorical):
    if series.empty:
        return np.nan
    if categorical:
        counts = series.value_counts()
        if len(counts) <= 1:
            return 1.0
        probs = counts / counts.sum()
        ent = entropy(probs, base=2)
        max_ent = np.log2(len(counts))
        normalized_entropy = ent / max_ent if max_ent > 0 else 0
        return 1.0 - normalized_entropy
    else:
        return calculate_focus_score_continuous_kde(series)


def compute_aggregated_focus_trajectory(df, categorical=False):
    """
    For a single run (DataFrame), compute the per-iteration aggregated focus score:
    - For each row, parse llm_candidate_proposal (list of dicts)
    - Compute focus score per feature, then average across features to get one value per iteration
    Returns list of floats (may contain np.nan).
    """
    traj = []
    for _, row in df.iterrows():
        proposals = parse_string_to_list(row.get("llm_candidate_proposal", "[]"))
        if not proposals:
            traj.append(np.nan)
            continue
        props_df = pd.DataFrame(proposals)
        feature_scores = []
        for feature in props_df.columns:
            s = props_df[feature].dropna()
            if s.empty:
                continue
            score = calculate_focus_score(s, categorical)
            if not np.isnan(score):
                feature_scores.append(score)
        if feature_scores:
            traj.append(float(np.nanmean(feature_scores)))
        else:
            traj.append(np.nan)
    return traj


def group_runs_by_method(files, filter_str=None):
    methods = {}
    for f in files:
        parts = f.split(os.sep)
        if len(parts) < 3:
            continue
        method = parts[-3]
        if filter_str and filter_str not in f:
            continue
        methods.setdefault(method, []).append(f)
    return methods


def load_trajectories_for_method(file_list, trials=None, categorical=False):
    runs = []
    for f in file_list:
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        if trials is not None:
            df = df[:trials]
        y = compute_aggregated_focus_trajectory(df, categorical=categorical)
        if y and any([not np.isnan(v) for v in y]):
            runs.append(np.array(y, dtype=float))
    return runs


def align_and_compute_stats(runs):
    """Align by iteration index (forward-fill + backfill) and compute mean & 95% CI."""
    if not runs:
        return np.array([]), np.array([]), np.array([])
    max_len = max(len(r) for r in runs)
    aligned = []
    for r in runs:
        if len(r) < max_len:
            s = pd.Series(r, index=np.arange(len(r)))
            s = s.reindex(np.arange(max_len), method="ffill").bfill()
            aligned.append(s.values)
        else:
            aligned.append(r)
    arr = np.vstack(aligned)
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0)
    n = arr.shape[0]
    stderr = 1.96 * std / np.sqrt(max(1, n))
    x = np.arange(max_len)
    return x, mean, stderr


def plot_aggregated_focus(methods_stats, title, outdir, filename):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    methods = sorted(methods_stats.keys())
    for i, method in enumerate(methods):
        x, mean, stderr = methods_stats[method]
        if x.size == 0:
            continue
        if method == "MOHOLLM":
            color = get_color("MOHOLLM (Gemini 2.0 Flash) (Context)", i)
            label = LABEL_MAP_HV.get("MOHOLLM (Gemini 2.0 Flash) (Context)", method)
        if method == "LLM":
            color = get_color("mohollm (Gemini 2.0 Flash)", i)
            label = LABEL_MAP_HV.get("mohollm (Gemini 2.0 Flash)", method)
        ax.plot(x, mean, label=label, color=color, linewidth=1.8)
        ax.fill_between(x, mean - stderr, mean + stderr, color=color, alpha=0.2)
    ax.set_xlabel("Trials")
    ax.set_ylabel("Aggregated Focus Score")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        ncol=2,
        frameon=True,
        framealpha=0.95,
        edgecolor="k",
        fancybox=False,
    )
    plt.tight_layout(rect=[0, 0.12, 1, 1])
    for ext in ("png", "svg", "pdf"):
        outpath = os.path.join(outdir, f"{filename}.{ext}")
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()


def classify_method(method):
    """Return 'MOHOLLM' for the two exact MOHOLLM names, 'LLM' for the exact mohollm name, else None."""
    mohollm_names = {
        "MOHOLLM (Gemini 2.0 Flash) (Context)",
        "MOHOLLM (Gemini 2.0 Flash)",
    }
    llm_names = {"mohollm (Gemini 2.0 Flash)"}
    if method in mohollm_names:
        return "MOHOLLM"
    if method in llm_names:
        return "LLM"
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate Focus Score across benchmarks and seeds"
    )
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to results folder"
    )
    parser.add_argument(
        "--benchmarks", type=str, required=True, help="Comma-separated benchmarks"
    )
    parser.add_argument("--trials", type=int, help="Number of trials to consider")
    parser.add_argument("--filter", type=str, help="Optional file substring filter")
    parser.add_argument(
        "--whitelist",
        type=str,
        default="",
        help="Comma-separated list of methods to include",
    )
    parser.add_argument(
        "--categorical",
        action="store_true",
        help="Treat proposal features as categorical",
    )
    parser.add_argument(
        "--title", type=str, default="Aggregated Focus Score", help="Plot title"
    )
    parser.add_argument(
        "--filename", type=str, default="aggregate_focus", help="Output filename prefix"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./plots/feature_focus/aggregate",
        help="Output directory",
    )
    args = parser.parse_args()

    benchmarks = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    all_files = []
    for b in benchmarks:
        search = os.path.join(args.data_path, b, "**", "*.csv")
        found = glob.glob(search, recursive=True)
        found = [f for f in found if "icl_llm_proposal_trajectory" in f]
        all_files.extend(found)
    if args.filter:
        all_files = [f for f in all_files if args.filter in f]

    method_files = group_runs_by_method(all_files)

    # apply whitelist if provided
    if args.whitelist:
        whitelist = [w.strip() for w in args.whitelist.split(",") if w.strip()]
        if whitelist:
            method_files = {
                m: files for m, files in method_files.items() if m in whitelist
            }

    method_runs = {}
    for m, files in method_files.items():
        runs = load_trajectories_for_method(
            files, trials=args.trials, categorical=args.categorical
        )
        if runs:
            method_runs[m] = runs

    # Filter and aggregate into strict groups (MOHOLLM vs LLM)
    filtered = {}
    for m, runs in method_runs.items():
        grp = classify_method(m)
        if grp is not None:
            filtered.setdefault(grp, []).extend(runs)

    groups_stats = {}
    for grp, runs in filtered.items():
        x, mean, stderr = align_and_compute_stats(runs)
        groups_stats[grp] = (x, mean, stderr)

    if not groups_stats:
        print("No valid MOHOLLM/LLM runs found. Exiting.")
        return

    plot_aggregated_focus(groups_stats, args.title, args.output_path, args.filename)
    print(
        f"Aggregated focus plot saved to {args.output_path}/{args.filename}.[png|svg|pdf]"
    )


if __name__ == "__main__":
    main()
