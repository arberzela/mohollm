#!/usr/bin/env python3

import os
import csv
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Validate results folder for minimum trials and file count."
    )
    parser.add_argument(
        "--min_trials",
        type=int,
        default=50,
        help="Minimum number of trials (rows) per CSV file.",
    )
    parser.add_argument(
        "--min_files",
        type=int,
        default=5,
        help="Minimum number of CSV files per method.",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "results"),
        help="Path to results directory.",
    )
    parser.add_argument(
        "--check_all",
        action="store_true",
        help="Check all methods, not just the whitelist.",
    )
    args = parser.parse_args()

    whitelist = [
        "CTAEA",
        "EHVI",
        "GDE3",
        "IBEA",
        "MOEAD",
        "MOHOLLM (Gemini 2.0 Flash)",
        "MOHOLLM (Gemini 2.0 Flash) (Context)",
        "mohollm (Gemini 2.0 Flash)",
        "NSGA2",
        "NSGA3",
        "PESA2",
        "qLogEHVI",
        "RNSGA2",
        "SMSEMOA",
        "SPEA2",
        "UNSGA3",
        "MOHOLLM + GP (Gemini 2.0 Flash) (Context)",
        "MOHOLLM + TabPFN (Gemini 2.0 Flash) (Context)"
        "RS (KD-Partitioning) + LLM Surrogate (Gemini 2.0 Flash) (Context)"
        "RS + LLM Surrogate (Gemini 2.0 Flash) (Context)",
    ]

    invalid_files = []
    invalid_file_counts_less = []
    invalid_file_counts_more = []

    for benchmark in os.listdir(args.results_dir):
        benchmark_path = os.path.join(args.results_dir, benchmark)
        if not os.path.isdir(benchmark_path):
            continue
        for method in os.listdir(benchmark_path):
            # Only check whitelisted methods unless --check_all is set
            if not args.check_all and method not in whitelist:
                continue
            method_path = os.path.join(benchmark_path, method)
            observed_fvals_path = os.path.join(method_path, "observed_fvals")
            if not os.path.isdir(observed_fvals_path):
                continue
            csv_files = [
                f for f in os.listdir(observed_fvals_path) if f.endswith(".csv")
            ]
            # Check file count
            if len(csv_files) < args.min_files:
                invalid_file_counts_less.append((benchmark, method, len(csv_files)))
            elif len(csv_files) > args.min_files:
                invalid_file_counts_more.append((benchmark, method, len(csv_files)))
            # Check each file for row count
            for csv_file in csv_files:
                csv_path = os.path.join(observed_fvals_path, csv_file)
                try:
                    with open(csv_path, "r", newline="") as f:
                        reader = csv.reader(f)
                        row_count = sum(1 for _ in reader) - 1  # Exclude header
                    if row_count < args.min_trials:
                        invalid_files.append((benchmark, method, csv_file, row_count))
                except Exception as e:
                    print(f"Error reading {csv_path}: {e}")

    # Print and collect output for console
    print("\n=== Invalid CSV Files (less than min_trials rows) ===")
    for benchmark, method, csv_file, row_count in invalid_files:
        print(f"{benchmark}/{method}/observed_fvals/{csv_file}: {row_count} rows")

    print("\n=== Invalid File Counts (less than min_files) ===")
    for benchmark, method, count in invalid_file_counts_less:
        print(
            f"{benchmark}/{method}/observed_fvals: {count} files (expected {args.min_files})"
        )

    print("\n=== Invalid File Counts (more than min_files) ===")
    for benchmark, method, count in invalid_file_counts_more:
        print(
            f"{benchmark}/{method}/observed_fvals: {count} files (expected {args.min_files})"
        )

    # Write results to CSV files in the root directory
    root_dir = os.path.dirname(os.path.dirname(__file__))
    invalid_csv_path = os.path.join(root_dir, "invalid_results_files.csv")
    invalid_count_less_path = os.path.join(root_dir, "invalid_results_counts_less.csv")
    invalid_count_more_path = os.path.join(root_dir, "invalid_results_counts_more.csv")

    # Save invalid files (row count)
    with open(invalid_csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["file", "row_count", "expected_min_trials"])
        for benchmark, method, csv_file, row_count in invalid_files:
            file_path = f"{benchmark}/{method}/observed_fvals/{csv_file}"
            writer.writerow([file_path, row_count, args.min_trials])

    # Save invalid file counts (less than min_files)
    with open(invalid_count_less_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["folder", "file_count", "expected_min_files"])
        for benchmark, method, count in invalid_file_counts_less:
            folder_path = f"{benchmark}/{method}/observed_fvals"
            writer.writerow([folder_path, count, args.min_files])

    # Save invalid file counts (more than min_files)
    with open(invalid_count_more_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["folder", "file_count", "expected_min_files"])
        for benchmark, method, count in invalid_file_counts_more:
            folder_path = f"{benchmark}/{method}/observed_fvals"
            writer.writerow([folder_path, count, args.min_files])


if __name__ == "__main__":
    main()
