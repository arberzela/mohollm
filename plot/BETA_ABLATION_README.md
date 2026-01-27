# Beta Ablation Plot for Poloni Benchmark

This directory contains a script to generate log hypervolume difference plots for beta ablation studies on the Poloni benchmark.

## Overview

The script `plot_beta_ablation_poloni.py` does the following:

1. **Computes Max HV from Good Pool**: Reads the Pareto front approximation from `scripts/data_pools/custom_poloni_good_pool.csv` and computes the maximum achievable hypervolume
2. **Saves Max HV**: Stores the computed max HV value to `scripts/data_pools/max_hv_values.json` for future use
3. **Generates Plot**: Creates a log hypervolume difference plot showing the performance of different beta values over time

## Key Features

- **Data Pool-Based Max HV**: Unlike other benchmarks that use empirically determined max HV values, this script computes the max HV directly from the known Pareto front approximation
- **Beta Ablation Support**: Automatically labels methods with beta values using LaTeX formatting (e.g., `MOHOLLM ($\beta=0.5$)`)
- **Configurable**: Supports filtering, whitelisting, blacklisting, and custom axis limits

## Usage

### Basic Usage

```bash
python plot/plot_beta_ablation_poloni.py \
    --benchmark "Poloni" \
    --title "Poloni Beta Ablation" \
    --data_path "./results" \
    --pool_file "./scripts/data_pools/custom_poloni_good_pool.csv" \
    --trials 100 \
    --filename "poloni_beta_ablation_log_hv_diff"
```

### Using the Shell Script

A convenience script is provided:

```bash
./plot/scripts/plot_beta_ablation_poloni.sh
```

You may need to edit the script to adjust paths and parameters.

## Arguments

- `--benchmark`: Benchmark name (default: "Poloni")
- `--title`: Plot title (default: "Poloni")
- `--data_path`: Path to results directory (required)
- `--pool_file`: Path to good pool CSV file (default: "./scripts/data_pools/custom_poloni_good_pool.csv")
- `--filter`: Filter for method names (optional)
- `--trials`: Number of trials to plot (required)
- `--blacklist`: Comma-separated list of methods to exclude (optional)
- `--whitelist`: Comma-separated list of methods to include (optional)
- `--filename`: Output filename without extension (default: "poloni_beta_ablation")
- `--x_lim`: X-axis limits as two values (optional)
- `--y_lim`: Y-axis limits as two values (optional)

## Output

The script generates three file formats:
- `plots/beta_ablation_poloni/pdf/[filename].pdf`
- `plots/beta_ablation_poloni/png/[filename].png`
- `plots/beta_ablation_poloni/svg/[filename].svg`

It also saves the computed max HV to:
- `scripts/data_pools/max_hv_values.json`

## Label Mapping

The script uses the `LABEL_MAP` from `plot_settings.py` to map method names to display labels:

- `MOHOLLM (Gemini 2.0 Flash) (Beta=X)` → `MOHOLLM ($\beta=X$)`
- `mohollm (Gemini 2.0 Flash) (Beta=X)` → `LLM ($\beta=X$)`

## Testing

To verify the max HV computation:

```bash
python scripts/test_max_hv_poloni.py
```

This will display:
- The reference point used
- Number of points in the good pool
- The computed maximum hypervolume
- Statistics about the objective values

## Requirements

- pandas
- numpy
- matplotlib
- pymoo

These should already be installed if you have the MO-LLAMBO environment set up.
