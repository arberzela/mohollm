


# ⚠️ **_This project is currently a Work in Progress._** ⚠️

HOLLM is under development. Please note that certain features may not be fully implemented or tested yet. Furthermore, setting up the project in the current state may be difficult due to the sparse documentation. All things in this readme can change quickly. The readme might not reflect the current state of the project all the time. For more information take a look into the commits and closed issues.

# HOLLM: Improving LLM-based Global Optimization with Search Space Partitioning

## Installation (Python 3.11)

1. Create a conda environment
   ```
   conda create -n HOLLM python=3.11
   conda activate HOLLM
   ```

2. Install dependencies:
   ```
   pip install -e .
   ```

   For benchmarks and experiment utilities:
   ```
   pip install -e ".[dev]"
   ```

   Note: Some benchmarks (e.g., `DTLZ`, `ZDT`, `VLMOP`, `WELDED_BEAM`) require `pymoo`. If you only need core functionality, you can skip the dev extras. To run those benchmarks, install the dev extras or `pymoo` directly:

   ```
   pip install -e ".[dev]"
   # or
   pip install pymoo
   ```

3. Set the required API keys as environment variables in your shell configuration (for example, in your ~/.bashrc or ~/.zshrc):

   ```bash
   export OPEN_AI_API_KEY="<YOUR_API_KEY>"
   export GOOGLE_AI_API_KEY=""
   export GROQ_AI_API_KEY=""
   ```

   Reload your terminal or source the configuration file to apply the changes. If you don't have access to the OpenAI API, set all keys to an empty string.

## Huggingface

1. Create an account
2. Create an API key: https://huggingface.co/settings/tokens
3. For some models you need the access rights. You have to accepts the terms of service on the huggingface site for that model and then login via the shell. This has to be done only once. (https://huggingface.co/docs/huggingface_hub/en/guides/cli)
   ```
   huggingface-cli login
   ```
   It will ask you for the API toke you previously created. The key will now be stored in a local cache file and you don't have to do anything anymore.


# How to use HOLLM (minimal)

## Quickstart (single benchmark)

Run the minimal example using the provided Simple2D config:

```
python main.py --config_file Simple2D/mohollm-Simple2D
```

Or with the installed CLI:

```
hollm --config_file Simple2D/mohollm-Simple2D
```

Note: list/dict/boolean CLI arguments are not intended for direct shell use. Use a JSON config in [configurations](configurations) and pass it via `--config_file`.

## Benchmarks (core)

Set `benchmark` in your config to one of:

- NB201
- ZDT
- WELDED_BEAM
- VLMOP
- DTLZ
- Simple2D
- ChankongHaimes
- TestFunction4
- SchafferN1
- SchafferN2
- Poloni
- BraninCurrin
- GMM
- ToyRobust
- penicillin
- vehicle_safety
- car_side_impact
- Kursawe

Benchmark-specific settings (e.g., `benchmark_settings`) are required for some options like `NB201`. See [benchmark_initialization.py](benchmark_initialization.py) for details.

## Benchmarks (Syne-Tune, dev extra)

Additional Syne-Tune benchmarks are defined in [experiments/benchmarks.py](experiments/benchmarks.py) (e.g., `fcnet-*`, `nas201-*`, `lcbench-*`, `tabrepo-*`). These require the dev extra (see below).