# LLM-based Global Optimization with Search Space Partitioning

This repository accompanies the paper "[Improving LLM-based Global Optimization with Search Space Partitioning](https://openreview.net/forum?id=y6nhcCdQYd)" (ICLR 2026) and the related arXiv preprint "[Multi-objective Hierarchical Optimization with Large Language Models](https://arxiv.org/pdf/2601.13892)". (MO)HOLLM use LLMs to help with global optimization by partitioning the search space.

## 1) Installation (recommended via conda)
1. Create and activate environment:
   ```bash
   conda create -n HOLLM python=3.11 -y
   conda activate HOLLM
   ```
2. Install package (editable):
   ```bash
   pip install -e .
   ```
3. Optional developer extras (benchmarks, experiments, testing):
   ```bash
   pip install -e ".[dev]"
   ```
   If you only need core functionality, skip the dev extras. Some benchmarks require additional packages (see Benchmarks section).

## 2) Required API keys / environment variables
Set any API keys you plan to use. Example (add to `~/.bashrc` or `~/.zshrc`):
```bash
export OPEN_AI_API_KEY="your_openai_key_or_empty_string"
export GOOGLE_AI_API_KEY="your_google_api_key_or_empty_string"
export GROQ_AI_API_KEY="your_groq_api_key_or_empty_string"
```
Note: If you do not have access to an API, set the keys to empty strings.

Reload shell:
```bash
source ~/.bashrc   # or ~/.zshrc
```

## 3) Hugging Face (if using HF models)
1. Create a Hugging Face account and generate a token: https://huggingface.co/settings/tokens  
2. Login from CLI (required once for some models):
```bash
huggingface-cli login
```
This stores your token locally and grants access to models for which you've accepted terms.

## 4) Minimal usage / Quickstart
Run the Simple2D example:
```bash
python main.py --config_file Simple2D/mohollm-Simple2D
```
Or use the installed CLI (after pip install -e .):
```bash
hollm --config_file Simple2D/mohollm-Simple2D
```
Notes:
- When using the CLI, complex nested args (lists/dicts) are best set via configuration JSON files inside `configurations/` and referenced with `--config_file`.
- For reproducibility, set a random seed inside your config.

## 5) Benchmarks (core)
Available `benchmark` options:
- NB201, ZDT, WELDED_BEAM, VLMOP, DTLZ, Simple2D, ChankongHaimes, TestFunction4, SchafferN1, SchafferN2, Poloni, BraninCurrin, GMM, ToyRobust, penicillin, vehicle_safety, car_side_impact, Kursawe

Some require additional `benchmark_settings` — see `benchmark_initialization.py` for details.

## 6) Benchmarks (Syne-Tune / dev extra)
Additional Syne-Tune benchmarks are provided in `experiments/benchmarks.py` (e.g., `fcnet-*`, `nas201-*`, `lcbench-*`, `tabrepo-*`). Install dev extras (`pip install -e ".[dev]"`) or `pymoo` as needed:
```bash
pip install -e ".[dev]"
# or for specific needs
pip install pymoo
```

## 7) Common troubleshooting
- "Missing API key" — ensure env vars are exported and your shell is reloaded.
- "Hugging Face permission error" — accept the model terms on HF website and run `huggingface-cli login`.
- Dependency problems — recreate the conda env and run `pip install -e .` again.
- If a benchmark fails, inspect `benchmark_initialization.py` and verify required extras are installed.

## 8) Running tests
If the project includes tests (dev extras), run:
```bash
pytest -q
```
(or the test command specified in the repository)

## 9) Contributing & contact
- Open issues or PRs for bugs and feature requests.
- Follow repository code style and tests for contributions.

## BibTeX

If you use this work, please cite related resources with the following BibTeX entries:

```bibtex
@inproceedings{schwanke2026improving,
   title       = {Improving {LLM}-based Global Optimization with Search Space Partitioning},
   author      = {Andrej Schwanke and Lyubomir Ivanov and David Salinas and Fabio Ferreira and Aaron Klein and Frank Hutter and Arber Zela},
   booktitle   = {The Fourteenth International Conference on Learning Representations},
   year        = {2026},
   url         = {https://openreview.net/forum?id=y6nhcCdQYd}
}

@misc{schwanke2026multi,
  title        = {Multi-objective Hierarchical Optimization with Large Language Models},
  author       = {Andrej Schwanke and Lyubomir Ivanov and David Salinas and Frank Hutter and Arber Zela},
  year         = {2026},
  note         = {arXiv preprint},
  url          = {https://arxiv.org/pdf/2601.13892}
}