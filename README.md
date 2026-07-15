# Evolving Multi-Channel Confidence-Aware Activation Functions

This repository accompanies our GECCO 2026 paper on *evolving multi-channel,
confidence-aware activation functions* for neural networks that learn from data with
missing values. Instead of applying a single fixed nonlinearity (ReLU, Swish, and the like)
to every unit, we use genetic programming to **evolve** an activation function that reacts
not only to a unit's value, but also to whether that value was originally missing and to how
confident the network is in it.

In the paper we evaluate the approach on **twelve benchmark datasets** across a range of
missing-data mechanisms and rates. To keep this repository small and easy to run, we include
a single, self-contained example — the UCI **HouseVotes84** dataset. The code is written so
you can point it at another dataset and reproduce the same pipeline without changing the core
method.

## The idea in brief

Real-world tabular data is rarely complete, and standard networks treat an imputed value the
same as a genuinely observed one. We give every hidden unit three channels instead of one:

- **`x`** — the value (as usual),
- **`m`** — a missing flag that tracks whether the value originated from a missing entry,
- **`c`** — a confidence score that is propagated through the network alongside the value.

The linear layers pass all three channels forward (confidence and missingness are propagated
using magnitude-normalized weights), and the activation function is an evolved expression over
`x`, `m`, and `c`. Because the activation can "see" `m` and `c`, it can down-weight or reshape
its response for values it should not trust — something a fixed activation cannot do.

Activation functions are represented as expression trees over the three inputs, numeric
constants, and a library of unary and binary operators (`sin`, `tanh`, `sigmoid`, `relu`,
`elu`, `log`, `exp`, `max`, `min`, `+`, `-`, `*`, `/`, and more). A genetic algorithm evolves
these trees using fitness-proportional selection, subtree crossover, point mutation, and
elitism. Fitness rewards validation accuracy while gently penalizing overly large or deep
trees and encouraging use of all three channels. The best evolved activation (which we call
**3C-EA**) is then retrained and compared against ReLU, Swish, LeakyReLU, and ELU, with
statistical significance assessed by the Wilcoxon signed-rank test.

## Repository contents

| File | Description |
|------|-------------|
| `new.py` | End-to-end pipeline: data loading, missing-data injection, evolution, training, evaluation, and statistical testing. |
| `metrics_report.py` | Helper functions for computing and printing the aggregated metrics. |
| `HouseVotes84.csv` | UCI Congressional Voting Records dataset. Votes are encoded `y`/`n`, missing entries as `?`, and the label lives in a `Class` column. |
| `requirements.txt` | Python dependencies. |

## Installation

```bash
pip install -r requirements.txt
```

The code targets Python 3.9+ and depends on numpy, pandas, torch, scikit-learn, and scipy.
A GPU is used automatically if available, but it is not required — the models are small.

## Running the example

```bash
python new.py
```

This runs the full study on HouseVotes84 (25 repeated runs by default). When it finishes you
will see, printed to the console:

- the aggregated results table (mean ± std for 3C-EA and every baseline, across all metrics),
- the Wilcoxon signed-rank tests comparing 3C-EA against each baseline.

A detailed report is also written to `results_<mechanism>_<rate>pct_<runs>runs.txt` with four
sections: per-run values, aggregated statistics, the full Wilcoxon tests, and ready-to-paste
LaTeX significance markers.

## Using a different dataset

The method itself is dataset-agnostic — only the loader is specific to HouseVotes84. To run
on your own data:

1. Provide a CSV whose feature columns are numeric and whose target is in a column named
   `Class` (or adjust the label handling in `load_dataset` in `new.py`).
2. Point the loader at your file, e.g. `load_dataset("my_dataset.csv")`, or change the default
   path at the top of that function.
3. Optionally choose how missingness is simulated via `CONFIG["missing_mechanism"]`
   (`MCAR`, `MAR`, or `MNAR`) and `CONFIG["missing_rate"]`. If your data already contains
   missing values, set the rate to `0.0` and they will be used as-is.

Everything else — the evolution, the three-channel network, the baselines, and the statistics
— stays the same.

## Configuration

All run settings live in the `CONFIG` dictionary near the bottom of `new.py`:

| Key | Meaning | Default |
|-----|---------|---------|
| `missing_rate` | Fraction of values made missing | `0.00` |
| `missing_mechanism` | `MCAR`, `MAR`, or `MNAR` | `MAR` |
| `population_size` | GA population size | `100` |
| `generations` | GA generations | `20` |
| `max_tree_depth` | Maximum activation-tree depth | `3` |
| `mutation_rate` | Per-node mutation probability | `0.15` |
| `crossover_rate` | Crossover probability | `0.70` |
| `elite_size` | Elites carried to the next generation | `2` |
| `hidden_sizes` | MLP hidden-layer sizes | `[64, 32]` |
| `num_epochs_fitness` | Training epochs per fitness evaluation | `30` |
| `final_epochs` | Training epochs for the final model | `100` |
| `final_patience` | Early-stopping patience | `15` |
| `base_seed` | Base random seed | `42` |
| `num_runs` | Number of repeated runs | `25` |

Run `i` uses seed `base_seed + i`, so results are fully reproducible.

## Baselines and statistical testing

Each evolved activation is compared against four widely used baselines — **ReLU**, **Swish**,
**LeakyReLU**, and **ELU** — trained under the same architecture and protocol. We report test
accuracy, precision, recall, specificity, F1, and AUC. Significance is assessed with the
one-sided Wilcoxon signed-rank test (`3C-EA > baseline`) over the repeated runs, using the
conventional thresholds `*` p < 0.05, `**` p < 0.01, and `***` p < 0.001.

## Citation

If you use this code, please cite the GECCO 2026 paper:

```bibtex
@inproceedings{10.1145/3795095.3805092,
author = {Shahabi Sani, Naeem and Najiantabriz, Ferial and Shafaei, Shayan and Hougen, Dean},
title = {Evolving Multi-Channel Confidence-Aware Activation Functions for Missing Data with Channel Propagation},
year = {2026},
isbn = {9798400724879},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3795095.3805092},
doi = {10.1145/3795095.3805092},
abstract = {Learning in the presence of missing data can result in biased predictions and poor generalizability, among other difficulties, which data imputation methods only partially address. In neural networks, activation functions significantly affect performance, yet typical options (e.g., ReLU, Swish) operate only on feature values and do not account for missingness indicators or confidence scores. We propose Three-Channel Evolved Activations (3C-EA), which we evolve using Genetic Programming to produce multivariate activation functions f(x, m, c) in the form of trees that take (i) the feature value x, (ii) a missingness indicator m, and (iii) an imputation confidence score c. To make these activations useful beyond the input layer, we introduce ChannelProp, an algorithm that deterministically propagates missingness and confidence values via linear layers based on weight magnitudes, retaining reliability signals throughout the network. We evaluate 3C-EA and ChannelProp on datasets with natural and injected (Missing Completely at Random, Missing at Random, and Missing Not at Random) missingness at multiple rates under identical preprocessing and splits. Results indicate that integrating missingness and confidence inputs into the activation search improves classification performance under missingness.},
booktitle = {Proceedings of the Genetic and Evolutionary Computation Conference},
pages = {1011–1019},
numpages = {9},
keywords = {genetic programming, missing data, activation functions, confidence propagation, neural networks, neuroevolution},
location = {Centro Internacional de Convenciones CIC-ANDE, San Jose, Costa Rica},
series = {GECCO '26}
}
```


