# PrDT4Rec

Implementation of *Trajectory-Reweighted Offline Reinforcement Learning for Preference-Aligned Recommendations*.

PrDT4Rec learns a recommendation policy from fixed offline trajectories. It fits a Bradley–Terry scorer to within-user session comparisons ordered by implicit-feedback aggregates. A global softmax converts the frozen scores into weights for autoregressive imitation. The causal Decision ConvFormer policy predicts actions from the interaction history. Here, preference alignment refers to using the proxy-derived session ordering to guide policy training. The inverse temperature β controls how strongly training emphasizes higher-scoring trajectories.

This repository provides model implementations, dataset configurations, data preparation tools, and training and evaluation commands for MovieLens, KuaiRec, Coat, YahooR3, KuaiRand, and VirtualTB.

![Figure 1. PrDT4Rec fits a trajectory scorer to within-user episode pairs, then uses its frozen scores to weight policy training on the same offline corpus.](docs/assets/prdt4rec-figure-1.png)

## Install

Use Python 3.11 or later and install a PyTorch build appropriate for your hardware.

```sh
python -m pip install -e '.[dev]'
python -m prdt4rec list
python -m pytest
```

For upstream benchmark environment and dataset setup, see the [EasyRL4Rec repository and setup instructions](https://github.com/chongminggao/EasyRL4Rec).

## Configuration

[configs/prdt.json](configs/prdt.json) contains the dataset configurations for MovieLens, KuaiRec, Coat, YahooR3, KuaiRand, and VirtualTB. Use `--config` to supply an edited copy.

## Train and evaluate

Prepare a data bundle using [docs/DATA.md](docs/DATA.md), then run:

```sh
python -m prdt4rec check --profile coat --data-root ./data/coat --dry-run
python -m prdt4rec prepare --profile coat --data-root ./data/coat --output-root ./outputs/coat/seed0 --seed 0
python -m prdt4rec scorer --profile coat --data-root ./data/coat --output-root ./outputs/coat/seed0 --seed 0 --device cuda
python -m prdt4rec policy --profile coat --data-root ./data/coat --output-root ./outputs/coat/seed0 --seed 0 --device cuda
python -m prdt4rec evaluate --profile coat --data-root ./data/coat --output-root ./outputs/coat/seed0 --seed 0 --device cuda
```

Use the profile names from `python -m prdt4rec list`. Use a separate output directory per seed and resume with the same inputs and `resume --through evaluate`. VirtualTB evaluation also takes `--simulator-root`. Evaluation uses frozen policies, matched development cases, native stopping, and a 30-step cap.

Tests use synthetic data. Benchmark data, response models, baseline implementations, and experimental outputs are external to this release. See [NOTICE.md](NOTICE.md) and [LICENSE](LICENSE).
