# PrDT4Rec

Preference-weighted offline recommendation with a Decision ConvFormer policy.
A Bradley–Terry trajectory scorer supplies frozen preference weights for policy training.

## Install

Use Python 3.11 or later and install a PyTorch build appropriate for your hardware.

```sh
python -m pip install -e '.[dev]'
python -m prdt4rec list
python -m pytest
```

## Configuration

[configs/prdt.json](configs/prdt.json) contains shared defaults and dataset overrides for MovieLens, KuaiRec, Coat, YahooR3, KuaiRand and VirtualTB. Nested overrides preserve common settings. Use `--config` to supply an edited copy. The config retains the reported sampling, label, feedback and training-budget differences rather than silently replacing them with a common recipe.

## Train and evaluate

The prepared-bundle launcher supports **MovieLens, KuaiRec and Coat**. Supply independently prepared trajectories, preference pairs, frozen feature tables and environment assets as described in [docs/DATA.md](docs/DATA.md). Dataset downloads, trained response models and experiment outputs are not bundled.

```sh
python -m prdt4rec check --profile coat --data-root ./data/coat --dry-run
python -m prdt4rec prepare --profile coat --data-root ./data/coat --output-root ./outputs/coat/seed0 --seed 0
python -m prdt4rec scorer --profile coat --data-root ./data/coat --output-root ./outputs/coat/seed0 --seed 0 --device cuda
python -m prdt4rec policy --profile coat --data-root ./data/coat --output-root ./outputs/coat/seed0 --seed 0 --device cuda
python -m prdt4rec evaluate --profile coat --data-root ./data/coat --output-root ./outputs/coat/seed0 --seed 0 --device cuda
```

Select `movielens` or `kuairec` for the other bundle adapters. Use seeds 0–4 with separate output directories. Devices include `cpu`, `mps`, `cuda` and `auto`. Resume with the same inputs using `resume --through evaluate` and the same arguments. Checkpoints preserve optimizer and random-number state.

Evaluation uses frozen policies, supplied matched development contexts, native stopping and H30. Reserved final users are excluded. The context capacity used for training is a separate setting.

YahooR3, KuaiRand and VirtualTB configurations are included as parameter specifications, but their dedicated data/simulator pipelines are not part of this compact release. The launcher rejects these profiles rather than using an incorrect adapter. Exact continuous PrDT policy, scorer and Gaussian likelihood primitives are provided in `prdt4rec/continuous.py`; a VirtualTB end-to-end runner is not included.

## Contents

- `prdt4rec/`: policy, scorer, objective, prepared-bundle training/evaluation and resume.
- `configs/prdt.json`: dataset parameters.
- `docs/DATA.md`: required data schema, units and evaluation adapters.
- `tests/`: synthetic pipeline, resume and native-stopping checks. Tests create temporary fixtures and require no benchmark data.

No baseline implementations, experimental results, checkpoints or generated datasets are included. Third-party attribution is in [NOTICE.md](NOTICE.md); see [LICENSE](LICENSE).
