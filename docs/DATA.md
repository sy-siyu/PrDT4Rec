# Data bundles and launcher

The entry point is `python -m prdt4rec` (also installed as `prdt`). Select `movielens`, `kuairec` or `coat` with `--profile`. All parameters are in `configs/prdt.json`: `defaults` contains shared settings, and `datasets` contains dataset-specific overrides. Nested overrides preserve the other shared values. Add `--config PATH` to use your own copy.

## Commands

- `list`: show public dataset profiles.
- `check --dry-run`: validate a prepared bundle without fitting or environment interaction.
- `prepare`: bind configuration, data and seed to an output directory.
- `scorer`: fit the Bradley–Terry scorer and cache TRAIN scores.
- `policy`: fit the preference-sampled policy.
- `evaluate`: evaluate the fixed policy on the declared development cases.
- `resume --through scorer|policy|evaluate`: continue an interrupted prepared run.

Use `--profile`, `--data-root`, `--output-root`, `--seed` and `--device` as shown in the README. Each seed needs a separate output directory. Data preparation precedes these commands; raw dataset CSV files are not launcher inputs.

## Bundle manifest

`DATA_ROOT/bundle.json` declares `dataset`, `profile_id`, `assets`, `split_counts`, `pair_counts`, `units`, `preference`, `evaluation` and `sealed_final_users`.

Every asset has a relative `path`, SHA-256 and an `arrays` map specifying each NumPy array's shape and dtype. JSON assets use an empty arrays map. Paths must stay inside the bundle. Split and pair counts must match the selected profile.

| Asset | Contents |
|---|---|
| `corpus` | Padded actions, scorer feedback, policy feedback, valid masks, encoded and raw users, episode IDs, split IDs, reset contexts, vocabulary sizes and context capacity |
| `features` | Frozen item vectors; MovieLens and Coat also require user vectors |
| `pairs` | Ordered TRAIN and validation endpoint IDs, plus explicit targets in {0, 0.5, 1} |
| `collectors` | Episode IDs and collector names |
| `cohorts` | JSON containing `train_users`, `validation_users`, `development_users`, `final_users` in raw IDs |
| `dev_cases` | Raw users, encoded users and initial item contexts |
| `world` | Native reward matrix and aligned raw user/item IDs |
| `distance` | MovieLens/Coat item distances |
| `categories`, `response_transform` | KuaiRec category rows and TRAIN-derived response transformation |

The corpus arrays are `actions`, `feedback`, `policy_feedback`, `mask`, `users`, `raw_users`, `indices`, `splits`, `initial_context_actions`, `preference_score`, `n_users`, `n_items`, and `max_length`. Cumulative-label profiles additionally provide `label_feedback`.

Actions use 0 for padding, 1 for unknown and catalog IDs starting at 2. Masks contain a contiguous valid prefix and retain the terminal action. Splits are 0 for TRAIN and 1 for offline validation. Model storage capacity is 40; evaluate at H30 without truncating training episodes. TRAIN and validation pairs never cross episode partitions. User overlap between TRAIN and offline validation is permitted; development and final users are excluded from both.

Pair arrays are `train_preferred`, `train_rejected`, `validation_preferred`, `validation_rejected`, `train_targets` and `validation_targets`. Endpoint names indicate stored order; a target of 0 reverses that ordering, and 0.5 expresses an exact tie. Preserve pair order and include all three target values in the logistic loss.

## Feedback and labels

`units.training` specifies the scorer input mapping; `units.policy_training` specifies the policy input mapping; `units.evaluation` specifies the mapping for feedback observed during evaluation. An affine mapping contains `mapping: affine`, `offset`, `scale`, `lower_clip`, `upper_clip`; identity means the saved channel is already transformed. Affine feedback is `(value + offset) / scale` with only the declared clipping.

`preference` contains `rule`, `stored_feedback_threshold`, `validation_rule`, and `validation_stored_feedback_threshold`, matching the profile. For `cumulative_logged_feedback`, sum the saved `label_feedback` over valid actions in float64, including terminal feedback, to produce `preference_score` on the split using that rule. For a count-based split, `preference_score` is the count above its declared threshold. Compare the two scores to obtain 1, 0 or 0.5. Labels are computed before input normalization.

| Dataset | Scorer input | Policy training input | Evaluation feedback | Labels |
|---|---|---|---|---|
| MovieLens | Saved shifted modeled feedback, divided by 5.957404786348342 and clipped to [0,1] | Same as scorer | Native rating / 5 | Sum of saved shifted modeled feedback; validation diagnostics use positive counts at threshold 3.5120950520038603 |
| KuaiRec | Canonical response feedback, already transformed | Same as scorer | Per-item TRAIN range normalization minus the response-model minimum | Counts above 0.044278962537646294 |
| Coat | Saved modeled Base/S feedback, already transformed | Saved native rating / 5 | Native rating / 5 | Sum of saved native ratings |

Provide policy feedback as a separate aligned array even when its values equal scorer feedback. For Coat, the modeled scorer channel and native policy channel are different inputs. KuaiRec's per-item evaluation transform requires aligned `watch_min`, `watch_max`, `raw_items` and scalar `response_matrix_minimum`; zero TRAIN range maps to zero before subtracting the response minimum. No additional clipping is applied.

## Native evaluation

Use `phase: development`, `horizon: 30`, the declared `cases_asset` and the dataset adapter:

| Dataset | Adapter | Leave window | Leave threshold |
|---|---|---:|---:|
| MovieLens | `movielens_native_v1` | 7 | 75 |
| KuaiRec | `kuairec_native_v1` | 9 | 1 |
| Coat | `coat_native_v1` | 7 | 6 |

Native stopping remains active before the horizon. Repeated recommendations are permitted. Reset contexts are not counted as executed recommendations; terminal rewards are included. Reports give mean cumulative raw reward, total raw reward divided by total steps, and mean episode length.

## Training parameters

All three profiles use 25 reference epochs of 59,200 trajectory draws, batch size 128 including the final partial batch, and 11,575 policy updates. The cosine schedule spans 46,300 updates; it is not compressed to the fitted 25-epoch budget. Scorer training uses ten complete passes over TRAIN pairs.

MovieLens fixes collector sampling mass to A2C 0.360843300498337, DORL 0.5302017011589523 and PPOC 0.10895499834271079, retaining preference ratios within each collector. KuaiRec and Coat normalize globally. Sampling weights are not multiplied into the policy loss again.

Raw datasets, generated corpora and feature assets are prepared separately. This repository provides the model, profiles, bundle validation and stage commands.

