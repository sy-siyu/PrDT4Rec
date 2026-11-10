# Data bundles and launcher

The entry point is `python -m prdt4rec` (also installed as `prdt`). Use `list` for the six profile names and `configs/prdt.json` for their settings. Add `--config PATH` to use your own copy.

## Commands

- `list`: show public dataset profiles.
- `check --dry-run`: validate a prepared bundle without fitting or environment interaction.
- `prepare`: bind configuration, data and seed to an output directory.
- `scorer`: fit the Bradley–Terry scorer and cache TRAIN scores.
- `policy`: fit the preference-sampled policy.
- `evaluate`: evaluate the fixed policy on the declared development cases.
- `resume --through scorer|policy|evaluate`: continue an interrupted prepared run.

Use `--profile`, `--data-root`, `--output-root`, `--seed` and `--device` as shown in the README. Each seed needs a separate output directory. Prepare the frozen input assets before running these commands.

## Bundle conversion

The converters below take prepared benchmark assets. Obtain the underlying data, frozen representations, and simulator assets separately.

```sh
python -m prdt4rec.data.yahoor3 --source-root ./retained/yahoor3/shared \
  --cohorts ./retained/yahoor3/cohorts.json --matrix ./retained/yahoor3/yahoo_pseudoGT_ratingM.ascii \
  --distance ./retained/yahoor3/distance.npy --data-root ./data/yahoor3
python -m prdt4rec.data.kuairand --source-root ./retained/kuairand/shared \
  --cohorts ./retained/kuairand/cohorts.json --world ./retained/kuairand/dev-only.npz \
  --categories ./retained/kuairand/item_metadata.csv --data-root ./data/kuairand
python -m prdt4rec.data.virtualtb --train ./retained/virtualtb/train.npz \
  --validation ./retained/virtualtb/validation.npz --development ./retained/virtualtb/profiles.npz \
  --output-root ./data/virtualtb
```

YahooR3 takes the retained shared inputs, native cohort JSON, official ASCII rating matrix, and frozen item distances as a numeric NumPy array. KuaiRand takes the retained shared corpus, representation, pairs, lineage and DEV cases, native cohort JSON, a DEV-only float32 response matrix (`rewards`, `raw_users`, `raw_items`), and the item metadata CSV (`tag`, optional `video_id`). MovieLens, Coat and KuaiRec take prepared assets using the catalog bundle schema below.

## Catalog bundle manifest

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

Actions use 0 for padding, 1 for unknown and catalog IDs starting at 2. Masks contain a contiguous valid prefix and retain the terminal action. Splits are 0 for TRAIN and 1 for offline validation. Keep the configured training context capacity and evaluate at H30. TRAIN and validation pairs never cross episode partitions. User overlap between TRAIN and offline validation is permitted. Development and final users are excluded from both fitting partitions.

Pair arrays are `train_preferred`, `train_rejected`, `validation_preferred`, `validation_rejected`, `train_targets` and `validation_targets`. Endpoint names indicate stored order; a target of 0 reverses that ordering, and 0.5 expresses an exact tie. Preserve pair order and include all three target values in the logistic loss.

## Feedback and labels

`units.training` specifies the scorer input mapping; `units.policy_training` specifies the policy input mapping; `units.evaluation` specifies the mapping for feedback observed during evaluation. An affine mapping contains `mapping: affine`, `offset`, `scale`, `lower_clip`, `upper_clip`; identity means the saved channel is already transformed. Affine feedback is `(value + offset) / scale` with only the declared clipping.

`preference` contains `rule`, `stored_feedback_threshold`, `validation_rule`, and `validation_stored_feedback_threshold`, matching the profile. For a cumulative-label split, `preference_score` is the sum of valid `label_feedback` including terminal feedback. The YahooR3 and KuaiRand converters preserve the original float32 reductions. For a count-based split, `preference_score` is the count above its declared threshold. Compare the scores to obtain 1, 0 or 0.5.

| Dataset | Scorer input | Policy training input | Evaluation feedback | Labels |
|---|---|---|---|---|
| MovieLens | Saved shifted modeled feedback, divided by 5.957404786348342 and clipped to [0,1] | Same as scorer | Native rating / 5 | Sum of saved shifted modeled feedback; validation diagnostics use positive counts at threshold 3.5120950520038603 |
| KuaiRec | Canonical response feedback, already transformed | Same as scorer | Per-item TRAIN range normalization minus the response-model minimum | Counts above 0.044278962537646294 |
| Coat | Saved modeled Base/S feedback, already transformed | Saved native rating / 5 | Native rating / 5 | Sum of saved native ratings |
| YahooR3 | Native rating / 5 | Same as scorer | Native rating / 5 | Sum of normalised native ratings |
| KuaiRand | Native feedback | Same as scorer | Native feedback | Sum of native feedback |

Provide policy feedback as a separate aligned array even when its values equal scorer feedback. For Coat, the modeled scorer channel and native policy channel are different inputs. KuaiRec's per-item evaluation transform requires aligned `watch_min`, `watch_max`, `raw_items` and scalar `response_matrix_minimum`; zero TRAIN range maps to zero before subtracting the response minimum. No additional clipping is applied.

## Native evaluation

Use `phase: development`, `horizon: 30`, the declared `cases_asset` and the dataset adapter:

| Dataset | Adapter | Leave window | Leave threshold |
|---|---|---:|---:|
| MovieLens | `movielens_native_v1` | 7 | 75 |
| KuaiRec | `kuairec_native_v1` | 9 | 1 |
| Coat | `coat_native_v1` | 7 | 6 |
| YahooR3 | `yahoor3_native_v1` | 3 | 120 |
| KuaiRand | `kuairand_native_v1` | 10 | Any matching category |

Native stopping remains active before the horizon. Repeated recommendations are permitted. Reset contexts are not counted as executed recommendations; terminal rewards are included. Reports give mean cumulative raw reward, total raw reward divided by total steps, and mean episode length.

## Training parameters

The five catalog profiles use 25 epochs of 59,200 trajectory draws, with 11,575 policy updates on a 46,300-update cosine schedule. Scorer training uses ten passes over TRAIN pairs. MovieLens fixes collector sampling mass. The other catalog profiles normalise globally.

## VirtualTB bundle and simulator

The TRAIN and offline validation archives hold `observations` (float32, N×30×91), `actions` (float32, N×30×27), `rewards` (float32, N×30), `mask` (boolean, N×30), and `episode_ids`. The validation archive also holds the fixed local pair arrays `first`, `second`, and `targets`. The profile archive supplies `development_profiles` and `development_sha256`. The converter builds the within-profile TRAIN preference pairs and exports only the named development arrays.

For evaluation, set `--simulator-root` to a trusted VirtualTaobao checkout at commit `ff4f1cce83c824b9542ccd81300672cc81bdec7a`. It requires `gym==0.26.2` and the native `virtualTB/data/{action,generator,leave}_model.pt` files. The evaluator uses the policy's deterministic action mean and native simulator rewards and quitting. The scorer uses the lowest offline-validation loss checkpoint, and the policy uses its fixed 1,050-update endpoint.

