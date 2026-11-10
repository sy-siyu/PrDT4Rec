"""Convert retained KuaiRand inputs into a portable bundle."""
import ast
import csv
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


def preference_scores(feedback, mask):
    """Sum each executed native float32 prefix as in the retained pair builder."""
    feedback = np.asarray(feedback)
    mask = np.asarray(mask)
    if feedback.dtype != np.float32 or feedback.shape != mask.shape or feedback.ndim != 2:
        raise ValueError("KuaiRand preference labels require aligned native float32 feedback")
    if not np.isfinite(feedback[mask]).all() or np.any((feedback[mask] < 0) | (feedback[mask] > 1)):
        raise ValueError("KuaiRand logged click proxies must be in [0, 1]")
    # Padding changes the summation grouping for short prefixes. Preserve the
    # original NumPy reduction over each trajectory's executed elements.
    return np.asarray([values[valid].sum() for values, valid in zip(feedback, mask)],
                      dtype=np.float64)


def _native_categories(path, count):
    tags = np.full((count, 3), -2, dtype=np.int64)
    with Path(path).open(encoding="utf-8", newline="") as stream:
        rows = csv.DictReader(stream)
        if rows.fieldnames is None or "tag" not in rows.fieldnames:
            raise ValueError("KuaiRand item metadata requires the native tag column")
        seen = 0
        for expected, row in enumerate(rows):
            if expected >= count or (row.get("video_id") not in (None, "")
                                     and int(row["video_id"]) != expected):
                raise ValueError("KuaiRand metadata differs from native catalog order")
            raw = row["tag"]
            values = [-1] if raw in (None, "", "nan", "NaN") else ast.literal_eval(f"[{raw}]")
            if not 1 <= len(values) <= 3:
                raise ValueError("invalid KuaiRand native tag list")
            native = [int(value) for value in values]
            if any(value < -1 for value in native):
                raise ValueError("invalid native KuaiRand tag")
            tags[expected, :len(native)] = native
            seen += 1
    if seen != count:
        raise ValueError("KuaiRand metadata does not cover the complete catalog")
    return tags


def _npz(path, names):
    with np.load(path, allow_pickle=False) as arrays:
        return {name: arrays[name] for name in names}


def convert_bundle(source_root, data_root, *, cohorts, world, categories, profile="kuairand"):
    """Convert frozen TRAIN/offline-VAL and DEV-only assets without refitting."""
    from ..environments.kuairand import KuaiRandDEV
    from ..runner import check, digest, load_config, put, read

    source = Path(source_root).resolve()
    destination = Path(data_root).resolve()
    config = load_config(profile)
    if config["dataset"] != "KuaiRand" or destination.exists():
        raise ValueError("choose a KuaiRand profile and a new destination")
    data = config["data"]
    binding = read(source / "bundle.json")
    if data.get("bundle_identity") and binding.get("identity") != data["bundle_identity"]:
        raise ValueError("source does not match the retained KuaiRand bundle")
    originals = ("offline.npz", "representation.npz", "pairs.npz", "lineage.npz", "dev_cases.npz")
    for name in originals:
        if digest(source / name) != binding["assets"][name]["sha256"]:
            raise ValueError(f"retained KuaiRand asset changed: {name}")

    corpus = _npz(source / "offline.npz",
                  ("actions", "feedback", "mask", "users", "indices", "splits",
                   "initial_context_actions", "n_users", "n_items", "max_length"))
    corpus["raw_users"] = corpus["users"] - 1
    corpus["policy_feedback"] = corpus["label_feedback"] = corpus["feedback"]
    corpus["preference_score"] = preference_scores(corpus["label_feedback"], corpus["mask"])
    features = _npz(source / "representation.npz", ("user_vectors", "item_vectors"))
    pairs = _npz(source / "pairs.npz",
                 tuple(f"{part}_{side}" for part in ("train", "validation")
                       for side in ("preferred", "rejected")))
    for part in ("train", "validation"):
        pairs[f"{part}_targets"] = np.ones(len(pairs[f"{part}_preferred"]), dtype=np.float32)
    lineage = _npz(source / "lineage.npz", ("global_episode_ids", "collector_codes"))
    names = np.asarray(["a2c", "dorl", "ppoc", "guided_top2_low_feedback"])
    codes = lineage["collector_codes"]
    if np.any(codes < 0) or np.any(codes >= len(names)):
        raise ValueError("unknown KuaiRand collector code")
    collectors = dict(indices=lineage["global_episode_ids"], collectors=names[codes])

    saved_cases = _npz(source / "dev_cases.npz",
                       ("encoded_user_ids", "initial_context_item_columns"))
    cases = dict(raw_users=saved_cases["encoded_user_ids"],
                 encoded_users=saved_cases["encoded_user_ids"] + 1,
                 initial_context_actions=saved_cases["initial_context_item_columns"] + 2)
    retained_roles = read(Path(cohorts))
    if (retained_roles.get("schema") != "prdt4rec-kuairand-cohorts-v1"
            or retained_roles.get("frozen_before_new_simulator_outcomes") is not True):
        raise ValueError("a frozen native KuaiRand cohort manifest is required")
    role_names = {"train_users": "policy_train", "validation_users": "offline_validation",
                  "development_users": "policy_dev", "final_users": "final_sealed"}
    roles = {name: retained_roles["roles"][source_name]["encoded_zero_based"]
             for name, source_name in role_names.items()}
    if cases["raw_users"].tolist() != roles["development_users"]:
        raise ValueError("retained DEV cases must match the ordered cohort manifest")
    world_arrays = _npz(Path(world), ("rewards", "raw_users", "raw_items"))
    tags = _native_categories(categories, int(corpus["n_items"]) - 2)
    KuaiRandDEV(world_arrays["rewards"], world_arrays["raw_users"],
                world_arrays["raw_items"], cases, tags)
    evaluation = dict(adapter="kuairand_native_v1", phase="development", horizon=30,
                      leave_window=10, cases_asset="dev_cases")
    preference = dict(rule=data["preference_rule"],
                      validation_rule=data["validation_preference_rule"],
                      stored_feedback_threshold=data["stored_feedback_threshold"],
                      validation_stored_feedback_threshold=data["validation_stored_feedback_threshold"])

    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".kuairand-", dir=destination.parent) as temp:
        root = Path(temp) / "bundle"
        root.mkdir()
        assets = {}
        for key, value in dict(corpus=corpus, features=features, pairs=pairs, collectors=collectors,
                               dev_cases=cases, world=world_arrays,
                               categories={"categories": tags}).items():
            name = f"{key}.npz"
            np.savez_compressed(root / name, **value)
            assets[key] = dict(path=name, sha256=digest(root / name),
                               arrays={k: dict(shape=list(v.shape), dtype=str(v.dtype))
                                       for k, v in value.items()})
        put(root / "cohorts.json", roles)
        assets["cohorts"] = dict(path="cohorts.json", sha256=digest(root / "cohorts.json"), arrays={})
        put(root / "bundle.json", dict(dataset="KuaiRand", profile_id=config["profile_id"],
            assets=assets, split_counts=data["split_counts"], pair_counts=data["pair_counts"],
            units=dict(training=data["training_units"], policy_training=data["policy_training_units"],
                       evaluation=config["evaluation"]["feedback_units"]),
            preference=preference, evaluation=evaluation, sealed_final_users=roles["final_users"]))
        check(profile, root)
        if destination.exists():
            raise FileExistsError(destination)
        root.rename(destination)
    return destination


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True,
                        help="retained shared directory containing the input manifest and NPZs")
    parser.add_argument("--data-root", type=Path, required=True, help="new portable bundle directory")
    parser.add_argument("--cohorts", type=Path, required=True, help="retained native cohort JSON")
    parser.add_argument("--world", type=Path, required=True,
                        help="DEV-only NPZ of native float32 rewards, raw_users and raw_items")
    parser.add_argument("--categories", type=Path, required=True,
                        help="native item metadata CSV containing tag and optionally video_id")
    parser.add_argument("--profile", default="kuairand", help="profile name or config path with #profile")
    args = parser.parse_args(argv)
    print(convert_bundle(args.source_root, args.data_root, cohorts=args.cohorts, world=args.world,
                         categories=args.categories, profile=args.profile))


if __name__ == "__main__":
    main()
