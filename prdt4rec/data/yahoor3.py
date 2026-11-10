"""Convert retained YahooR3 inputs into a portable bundle."""
from pathlib import Path
import tempfile

import numpy as np


def preference_scores(label_feedback, mask):
    """Sum normalized ratings using the retained per-episode float32 reduction."""
    ratings = label_feedback[mask]
    if not np.isfinite(ratings).all() or np.any((ratings < 1) | (ratings > 5)) or np.any(ratings != np.rint(ratings)):
        raise ValueError("YahooR3 preference labels require native integer ratings on executed steps")
    normalized = label_feedback.astype(np.float32) / np.float32(5)
    # Preserve the native one-dimensional reduction over each valid prefix.
    # Reducing over the entire padded width may round a different way.
    return np.fromiter((row[:length].sum(dtype=np.float32)
                        for row, length in zip(normalized, mask.sum(axis=1))),
                       dtype=np.float32, count=len(normalized))


def read_development_rows(matrix, users, catalog_size, *, total_rows=15400):
    """Parse requested rows of the official ASCII matrix in saved DEV order."""
    wanted = {int(user) - 1: position for position, user in enumerate(users)}
    if len(wanted) != len(users) or not all(0 <= row < total_rows for row in wanted):
        raise ValueError("YahooR3 matrix access requires unique one-based DEV user IDs")
    result = np.empty((len(users), catalog_size), dtype=np.float32)
    seen = set()
    row_count = 0
    with Path(matrix).open(encoding="ascii") as stream:
        for index, line in enumerate(stream):
            row_count = index + 1
            if index not in wanted:
                continue
            fields = line.split()
            if len(fields) != catalog_size:
                raise ValueError(f"YahooR3 DEV matrix row {row_count} has the wrong catalog width")
            row = np.asarray(fields, dtype=np.float32)
            if not np.isfinite(row).all() or np.any((row < 1) | (row > 5)) or np.any(row != np.rint(row)):
                raise ValueError(f"YahooR3 DEV matrix row {row_count} contains invalid native ratings")
            result[wanted[index]] = row
            seen.add(index)
    if row_count != total_rows or seen != wanted.keys():
        raise ValueError("YahooR3 matrix row count or DEV coverage differs")
    return result


def _npz(path):
    with np.load(path, allow_pickle=False) as source:
        return {name: source[name] for name in source.files}


def _source_cohorts(path):
    from ..runner import read
    source = read(path)
    if (source.get("schema") != "prdt4rec-yahoor3-cohorts-v1"
            or source.get("frozen_before_simulator_outcomes") is not True):
        raise ValueError("expected the frozen YahooR3 cohort manifest")
    names = dict(train_users="policy_train", validation_users="offline_validation",
                 development_users="policy_dev", final_users="final_sealed")
    roles = {}
    for public, native in names.items():
        ids = source["roles"][native]["encoded_zero_based"]
        if not isinstance(ids, list) or any(type(user) is not int or user < 0 for user in ids):
            raise ValueError(f"invalid YahooR3 role {native}")
        roles[public] = [user + 1 for user in ids]
    combined = [user for users in roles.values() for user in users]
    if sorted(combined) != list(range(1, len(combined) + 1)):
        raise ValueError("YahooR3 cohorts must partition the declared user vocabulary")
    return roles


def convert_bundle(source_root, data_root, *, cohorts, matrix, distance, profile="yahoor3"):
    """Bind source episodes, pairs, lookups and DEV assets to the portable schema.

    The source root is the frozen YahooR3 shared directory. The distance input
    is the original cache exported as a NumPy array without changing its dtype.
    The ASCII matrix is read only to interpret the frozen development rows.
    """
    from ..runner import BUNDLE_SCHEMA, check, deep_check, digest, load_config, put, read

    source_root = Path(source_root).resolve()
    destination = Path(data_root).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"bundle destination already exists: {destination}")
    config = load_config(profile)
    if config["dataset"] != "YahooR3":
        raise ValueError("YahooR3 input conversion requires a YahooR3 profile")
    manifest = read(source_root / "bundle.json")
    if manifest.get("status") != "shared_inputs_frozen" or manifest.get("frozen_before_fit") is not True:
        raise ValueError("expected a frozen YahooR3 shared input manifest")
    sources = ("offline.npz", "representation.npz", "pairs.npz", "lineage.npz", "dev_cases.npz")
    for name in sources:
        if digest(source_root / name) != manifest["assets"][name]["sha256"]:
            raise ValueError(f"YahooR3 source asset differs from frozen manifest: {name}")
    if digest(matrix) != manifest["source"]["matrix_sha256"]:
        raise ValueError("YahooR3 ASCII matrix differs from the retained input")
    original, features, pairs, lineage, native_cases = (_npz(source_root / name) for name in sources)
    roles = _source_cohorts(cohorts)
    width = int(original["n_items"]) - 2
    if int(original["n_users"]) != 1 + sum(map(len, roles.values())):
        raise ValueError("YahooR3 cohort universe differs from the original model vocabulary")
    users = native_cases["encoded_user_ids"].astype(np.int64) + 1
    if users.tolist() != roles["development_users"] or not np.array_equal(users, native_cases["raw_user_ids"]):
        raise ValueError("YahooR3 saved reset cases differ from the frozen DEV cohort")
    expected_contexts = np.random.default_rng(config["evaluation"]["context_seed"]).integers(
        0, width, size=len(users), dtype=np.int32
    )
    if not np.array_equal(expected_contexts, native_cases["initial_context_item_columns"]):
        raise ValueError("YahooR3 saved DEV contexts differ from the configured reset seed")
    raw_ratings = np.rint(original["feedback"] * np.float32(5)).astype(np.float32)
    mask = original["mask"]
    if not np.array_equal((raw_ratings / np.float32(5))[mask], original["feedback"][mask]):
        raise ValueError("cannot restore exact YahooR3 native ratings from the frozen model feedback")
    labels = preference_scores(raw_ratings, mask)
    if not np.array_equal(labels, original["positive_count"]):
        raise ValueError("YahooR3 label scores differ from the original float32 trajectory sums")
    if not np.array_equal(lineage["global_episode_ids"], original["indices"]):
        raise ValueError("YahooR3 collector lineage is misaligned")
    codes = lineage["collector_codes"]
    if codes.dtype.kind not in "iu" or np.any((codes < 0) | (codes >= 3)):
        raise ValueError("unexpected YahooR3 collector codes")
    distance_values = np.load(distance, allow_pickle=False)
    if not isinstance(distance_values, np.ndarray):
        distance_values.close()
        raise ValueError("export the native distance cache as a numeric .npy array")
    worlds = read_development_rows(matrix, users, width)
    corpus_keys = ("actions", "mask", "users", "indices", "splits", "initial_context_actions",
                   "n_users", "n_items", "max_length")
    pair_arrays = {}
    for split in ("train", "validation"):
        for side in ("preferred", "rejected"):
            name = f"{split}_{side}"
            pair_arrays[name] = pairs[name]
        pair_arrays[f"{split}_targets"] = np.ones(len(pair_arrays[f"{split}_preferred"]), dtype=np.float32)

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".yahoor3-bundle-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "bundle"
        staging.mkdir()
        assets = {}

        def write_npz(name, **arrays):
            filename = f"{name}.npz"
            path = staging / filename
            np.savez_compressed(path, **arrays)
            assets[name] = dict(path=filename, sha256=digest(path), arrays={
                key: dict(shape=list(value.shape), dtype=str(value.dtype)) for key, value in arrays.items()
            })

        write_npz("corpus", **{key: original[key] for key in corpus_keys},
                  raw_users=original["users"], feedback=raw_ratings, policy_feedback=raw_ratings,
                  label_feedback=raw_ratings, preference_score=labels)
        write_npz("features", user_vectors=features["user_vectors"], item_vectors=features["item_vectors"])
        write_npz("pairs", **pair_arrays)
        write_npz("collectors", indices=original["indices"], collectors=np.asarray(["a2c", "dorl", "ppoc"])[codes])
        write_npz("dev_cases", raw_users=users, encoded_users=users,
                  initial_context_actions=native_cases["initial_context_item_columns"].astype(np.int64) + 2)
        write_npz("world", rewards=worlds, raw_users=users, raw_items=np.arange(1, width + 1, dtype=np.int64))
        write_npz("distance", distance=distance_values)
        put(staging / "cohorts.json", roles)
        assets["cohorts"] = dict(path="cohorts.json", sha256=digest(staging / "cohorts.json"), arrays={})
        data, evaluation = config["data"], config["evaluation"]
        put(staging / "bundle.json", {
            "schema_version": BUNDLE_SCHEMA, "dataset": "YahooR3", "profile_id": config["profile_id"],
            "assets": assets, "split_counts": data["split_counts"], "pair_counts": data["pair_counts"],
            "units": dict(training=data["training_units"], policy_training=data["policy_training_units"],
                          evaluation=evaluation["feedback_units"]),
            "preference": dict(rule=data["preference_rule"], stored_feedback_threshold=data["stored_feedback_threshold"],
                               validation_rule=data["validation_preference_rule"],
                               validation_stored_feedback_threshold=data["validation_stored_feedback_threshold"]),
            "evaluation": dict(phase="development", horizon=30, adapter="yahoor3_native_v1",
                               cases_asset="dev_cases", leave_window=3, leave_threshold=120.),
            "sealed_final_users": roles["final_users"],
        })
        deep_check(check(profile, staging), staging)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"bundle destination already exists: {destination}")
        staging.rename(destination)
    return {"status": "prepared", "profile": config["profile_id"], "data_root": str(destination)}


def main(argv=None):
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Convert frozen YahooR3 inputs to a PrDT4Rec data bundle.")
    parser.add_argument("--source-root", type=Path, required=True, help="original frozen shared inputs directory")
    parser.add_argument("--data-root", type=Path, required=True, help="new portable bundle directory")
    parser.add_argument("--cohorts", type=Path, required=True, help="original cohort JSON")
    parser.add_argument("--matrix", type=Path, required=True, help="official native ASCII reward matrix")
    parser.add_argument("--distance", type=Path, required=True, help="original distance cache exported as .npy")
    parser.add_argument("--config", type=Path, help="optional shared profile configuration")
    args = parser.parse_args(argv)
    result = convert_bundle(args.source_root, args.data_root, cohorts=args.cohorts, matrix=args.matrix,
                            distance=args.distance, profile=f"{args.config}#yahoor3" if args.config else "yahoor3")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
