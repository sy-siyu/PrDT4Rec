"""VirtualTB corpus, preference pairs, and bundle conversion."""
from collections import defaultdict
import argparse
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path

import numpy as np
import torch

from .. import runner
from .bundles import load_asset


def profile_digest(profile):
    return hashlib.sha256(np.ascontiguousarray(profile, dtype=np.float32).tobytes()).hexdigest()


def validate_config(config):
    model, scorer, policy = (config[key] for key in ("model", "scorer", "policy"))
    data, ev, weighting = (config[key] for key in ("data", "evaluation", "weighting"))
    if (config["dataset"] != "VirtualTB" or config["profile_id"] != "virtualtb"
            or (model["horizon"], model["observation_dim"], model["action_dim"]) != (30, 91, 27)
            or model["action_distribution"] != "fixed_gaussian" or model["action_mean"] != "tanh"
            or data["observation_transform"] != "identity" or data["action_bounds"] != [-1.0, 1.0]
            or data["training_units"] != {"mapping": "identity"}
            or data["policy_training_units"] != {"mapping": "identity"}
            or data["preference_rule"] != "mean_logged_clicks_per_page"
            or data["validation_preference_rule"] != data["preference_rule"]
            or ev["allow_final"] is not False or ev["phase"] != "development" or ev["horizon"] != 30
            or ev["batch_size"] != 1 or ev["deterministic_action"] != "mean"
            or weighting["rule"] != "global_softmax" or weighting["collector_prior"] is not None):
        raise ValueError("invalid VirtualTB observation, preference, or evaluation recipe")
    if (data["split_counts"].get("test") != 0
            or min(data["split_counts"]["train"], data["split_counts"]["validation"],
                   data["pair_counts"]["train"], data["pair_counts"]["validation"], ev["users"]) < 1
            or policy["draws_per_epoch"] > data["split_counts"]["train"]
            or scorer["pair_draws"] > data["pair_counts"]["train"]
            or policy["trajectory_draws"] != policy["epochs"] * policy["draws_per_epoch"]
            or policy["updates"] != policy["epochs"] * math.ceil(policy["draws_per_epoch"] / policy["batch_size"])
            or policy["cosine_horizon_updates"] != policy["updates"]
            or scorer["pair_draws"] != scorer["updates"] * scorer["batch_size"]
            or scorer["updates"] % scorer["checkpoint_interval_updates"] != 0):
        raise ValueError("inconsistent VirtualTB corpus or optimizer budgets")
    if (not policy["full_context_batches"] or not scorer["full_context_batches"]
            or policy["sampling_rng"] != "numpy_circular_permutation"
            or policy["loss"] != "batch_mean_N_times_global_weight_times_mean_valid_action_gaussian_NLL"
            or scorer["pair_order_rng"] != "seed0_hash_ranked_then_seeded_permutation"
            or scorer["regularizer"] != "mean_squared_paired_scores"
            or scorer["selection"] != "minimum_offline_validation_soft_target_BT_loss"
            or scorer["selection_tie"] != "earliest_checkpoint"):
        raise ValueError("unsupported VirtualTB training procedure")
    for name in ("exposure_digest_domain", "pair_order_seed_offset"):
        if name not in scorer:
            raise ValueError(f"missing VirtualTB scorer setting {name}")
    if not ev["reset_seed_domain"] or not data["pairing"]["endpoint_selection_digest_domain"]:
        raise ValueError("missing VirtualTB seed domain")


def rank_pair(domain, left, right):
    return hashlib.sha256(domain.encode("ascii") + b"\0" + left.encode() + b"\0" + right.encode()).digest()


def preference_targets(first, second, returns, lengths):
    # Exact integer cross multiplication preserves equal clicks/page.
    margin = returns[first] * lengths[second] - returns[second] * lengths[first]
    return np.where(margin == 0, 0.5, margin > 0).astype(np.float32)


def prepare_train_pairs(episode_ids, profile_hashes, rewards, mask, *, cap, domain):
    """Build the retained within-profile TRAIN pairs without outcome selection."""
    groups = defaultdict(list)
    for index, digest in enumerate(profile_hashes):
        groups[str(digest)].append(index)
    pairs = []
    for digest in sorted(groups):
        candidates = []
        members = sorted(groups[digest], key=lambda i: episode_ids[i])
        for offset, first in enumerate(members):
            for second in members[offset + 1:]:
                rank = rank_pair(domain, str(episode_ids[first]), str(episode_ids[second]))
                candidates.append((rank, first, second))
        pairs.extend((first, second) for _, first, second in sorted(candidates)[:cap])
    indices = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    first, second = indices.T
    targets = preference_targets(first, second, (rewards * mask).sum(axis=1).astype(np.int64),
                                 mask.sum(axis=1, dtype=np.int64))
    return first, second, targets


@dataclass
class Corpus:
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    mask: torch.Tensor
    episode_ids: np.ndarray

    def batch(self, indices, device):
        ids = torch.as_tensor(indices, dtype=torch.long)
        return tuple(array.index_select(0, ids).to(device)
                     for array in (self.observations, self.actions, self.rewards, self.mask))


def check(profile, root, seed):
    config_path, _, name = str(profile).rpartition("#")
    if name != "virtualtb" or type(seed) is not int or seed < 0:
        raise ValueError("VirtualTB requires the virtualtb profile and a nonnegative seed")
    config = runner.dataset_settings(config_path or None)[name]
    validate_config(config)
    manifest = runner.relative(root, "bundle.json")
    bundle = runner.read(manifest)
    if (bundle.get("schema_version", runner.BUNDLE_SCHEMA) != runner.BUNDLE_SCHEMA
            or bundle["profile_id"] != name or bundle["dataset"] != config["dataset"]
            or bundle["split_counts"] != config["data"]["split_counts"]
            or bundle["pair_counts"] != config["data"]["pair_counts"]
            or bundle["preference"]["rule"] != config["data"]["preference_rule"]
            or bundle["preference"]["validation_rule"] != config["data"]["validation_preference_rule"]
            or bundle["evaluation"] != {"phase": "development", "horizon": 30,
                                        "adapter": "virtualtb_native_v1", "cases_asset": "dev_cases"}
            or bundle["units"] != {key: {"mapping": "identity"}
                                   for key in ("training", "policy_training", "evaluation")}
            or type(bundle["sealed_final_users"]) is not int or bundle["sealed_final_users"] < 0):
        raise ValueError("VirtualTB bundle manifest differs from the configured recipe")
    assets = bundle["assets"]
    if set(assets) != {"corpus", "pairs", "dev_cases"}:
        raise ValueError("VirtualTB bundle needs corpus, pairs and DEV cases only")
    for name, asset in assets.items():
        path = runner.relative(root, asset["path"])
        if set(asset) != {"path", "sha256", "arrays"} or runner.digest(path) != asset["sha256"]:
            raise ValueError(f"missing or changed VirtualTB asset: {name}")
    implementation = ("data/bundles.py", "data/virtualtb.py", "environments/virtualtb.py",
                      "environments/virtualtb_native.py", "continuous.py", "models.py",
                      "runner.py", "training.py", "objective.py")
    run = dict(version=runner.VERSION, config=config, bundle=bundle, seed=seed,
               bundle_sha256=runner.digest(manifest),
               implementation_sha256=runner._implementation_hashes(implementation))
    run["identity"] = runner.identity(run)
    return run


def load_data(run, root):
    config, bundle = run["config"], run["bundle"]
    corpus = load_asset(root, bundle, "corpus")
    pairs = load_asset(root, bundle, "pairs")
    cases = load_asset(root, bundle, "dev_cases")
    train_n = config["data"]["split_counts"]["train"]
    n = train_n + config["data"]["split_counts"]["validation"]
    shapes = {"observations": (n, 30, 91), "actions": (n, 30, 27),
              "rewards": (n, 30), "mask": (n, 30)}
    for name, shape in shapes.items():
        values = corpus[name]
        dtype = np.bool_ if name == "mask" else np.float32
        if values.shape != shape or values.dtype != dtype or not np.isfinite(values).all():
            raise ValueError(f"invalid VirtualTB corpus {name}")
    actions, rewards, mask = (corpus[name] for name in ("actions", "rewards", "mask"))
    if (not mask[:, 0].all() or (mask[:, 1:] & ~mask[:, :-1]).any()
            or np.any(np.abs(actions[mask]) > 1) or np.any((rewards[mask] < 0) | (rewards[mask] > 10))
            or np.any(rewards[mask] != np.floor(rewards[mask]))):
        raise ValueError("invalid VirtualTB action, click or terminal-mask geometry")
    if "train_steps" in config["data"] and int(mask[:train_n].sum()) != config["data"]["train_steps"]:
        raise ValueError("VirtualTB TRAIN transition budget differs")
    ids, hashes = corpus["episode_ids"], corpus["profile_sha256"]
    if (ids.shape != (n,) or ids.dtype.kind != "U" or len(set(ids)) != n or np.any(ids == "")
            or hashes.shape != (n,) or hashes.dtype.kind != "U"
            or any(profile_digest(p) != h for p, h in zip(corpus["observations"][:, 0, :88], hashes))):
        raise ValueError("invalid VirtualTB episode IDs or profile fingerprints")
    dev, dev_hashes = cases["profiles"], cases["profile_sha256"]
    dev_count = config["evaluation"]["users"]
    if (dev.shape != (dev_count, 88) or dev.dtype != np.float32 or not np.isfinite(dev).all()
            or dev_hashes.shape != (dev_count,) or dev_hashes.dtype.kind != "U"
            or len(set(dev_hashes)) != dev_count
            or any(profile_digest(p) != h for p, h in zip(dev, dev_hashes))
            or set(dev_hashes) & set(hashes) or set(hashes[:train_n]) & set(hashes[train_n:])):
        raise ValueError("invalid or overlapping VirtualTB TRAIN, offline-VAL and DEV profiles")
    result = {}
    for split, start, stop in (("train", 0, train_n), ("validation", train_n, n)):
        first, second, target = (pairs[f"{split}_{key}"] for key in ("preferred", "rejected", "targets"))
        count = config["data"]["pair_counts"][split]
        if (first.shape != (count,) or second.shape != (count,) or target.shape != (count,)
                or first.dtype != np.int64 or second.dtype != np.int64 or target.dtype != np.float32
                or np.any((first < 0) | (second < 0) | (first >= stop - start) | (second >= stop - start))
                or np.any(first == second) or np.any(hashes[start + first] != hashes[start + second])):
            raise ValueError(f"invalid VirtualTB {split} pair endpoints")
        if not np.array_equal(target, preference_targets(
                first, second, (rewards[start:stop] * mask[start:stop]).sum(axis=1).astype(np.int64),
                mask[start:stop].sum(axis=1, dtype=np.int64))):
            raise ValueError(f"VirtualTB {split} targets must compare clicks/page")
        if split == "train":
            pairing = config["data"]["pairing"]
            expected = prepare_train_pairs(ids[:stop], hashes[:stop], rewards[:stop], mask[:stop],
                                           cap=pairing["cap_per_group"],
                                           domain=pairing["endpoint_selection_digest_domain"])
            if any(not np.array_equal(actual, reference)
                   for actual, reference in zip((first, second, target), expected)):
                raise ValueError("VirtualTB TRAIN pair endpoints or order differ from the retained rule")
        tensors = [torch.from_numpy(corpus[key][start:stop])
                   for key in ("observations", "actions", "rewards", "mask")]
        result[split] = Corpus(*tensors, ids[start:stop]), (first, second, target)
    return result, cases


def export_bundle(train_path, validation_path, development_path, output, config_path=None):
    """Convert prepared native artifacts without copying FINAL or collector state."""
    config = runner.dataset_settings(config_path)["virtualtb"]
    validate_config(config)
    arrays = ("observations", "actions", "rewards", "mask", "episode_ids")
    with (np.load(train_path, allow_pickle=False) as train_data,
          np.load(validation_path, allow_pickle=False) as val_data,
          np.load(development_path, allow_pickle=False) as dev_data):
        train_hashes = np.asarray([profile_digest(p) for p in train_data["observations"][:, 0, :88]])
        val_hashes = np.asarray([profile_digest(p) for p in val_data["observations"][:, 0, :88]])
        if "profile_sha256" in train_data and not np.array_equal(train_hashes, train_data["profile_sha256"]):
            raise ValueError("TRAIN observation does not match its recorded VirtualTB profile")
        corpus = {key: np.concatenate((train_data[key], val_data[key])) for key in arrays}
        corpus["profile_sha256"] = np.concatenate((train_hashes, val_hashes))
        n = len(train_hashes)
        pairing = config["data"]["pairing"]
        first, second, targets = prepare_train_pairs(corpus["episode_ids"][:n], train_hashes,
            corpus["rewards"][:n], corpus["mask"][:n], cap=pairing["cap_per_group"],
            domain=pairing["endpoint_selection_digest_domain"])
        pairs = dict(train_preferred=first, train_rejected=second, train_targets=targets,
                     validation_preferred=val_data["first"], validation_rejected=val_data["second"],
                     validation_targets=val_data["targets"])
        # Deliberately access only the named development arrays even when
        # the caller's source archive contains other cohorts.
        cases = dict(profiles=dev_data["development_profiles"], profile_sha256=dev_data["development_sha256"])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    assets = {}
    for name, payload in (("corpus", corpus), ("pairs", pairs), ("dev_cases", cases)):
        path = output / f"{name}.npz"
        np.savez_compressed(path, **payload)
        assets[name] = {"path": path.name, "sha256": runner.digest(path),
                        "arrays": {key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                                   for key, value in payload.items()}}
    bundle = dict(schema_version=runner.BUNDLE_SCHEMA, dataset="VirtualTB", profile_id="virtualtb", assets=assets,
        split_counts=config["data"]["split_counts"], pair_counts=config["data"]["pair_counts"],
        units={key: {"mapping": "identity"} for key in ("training", "policy_training", "evaluation")},
        preference={"rule": "mean_logged_clicks_per_page", "validation_rule": "mean_logged_clicks_per_page"},
        evaluation={"phase": "development", "horizon": 30, "adapter": "virtualtb_native_v1", "cases_asset": "dev_cases"},
        sealed_final_users=config["evaluation"].get("sealed_final_users", 671))
    runner.put(output / "bundle.json", bundle)
    profile = f"{config_path}#virtualtb" if config_path else "virtualtb"
    run = check(profile, output, seed=0)
    load_data(run, output)
    return dict(status="preflight_pass", profile="virtualtb", identity=run["identity"],
                writes=False, models_constructed=False, payload_validated=True,
                native_environment_reset_or_step=False)


def main():
    import json
    parser = argparse.ArgumentParser(description=export_bundle.__doc__)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--validation", required=True, type=Path)
    parser.add_argument("--development", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    print(json.dumps(export_bundle(args.train, args.validation, args.development,
                                   args.output_root, args.config), sort_keys=True))


if __name__ == "__main__":
    main()
