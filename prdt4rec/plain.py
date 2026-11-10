"""Train and evaluate PrDT4Rec using shared settings and dataset overrides."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sysconfig

VERSION = "portable-plain-prdt-v1"
CONFIG_SCHEMA = "plain-prdt-config-v1"
PUBLIC_SCHEMA = "plain-prdt-config-v2"
BUNDLE_SCHEMA = "plain-prdt-bundle-v1"
PROFILES_SCHEMA = "plain-prdt-profiles-v1"
PROJECT = Path(__file__).resolve().parents[1]
PUBLIC_CONFIG = PROJECT / "configs/prdt.json"
if not PUBLIC_CONFIG.exists(): PUBLIC_CONFIG = Path(sysconfig.get_path("data")) / "share/prdt4rec/prdt.json"
REGISTRY = PROJECT / "configs/prdt_profiles"
CONFIGS = PROJECT / "configs/plain_prdt"
if not REGISTRY.exists(): REGISTRY = Path(sysconfig.get_path("data")) / "share/prdt4rec/registry"
if not CONFIGS.exists(): CONFIGS = Path(sysconfig.get_path("data")) / "share/prdt4rec/profiles"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def put(path, value, *, immutable=False):
    path = Path(path)
    encoded = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if immutable and path.exists():
        if read(path) != value:
            raise ValueError(f"immutable output collision: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(encoded); temporary.replace(path)


def save(path, payload):
    import torch
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary); temporary.replace(path)


def relative(root, name):
    """Reject absolute paths, parent escapes and symlinks outside the bundle."""
    root = Path(root).resolve(); name = Path(name)
    if name.is_absolute() or ".." in name.parts:
        raise ValueError("bundle asset paths must be root-relative without parent traversal")
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError("bundle symlink escapes its data root")
    return path


def merge_settings(defaults, overrides):
    """Recursively merge dataset overrides without modifying shared defaults."""
    result = deepcopy(defaults)
    for key, value in overrides.items():
        result[key] = (merge_settings(result[key], value)
                       if isinstance(result.get(key), dict) and isinstance(value, dict)
                       else deepcopy(value))
    return result


def dataset_settings(config_file=None):
    """Resolve parameters for every dataset, including dedicated experiment runners."""
    document = read(config_file or PUBLIC_CONFIG)
    if (set(document) - {"schema_version"} != {"defaults", "datasets"}
            or document.get("schema_version", PROFILES_SCHEMA) != PROFILES_SCHEMA
            or not isinstance(document["defaults"], dict)
            or not isinstance(document["datasets"], dict) or not document["datasets"]):
        raise ValueError("expected a configuration file containing shared defaults and dataset overrides")
    profiles = {}
    for name, overrides in document["datasets"].items():
        if not isinstance(overrides, dict):
            raise ValueError("dataset overrides must be objects")
        config = merge_settings(document["defaults"], overrides)
        if config.get("profile_id") != name:
            raise ValueError("dataset key must match its public profile_id")
        profiles[name] = config
    return profiles


# These recipes currently execute through their dataset-specific experiment runners.
DEDICATED_DATASETS = frozenset({"yahoor3", "virtualtb", "kuairand"})


def public_profiles(config_file=None):
    """Return profiles supported by the prepared-bundle launcher."""
    profiles = {}
    for name, config in dataset_settings(config_file).items():
        if name in DEDICATED_DATASETS:
            continue
        # Format metadata stays internal; preserve historical resolved identities.
        config.setdefault("schema_version", PUBLIC_SCHEMA)
        validate_config(config)
        if config["schema_version"] != PUBLIC_SCHEMA:
            raise ValueError("dataset key must match its public profile_id")
        profiles[name] = config
    return profiles


def list_profiles(config_file=None):
    return [dict(profile=name, dataset=c["dataset"], runnable_recipe=True)
            for name, c in sorted(public_profiles(config_file).items())]


def load_config(profile):
    name = str(profile)
    if "#" in name:
        config_file, dataset = name.rsplit("#", 1)
        if dataset in DEDICATED_DATASETS:
            raise ValueError(f"{dataset} uses its dedicated experiment runner; see docs/DATA.md")
        profiles = public_profiles(config_file)
        if dataset not in profiles: raise ValueError(f"unknown dataset profile: {dataset}")
        return profiles[dataset]
    source = Path(profile)
    if name in DEDICATED_DATASETS:
        raise ValueError(f"{name} uses its dedicated experiment runner; see docs/DATA.md")
    if not source.suffix and source.name == name:
        profiles = public_profiles()
        if name in profiles: return profiles[name]
    path = source if source.suffix == ".json" else CONFIGS / f"{profile}.json"
    # Previously published per-dataset paths resolve to the same merged recipe.
    if not path.exists() and path.parent.resolve() == CONFIGS.resolve():
        profiles = public_profiles()
        if path.stem in profiles: return profiles[path.stem]
    config = read(path)
    config.setdefault("schema_version", PUBLIC_SCHEMA)
    validate_config(config)
    if config["schema_version"] == CONFIG_SCHEMA and config["dataset"] != "fixture":
        name = config["profile_id"]
        registry = read(REGISTRY / "index.json")["profiles"]
        if name not in registry or config["registry_record_sha256"] != digest(REGISTRY / registry[name]):
            raise ValueError("runnable recipe does not match its immutable provenance record")
        if config != read(CONFIGS / f"{name}.json"):
            raise ValueError("native recipe differs from its versioned runnable configuration")
    return config


def validate_config(c):
    public = c.get("schema_version") == PUBLIC_SCHEMA
    expected = {"schema_version", "profile_id", "dataset", "model", "policy", "scorer", "weighting", "data", "evaluation"}
    if not public:
        expected |= {"registry_record_sha256", "status", "runnable", "blocked_reason"}
    if c.keys() != expected or c["schema_version"] not in (CONFIG_SCHEMA, PUBLIC_SCHEMA):
        raise ValueError("unknown configuration format or missing required settings")
    if not public and not c["runnable"]:
        raise ValueError(f"profile is blocked: {c['blocked_reason']}")
    m, p, s, w, e = (c[k] for k in ("model", "policy", "scorer", "weighting", "evaluation"))
    keys = [
        (m, {"hidden_dim", "layers", "heads", "dropout", "conv_window", "feature_dim", "horizon"}),
        (p, {"epochs", "draws_per_epoch", "batch_size", "learning_rate", "minimum_learning_rate", "cosine_horizon_updates", "weight_decay", "gradient_clip", "loss", "checkpoint_interval_updates", "sampling_rng", "full_context_batches"}),
        (s, {"hidden_dim", "layers", "heads", "dropout", "epochs", "batch_size", "learning_rate", "weight_decay", "lambda_score", "regularizer_seed_offset", "checkpoint_interval_updates", "pair_order_rng", "full_context_batches"}),
        (w, {"rule", "beta", "collector_prior", "precision"}),
        (e, {"phase", "users", "horizon", "batch_size", "reset_block_size", "selection", "allow_final"}),
        (c["data"], {"split_counts", "pair_counts", "preference_rule", "stored_feedback_threshold", "training_units"}
         | ({"validation_preference_rule", "validation_stored_feedback_threshold", "policy_training_units"} if public else set()))]
    if any(obj.keys() != names for obj, names in keys):
        raise ValueError("unknown or missing executable recipe fields")
    for obj, names in ((m, ("hidden_dim", "layers", "heads", "conv_window", "feature_dim", "horizon")),
                       (p, ("epochs", "draws_per_epoch", "batch_size", "cosine_horizon_updates", "checkpoint_interval_updates")),
                       (s, ("hidden_dim", "layers", "heads", "epochs", "batch_size", "checkpoint_interval_updates")),
                       (e, ("users", "horizon", "batch_size", "reset_block_size"))):
        if any(type(obj[k]) is not int or obj[k] <= 0 for k in names):
            raise ValueError("recipe dimensions and budgets must be positive integers")
    updates = p["epochs"] * math.ceil(p["draws_per_epoch"] / p["batch_size"])
    if (updates > p["cosine_horizon_updates"] or p["cosine_horizon_updates"] < 2
            or not 0 < p["minimum_learning_rate"] <= p["learning_rate"]
            or p["loss"] != "mean_valid_action_NLL_per_trajectory_then_batch_mean"
            or isinstance(p["gradient_clip"], bool) or p["gradient_clip"] != 1. or s["weight_decay"] != 0.
            or s["regularizer_seed_offset"] != 991 or m["hidden_dim"] % m["heads"]
            or s["hidden_dim"] % s["heads"] or not 0 <= m["dropout"] < 1 or not 0 <= s["dropout"] < 1
            or e["phase"] != "development" or e["allow_final"] is not False
            or not 0 < e["horizon"] <= m["horizon"]
            or e["selection"] != "fixed_endpoint_no_checkpoint_or_target_search"):
        raise ValueError("invalid fixed plain policy/scorer/schedule/DEV recipe")
    if (p["sampling_rng"] not in ("torch_weighted_dataloader", "numpy_choice_persistent")
            or s["pair_order_rng"] not in (("torch_private_dataloader", "numpy_epoch_seed", "torch_randperm") if public else ("torch_private_dataloader", "numpy_epoch_seed"))
            or type(p["full_context_batches"]) is not bool or type(s["full_context_batches"]) is not bool):
        raise ValueError("declare the native sampling chronology and scorer/policy padding explicitly")
    validate_units(c["data"]["training_units"])
    if public:
        validate_units(c["data"]["policy_training_units"])
        for rule_key, threshold_key in (("preference_rule", "stored_feedback_threshold"),
                                        ("validation_preference_rule", "validation_stored_feedback_threshold")):
            rule, threshold = c["data"][rule_key], c["data"][threshold_key]
            if rule == "cumulative_logged_feedback":
                if threshold is not None:
                    raise ValueError("cumulative labels have no count threshold")
            elif not isinstance(threshold, (int, float)) or isinstance(threshold, bool) or not math.isfinite(threshold):
                raise ValueError("count labels need an explicit finite threshold")
    if (w["rule"] not in ("global_softmax", "fixed_collector_prior") or not math.isfinite(w["beta"]) or w["beta"] < 0
            or w["precision"] not in ("saved_global_float32_then_TRAIN_float64_normalized", "native_float64_softmax")):
        raise ValueError("declare the global softmax or fixed collector prior and its precision explicitly")
    if w["rule"] == "fixed_collector_prior":
        prior = w["collector_prior"]
        if not isinstance(prior, dict) or not prior or any(v <= 0 for v in prior.values()) or not math.isclose(sum(prior.values()), 1., abs_tol=1e-12):
            raise ValueError("collector prior must be a positive normalized fixed map")
    elif w["collector_prior"] is not None:
        raise ValueError("global sampling does not use a collector prior")


def check(profile, data_root, *, seed=0):
    """Hash/metadata preflight; no NN constructor, training or environment reset."""
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    config = load_config(profile); root = Path(data_root).resolve()
    bundle_path = root / "bundle.json"
    if not bundle_path.exists():
        raise ValueError("missing portable bundle.json; native assets must be exported explicitly, never inferred from historical absolute paths")
    b = read(bundle_path)
    b.setdefault("schema_version", BUNDLE_SCHEMA)
    adapter_aliases = {"movielens": "movielens_native_v1", "kuairec": "kuairec_native_v1", "coat": "coat_native_v1"}
    evaluation = b.get("evaluation", {})
    if isinstance(evaluation, dict) and evaluation.get("adapter") in adapter_aliases:
        evaluation["adapter"] = adapter_aliases[evaluation["adapter"]]
    required = {"schema_version", "dataset", "profile_id", "assets", "split_counts", "pair_counts", "units", "preference", "evaluation", "sealed_final_users"}
    optional = {"score_cache_seed"}
    if not required <= b.keys() or b.keys() - required - optional or b["schema_version"] != BUNDLE_SCHEMA:
        raise ValueError("invalid dataset bundle manifest")
    if b["dataset"] != config["dataset"] or b["profile_id"] != config["profile_id"]:
        raise ValueError("bundle dataset/profile differs from the runnable recipe")
    if b["split_counts"] != config["data"]["split_counts"] or b["pair_counts"] != config["data"]["pair_counts"]:
        raise ValueError("frozen corpus/pair budgets differ from the profile")
    if b["split_counts"].get("test") != 0 or b["preference"].get("rule") != config["data"]["preference_rule"]:
        raise ValueError("bundle must exclude final episodes and declare the frozen native preference rule")
    if (b["evaluation"].get("phase") != "development"
            or b["evaluation"].get("horizon") != config["evaluation"]["horizon"]
            or "cases_asset" not in b["evaluation"]):
        raise ValueError("bundle evaluation requires its fixed DEV cases and horizon")
    native = {"MovieLens": ("movielens_native_v1", {"world", "distance"}, 7, 75),
              "KuaiRec": ("kuairec_native_v1", {"world", "categories", "response_transform"}, 9, 1),
              "Coat": ("coat_native_v1", {"world", "distance"}, 7, 6)}
    if b["dataset"] in native:
        adapter, assets, window, threshold = native[b["dataset"]]
        if (b["evaluation"].get("adapter") != adapter or not assets <= b["assets"].keys()
                or b["evaluation"].get("leave_window") != window or b["evaluation"].get("leave_threshold") != threshold):
            raise ValueError("missing explicit native environment assets/window/threshold; no fabricated fallback")
    elif b["dataset"] != "fixture":
        raise ValueError("no validated adapter/profile for this environment")
    public = config["schema_version"] == PUBLIC_SCHEMA
    unit_keys = {"training", "evaluation"} | ({"policy_training"} if public else set())
    if set(b["units"]) != unit_keys:
        raise ValueError("declare separate scorer, policy and rollout feedback units")
    if public and (b["units"]["policy_training"] != config["data"]["policy_training_units"]
                   or b["preference"].get("validation_rule") != config["data"]["validation_preference_rule"]
                   or b["preference"].get("validation_stored_feedback_threshold") != config["data"]["validation_stored_feedback_threshold"]):
        raise ValueError("policy units or validation preference semantics differ from the profile")
    if public and "cohorts" not in b["assets"]:
        raise ValueError("public bundle requires a declared cohort manifest")
    if (b["units"]["training"] != config["data"]["training_units"]
            or b["preference"].get("stored_feedback_threshold") != config["data"]["stored_feedback_threshold"]):
        raise ValueError("TRAIN feedback units or frozen preference threshold differ from the native profile")
    if not {"corpus", "features", "pairs", "collectors", b["evaluation"]["cases_asset"]} <= b["assets"].keys():
        raise ValueError("missing corpus/features/pairs/collector/DEV assets")
    for name, asset in b["assets"].items():
        if asset.keys() != {"path", "sha256", "arrays"} or not isinstance(asset["arrays"], dict):
            raise ValueError(f"asset {name} needs path/sha256/arrays schema")
        path = relative(root, asset["path"])
        if not path.is_file() or digest(path) != asset["sha256"]:
            raise ValueError(f"missing or changed portable asset {name}")
    if ("scores" in b["assets"] or "weights" in b["assets"]) and (type(b.get("score_cache_seed")) is not int or b.get("score_cache_seed") != seed):
        raise ValueError("cached scorer assets must belong to this exact training seed")
    numerical_sources = ["plain.py", "plain_environments.py", "data.py", "models.py", "training.py", "objective.py", "features.py", "loss.py"]
    if config["dataset"] == "KuaiRec":
        numerical_sources += ["kuairec.py", "kuairec_models.py"]
    summary = {"version": VERSION, "config": config, "bundle": b, "bundle_sha256": digest(bundle_path), "seed": seed,
               "implementation_sha256": {name: digest(Path(__file__).parent / name) for name in numerical_sources}}
    summary["identity"] = identity(summary)
    return summary


def load_asset(root, b, name):
    import numpy as np
    asset = b["assets"][name]
    with np.load(relative(root, asset["path"]), allow_pickle=False) as archive:
        if set(archive.files) != set(asset["arrays"]):
            raise ValueError(f"asset {name} array schema differs")
        arrays = {key: archive[key].copy() for key in archive.files}
    for key, value in arrays.items():
        desc = asset["arrays"][key]
        if desc.keys() != {"shape", "dtype"} or list(value.shape) != desc["shape"] or str(value.dtype) != desc["dtype"]:
            raise ValueError(f"asset {name}/{key} shape or dtype differs")
    return arrays


def load_data(run, root):
    """Decode only the supplied TRAIN/offline-VAL assets after strict preflight."""
    import numpy as np
    from .data import Trajectory, TrajectoryCorpus
    b, c = run["bundle"], run["config"]
    a = load_asset(root, b, "corpus")
    public = c["schema_version"] == PUBLIC_SCHEMA
    required = {"actions", "feedback", "mask", "users", "indices", "splits", "initial_context_actions", "n_users", "n_items", "max_length"}
    required |= {"raw_users", "policy_feedback", "preference_score"} if public else {"positive_count"}
    if not required <= a.keys():
        raise ValueError("corpus requires canonical actions/feedback/mask/users/globalIDs/splits/labels/reset and dimensions")
    actions, feedback, mask = a["actions"], a["feedback"], a["mask"]
    n, horizon = actions.shape
    if (mask.dtype != np.bool_ or feedback.shape != actions.shape or mask.shape != actions.shape
            or actions.dtype.kind not in "iu" or horizon != c["model"]["horizon"]
            or int(a["max_length"]) != horizon or not np.all(np.isin(a["splits"], [0, 1]))
            or len(np.unique(a["indices"])) != n or np.any(a["indices"] < 0)
            or not np.all(mask == (np.arange(horizon)[None, :] < mask.sum(1)[:, None]))
            or np.any(mask.sum(1) == 0) or not np.isfinite(feedback[mask]).all()
            or np.any(actions[mask] < 2) or np.any(actions[mask] >= int(a["n_items"]))):
        raise ValueError("invalid complete, contiguous, finite TRAIN/VAL corpus arrays")
    for name in ("users", "indices", "splits", "initial_context_actions") + (("raw_users", "preference_score") if public else ("positive_count",)):
        if a[name].shape != (n,):
            raise ValueError("corpus row metadata must be aligned")
    user_lower = 0 if c["dataset"] == "KuaiRec" else 1
    if (np.any(a["users"] < user_lower) or np.any(a["users"] >= int(a["n_users"]))
            or np.any((a["initial_context_actions"] != 0) & ((a["initial_context_actions"] < 2) | (a["initial_context_actions"] >= int(a["n_items"]))))):
        raise ValueError("invalid encoded TRAIN users/reset contexts")
    counts = {"train": int((a["splits"] == 0).sum()), "validation": int((a["splits"] == 1).sum()), "test": 0}
    if counts != b["split_counts"]:
        raise ValueError("actual corpus splits differ from the declared frozen counts")
    if public:
        policy_feedback = a["policy_feedback"]
        if policy_feedback.shape != actions.shape or not np.isfinite(policy_feedback[mask]).all():
            raise ValueError("policy feedback must be aligned and finite on executed actions")
        scores = {}
        for split, rule, threshold in (
            ("train", c["data"]["preference_rule"], c["data"]["stored_feedback_threshold"]),
            ("validation", c["data"]["validation_preference_rule"], c["data"]["validation_stored_feedback_threshold"])):
            if rule == "cumulative_logged_feedback":
                label = a.get("label_feedback")
                if label is None or label.shape != actions.shape or not np.isfinite(label[mask]).all():
                    raise ValueError("cumulative labels require separate finite label_feedback on all executed actions")
                scores[split] = np.where(mask, label, 0.).sum(axis=1, dtype=np.float64)
            else:
                scores[split] = ((feedback > threshold) & mask).sum(axis=1).astype(np.float64)
        score = np.where(a["splits"] == 0, scores["train"], scores["validation"])
        if not np.isfinite(a["preference_score"]).all() or not np.allclose(a["preference_score"], score, rtol=1e-6, atol=1e-6):
            raise ValueError("preference_score disagrees with executed feedback and split-specific label rule")
    else:
        threshold = b["preference"].get("stored_feedback_threshold")
        if not isinstance(threshold, (int, float)) or not math.isfinite(threshold):
            raise ValueError("frozen preference requires its explicit canonical stored-feedback threshold")
        if not np.array_equal(((feedback > threshold) & mask).sum(1), a["positive_count"]):
            raise ValueError("stored positive counts disagree with the frozen label rule")
        score = a["positive_count"]
        policy_feedback = feedback
    trajectories = []
    for i in range(n):
        length = int(mask[i].sum())
        trajectories.append(Trajectory(int(a["indices"][i]), int(a["users"][i]), actions[i, :length].copy(),
            feedback[i, :length].copy(), np.arange(length), np.arange(length) + i * horizon,
            "train" if a["splits"][i] == 0 else "validation", float(score[i]), i,
            int(a["initial_context_actions"][i])))
    corpus = TrajectoryCorpus(trajectories, int(a["n_users"]), int(a["n_items"]), horizon)
    from dataclasses import replace
    policy_corpus = TrajectoryCorpus([replace(t, feedback=policy_feedback[i, :t.length].copy())
                                     for i, t in enumerate(trajectories)], corpus.n_users, corpus.n_items, horizon)
    features = load_asset(root, b, "features")
    width = c["model"]["feature_dim"]
    required_features = {"item_vectors"} if c["dataset"] == "KuaiRec" else {"user_vectors", "item_vectors"}
    if (set(features) != required_features
            or ("user_vectors" in features and features["user_vectors"].shape != (corpus.n_users, width))
            or features["item_vectors"].shape != (corpus.n_items - 2, width)
            or any(not np.isfinite(x).all() for x in features.values())
            or ("user_vectors" in features and np.any(features["user_vectors"][0] != 0))):
        raise ValueError("frozen feature lookups must match the declared shared projection width and PAD row")
    pairs = load_asset(root, b, "pairs")
    pair_keys = {f"{split}_{side}" for split in ("train", "validation") for side in ("preferred", "rejected")}
    if public: pair_keys |= {"train_targets", "validation_targets"}
    if set(pairs) != pair_keys:
        raise ValueError("pairs require fixed endpoints and explicit public soft targets")
    for split in ("train", "validation"):
        hi, lo = (pairs[f"{split}_{k}"] for k in ("preferred", "rejected"))
        if hi.shape != lo.shape or hi.shape != (b["pair_counts"][split],) or hi.dtype.kind not in "iu" or lo.dtype.kind not in "iu":
            raise ValueError("frozen preference endpoint counts/dtypes differ")
        target = pairs.get(f"{split}_targets", np.ones(len(hi)))
        if target.shape != hi.shape or not np.isfinite(target).all() or not np.all(np.isin(target, [0., .5, 1.])):
            raise ValueError("preference targets must be aligned finite 0/0.5/1 values")
        for high, low, value in zip(hi, lo, target):
            x, y = corpus.get(high), corpus.get(low)
            expected = 1. if x.positive_count > y.positive_count else 0. if x.positive_count < y.positive_count else .5
            if high == low or x.split != split or y.split != split or x.user_id != y.user_id or value != expected:
                raise ValueError("pair crosses user/split, compares itself, or violates its feedback target")
    collectors = load_asset(root, b, "collectors")
    if set(collectors) != {"indices", "collectors"} or not np.array_equal(collectors["indices"], a["indices"]) or collectors["collectors"].shape != (n,):
        raise ValueError("collector labels must align exactly with canonical global episode IDs")
    cases = load_asset(root, b, b["evaluation"]["cases_asset"])
    needed = {"raw_users", "encoded_users", "initial_context_actions"}
    if (set(cases) != needed or any(x.shape != (c["evaluation"]["users"],) or x.dtype.kind not in "iu" for x in cases.values())
            or len(np.unique(cases["raw_users"])) != len(cases["raw_users"])
            or set(map(int, cases["raw_users"])) & set(b["sealed_final_users"])
            or np.any(cases["encoded_users"] < user_lower) or np.any(cases["encoded_users"] >= corpus.n_users)
            or np.any(cases["initial_context_actions"] < 2) or np.any(cases["initial_context_actions"] >= corpus.n_items)):
        raise ValueError("invalid or final-overlapping saved development cases")
    if public:
        _validate_cohorts(b, root, a, cases)
    elif c["dataset"] == "MovieLens":
        if not np.array_equal(cases["raw_users"], np.arange(3701, 4701)) or not set(range(4701, 5701)) <= set(b["sealed_final_users"]):
            raise ValueError("MovieLens requires the exact retained DEV3701..4700 and sealed FINAL4701..5700")
    elif c["dataset"] == "KuaiRec":
        frozen = read(REGISTRY / "source_snapshots/kuairec_data_contract.json")["development_and_final_cohorts"]["development"]
        asset = b["assets"].get("dev_cohort_contract")
        if not asset or asset["sha256"] != frozen["contexts_file_sha256"]:
            raise ValueError("KuaiRec requires the immutable original DEV reset/cohort contract, not a relabeled roster")
        roster = read(relative(root, asset["path"]))
        if (roster.get("role") != "development" or roster.get("final_users_read") is not False
                or [int(x["raw_user"]) for x in roster["contexts"]] != cases["raw_users"].tolist()
                or [int(x["context_column"]) + 2 for x in roster["contexts"]] != cases["initial_context_actions"].tolist()
                or [int(x["encoded_user"]) for x in roster["contexts"]] != cases["encoded_users"].tolist()):
            raise ValueError("KuaiRec DEV users/reset contexts differ from the frozen native cohort")
    return dict(corpus=corpus, policy_corpus=policy_corpus, features=features, pairs=pairs,
                collectors=dict(zip(map(int, collectors["indices"]), map(str, collectors["collectors"]))), cases=cases)


def _validate_cohorts(bundle, root, arrays, cases):
    """Bind supplied raw-user roles; TRAIN/VAL may share users, never episodes."""
    import numpy as np
    asset = bundle["assets"]["cohorts"]
    if asset["arrays"] != {}:
        raise ValueError("cohorts must be a JSON manifest")
    roles = read(relative(root, asset["path"]))
    names = ("train_users", "validation_users", "development_users", "final_users")
    if set(roles) != set(names) or any(not isinstance(roles[n], list) or
            any(type(u) is not int or u < 0 for u in roles[n]) or len(set(roles[n])) != len(roles[n]) for n in names):
        raise ValueError("invalid raw-user cohort roles")
    sets = {n: set(roles[n]) for n in names}
    fitting = sets["train_users"] | sets["validation_users"]
    if (not sets["final_users"] or fitting & sets["development_users"] or
            (fitting | sets["development_users"]) & sets["final_users"] or
            sets["final_users"] != set(bundle["sealed_final_users"]) or
            cases["raw_users"].tolist() != roles["development_users"]):
        raise ValueError("cohort roles overlap DEV/FINAL or do not bind declared saved cases")
    if arrays["raw_users"].dtype.kind not in "iu":
        raise ValueError("raw corpus users must be integer IDs")
    for code, name in ((0, "train_users"), (1, "validation_users")):
        if not set(map(int, arrays["raw_users"][arrays["splits"] == code])) <= sets[name]:
            raise ValueError("corpus user appears outside its declared split role")
    # Prevent distinct raw users from being silently collapsed onto one model ID.
    raw = np.concatenate((arrays["raw_users"], cases["raw_users"]))
    encoded = np.concatenate((arrays["users"], cases["encoded_users"]))
    forward, reverse = {}, {}
    for r, e in zip(map(int, raw), map(int, encoded)):
        if (r in forward and forward[r] != e) or (e in reverse and reverse[e] != r):
            raise ValueError("raw-to-model user mapping is inconsistent")
        forward[r], reverse[e] = e, r


@contextmanager
def locked(output):
    import fcntl
    path = Path(output); path.mkdir(parents=True, exist_ok=True)
    with (path / ".plain.lock").open("a+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("this output is already owned by another launcher") from error
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def prepare(profile, data_root, output_root, *, seed=0):
    run = check(profile, data_root, seed=seed)
    with locked(output_root):
        if not (Path(output_root) / "run.json").exists() and any(p.name != ".plain.lock" for p in Path(output_root).iterdir()):
            raise ValueError("foreign nonempty output directory; choose a fresh run root")
        data = deep_check(run, data_root)
        put(Path(output_root) / "run.json", run, immutable=True)
        p = run["config"]["policy"]
        summary = dict(status="prepared", identity=run["identity"], profile=run["config"]["profile_id"], seed=seed,
            split_counts=run["bundle"]["split_counts"], pair_counts=run["bundle"]["pair_counts"],
            policy_updates=p["epochs"] * math.ceil(p["draws_per_epoch"] / p["batch_size"]),
            trajectory_draws=p["epochs"] * p["draws_per_epoch"], final_evaluation=False,
            training_or_scorer_fit_performed=False)
        put(Path(output_root) / "prepared.json", summary, immutable=True)
    return summary


def open_run(profile, data_root, output_root, seed):
    run = check(profile, data_root, seed=seed)
    path = Path(output_root) / "run.json"
    if not path.exists() or read(path) != run:
        raise ValueError("run identity missing or changed; prepare this exact config/data/seed first")
    return run, load_data(run, data_root)


def normalize(batch, units, *, root=None, bundle=None):
    import torch
    validate_units(units)
    result = dict(batch); feedback = batch["feedback"]
    if units.get("mapping") == "affine":
        feedback = (feedback + units["offset"]) / units["scale"]
    elif units.get("mapping") == "identity":
        pass
    else:
        raise ValueError("portable TRAIN units must be explicit affine or already-canonical identity; export frozen item transformations before preparation")
    if units.get("lower_clip") is not None: feedback = feedback.clamp_min(units["lower_clip"])
    if units.get("upper_clip") is not None: feedback = feedback.clamp_max(units["upper_clip"])
    result["feedback"] = feedback.masked_fill(~batch["mask"], 0.)
    result["actions"] = batch["actions"].masked_fill(~batch["mask"], 0)
    if not torch.isfinite(result["feedback"]).all():
        raise ValueError("nonfinite canonical model feedback")
    return result


def validate_units(units):
    if units.get("mapping") == "identity" and set(units) == {"mapping"}: return
    if (units.get("mapping") != "affine" or set(units) != {"mapping", "offset", "scale", "lower_clip", "upper_clip"}
            or not all(isinstance(units[k], (int, float)) and not isinstance(units[k], bool) and math.isfinite(units[k]) for k in ("offset", "scale"))
            or units["scale"] <= 0
            or any(units[k] is not None and (not isinstance(units[k], (int, float)) or not math.isfinite(units[k])) for k in ("lower_clip", "upper_clip"))
            or (units["lower_clip"] is not None and units["upper_clip"] is not None and units["lower_clip"] > units["upper_clip"])):
        raise ValueError("units require explicit finite affine offset/positive scale/clipping or already-canonical identity")


class FixtureDEV:
    """Synthetic packaging-test transitions; never selected for a native dataset."""
    def __init__(self, bundle, root):
        import numpy as np
        self.bundle = bundle
        data = load_asset(root, bundle, "fixture_world")
        self.raw_users, self.rewards, self.terminal_after = (data[k] for k in ("raw_users", "rewards", "terminal_after"))
        cases = load_asset(root, bundle, bundle["evaluation"]["cases_asset"])
        self.contexts = dict(zip(map(int, cases["raw_users"]), map(int, cases["initial_context_actions"])))
        if (set(data) != {"raw_users", "rewards", "terminal_after"} or self.rewards.ndim != 3
                or self.rewards.shape[:2] != (len(self.raw_users), bundle["evaluation"]["horizon"])
                or self.terminal_after.shape != (len(self.raw_users),) or self.terminal_after.dtype.kind not in "iu"
                or not np.isfinite(self.rewards).all() or np.any(self.terminal_after < 1)
                or np.any(self.terminal_after > self.rewards.shape[1]) or set(map(int, self.raw_users)) != set(self.contexts)):
            raise ValueError("invalid synthetic fixture transition table")
        validate_units(bundle["units"]["evaluation"])
    def fork(self):
        from copy import copy
        return copy(self)
    def reset(self, raw_user, saved_context):
        import numpy as np
        if self.contexts.get(raw_user) != saved_context: raise ValueError("fixture reset outside DEV cases")
        self.user = int(np.flatnonzero(self.raw_users == raw_user)[0]); self.steps = 0; self.done = False
        return saved_context
    def step(self, action):
        if self.done or type(action) is not int or not 2 <= action < self.rewards.shape[2] + 2:
            raise ValueError("fixture cannot step a terminal episode or a noncatalog action")
        reward = float(self.rewards[self.user, self.steps, action - 2]); self.steps += 1
        self.done = self.steps == int(self.terminal_after[self.user])
        return reward, self.done, dict(reason="fixture_terminal" if self.done else "continue")
    def feedback(self, reward, action):
        units = self.bundle["units"]["evaluation"]
        if units["mapping"] == "identity": return reward
        value = (reward + units["offset"]) / units["scale"]
        if units["lower_clip"] is not None: value = max(value, units["lower_clip"])
        if units["upper_clip"] is not None: value = min(value, units["upper_clip"])
        return value


def build_evaluation_adapter(bundle, root):
    if bundle["dataset"] == "fixture":
        if bundle["evaluation"]["adapter"] != "fixture_test_v1": raise ValueError("unknown fixture adapter")
        return FixtureDEV(bundle, root)
    from .plain_environments import build_adapter
    return build_adapter(bundle, root)


def deep_check(run, root):
    """Decode shapes/labels/units/native assets without a model or env reset."""
    data = load_data(run, root)
    validate_units(run["bundle"]["units"]["training"])
    build_evaluation_adapter(run["bundle"], root)
    if run["bundle"]["dataset"] in ("MovieLens", "Coat"):
        units = run["bundle"]["units"]["evaluation"]
        validate_units(units)
        if units["mapping"] != "affine" or units["offset"] != 0. or units["scale"] != 5.:
            raise ValueError("rating rollout units must preserve native raw rating/5")
    elif run["bundle"]["dataset"] == "KuaiRec":
        if run["bundle"]["units"]["evaluation"] != {"mapping": "item_train_range_minus_response_min"}:
            raise ValueError("KuaiRec rollout requires its frozen item-specific TRAIN response transform")
    return data


def make_model(data, config, device, *, scorer=False):
    """Execute the native constructor, including its discarded initial layers."""
    import torch
    from torch import nn
    from .features import make_dc, make_scorer
    corpus, features = data["corpus"], data["features"]
    m, s = config["model"], config["scorer"]
    if config["dataset"] != "KuaiRec" and m["feature_dim"] == 40:
        cfg = dict(policy=m, scorer=s)
        return (make_scorer if scorer else make_dc)(corpus, features, cfg, device)
    if config["dataset"] != "KuaiRec" or m["feature_dim"] != 41 or corpus.n_items != 3329 or corpus.max_length != 40:
        raise ValueError("native KuaiRec requires its released 3327-item, H40, identity-free 41D constructor")
    from .kuairec import KuairecPrDTPolicy, KuairecTrajectoryScorer
    values = torch.zeros((corpus.n_items, m["feature_dim"]), dtype=torch.float32)
    values[2:] = torch.as_tensor(features["item_vectors"], dtype=torch.float32)
    settings = s if scorer else m
    names = ("hidden_dim", "layers", "heads", "dropout") if scorer else ("hidden_dim", "layers", "heads", "dropout", "conv_window")
    model = (KuairecTrajectoryScorer if scorer else KuairecPrDTPolicy)(values, **{k: settings[k] for k in names})
    return model.to(device)


def moved(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def collate(rows, run, *, scorer=False):
    import torch
    from .data import collate_trajectories
    batch = collate_trajectories(rows)
    if run["config"]["dataset"] == "KuaiRec": batch["users"].zero_()
    stage = "scorer" if scorer else "policy"
    if run["config"][stage]["full_context_batches"]:
        horizon = run["config"]["model"]["horizon"]
        for key in ("actions", "feedback", "mask"):
            v = batch[key]; padding = torch.zeros((len(rows), horizon - v.shape[1]), dtype=v.dtype)
            batch[key] = torch.cat((v, padding), dim=1)
    return batch


def load_checkpoint(path, run, stage, device):
    import torch
    if not Path(path).exists(): return None
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if (saved.get("version") != VERSION or saved.get("identity") != run["identity"] or saved.get("config") != run["config"]
            or saved.get("stage") != stage or saved.get("device_type") != device.type):
        raise ValueError("checkpoint config/data/stage/backend identity mismatch; preserve existing output")
    state, cfg = saved["state"], run["config"][stage]
    if (type(state["epoch"]) is not int or not 0 <= state["epoch"] <= cfg["epochs"]
            or len(state["history"]) != state["epoch"] or type(state["batch_offset"]) is not int
            or state["batch_offset"] < 0):
        raise ValueError("checkpoint completed-epoch/history/offset counters differ")
    count = cfg["draws_per_epoch"] if stage == "policy" else run["bundle"]["pair_counts"]["train"]
    per_epoch = math.ceil(count / cfg["batch_size"])
    offset, order = state["batch_offset"], state["order"]
    if (offset > count or (offset != count and offset % cfg["batch_size"])
            or (order is None and offset != 0)
            or (order is not None and (order.shape != (count,) or order.dtype != torch.int64))
            or state["updates"] != state["epoch"] * per_epoch + math.ceil(offset / cfg["batch_size"])):
        raise ValueError("checkpoint update budget or saved epoch draw cursor differs")
    if stage == "policy":
        if state["draws"] != state["epoch"] * count + offset or saved["schedule"] != dict(total_updates=cfg["cosine_horizon_updates"], next_update=state["updates"]):
            raise ValueError("checkpoint draw/schedule counters differ")
        rates = [v for row in state["history"] for v in row["applied_learning_rates"]] + state["epoch_rates"]
        if rates != [learning_rate(cfg, i) for i in range(state["updates"])]:
            raise ValueError("checkpoint applied learning-rate trace differs from the full cosine")
    return saved


def learning_rate(policy, update):
    total = policy["cosine_horizon_updates"]
    if type(update) is not int or not 0 <= update < total:
        raise ValueError("applied cosine update index outside the original full horizon")
    if update == 0: return policy["learning_rate"]
    if update == total - 1: return policy["minimum_learning_rate"]
    return policy["minimum_learning_rate"] + .5 * (policy["learning_rate"] - policy["minimum_learning_rate"]) * (1. + math.cos(math.pi * update / (total - 1)))


def _stage_state(run, stage, model, optimizer, generator, device, state, **extra):
    from .training import _rng_state
    return dict(version=VERSION, identity=run["identity"], stage=stage, device_type=device.type,
        model=model.state_dict(), optimizer=optimizer.state_dict(), rng=_rng_state(generator),
        state=state, config=run["config"], **extra)


def _score_output(run, data, model, root, output, device, generator):
    import numpy as np
    import torch
    from .data import collate_trajectories
    from .training import _rng_state, _restore_rng_state
    before = _rng_state(generator); scores, ids = [], []
    try:
        model.eval()
        with torch.no_grad():
            rows = data["corpus"].split("train"); size = run["config"]["scorer"]["batch_size"]
            for start in range(0, len(rows), size):
                batch = moved(normalize(collate(rows[start:start + size], run, scorer=True), run["bundle"]["units"]["training"]), device)
                scores.extend(model(*(batch[k] for k in ("users", "actions", "feedback", "mask"))).float().cpu().tolist())
                ids.extend(batch["indices"].cpu().tolist())
    finally:
        _restore_rng_state(before, generator)
    if not np.isfinite(scores).all(): raise FloatingPointError("nonfinite frozen TRAIN scores")
    path = Path(output) / "scores.npz"
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream: np.savez_compressed(stream, indices=np.asarray(ids, dtype=np.int64), scores=np.asarray(scores, dtype=np.float32))
    temporary.replace(path)
    return path


def _public_preference_loss(logits, targets, dataset):
    import torch.nn.functional as F
    if dataset in ("Coat", "KuaiRec"):
        # Coat's soft targets and KuaiRec's strict pairs retain their native
        # softplus implementation; MovieLens uses the retained BCE primitive.
        return (targets * F.softplus(-logits) + (1 - targets) * F.softplus(logits)).mean()
    return F.binary_cross_entropy_with_logits(logits, targets)


def _scorer_validation(run, data, model, device):
    """Fixed-endpoint diagnostics with each source's validation RNG chronology."""
    import torch
    from torch.utils.data import DataLoader
    cfg=run["config"]["scorer"]; pairs=data["pairs"]
    high,low=pairs["validation_preferred"],pairs["validation_rejected"]
    if run["config"]["dataset"] == "Coat":
        # Native Coat iterates a non-shuffled, unseeded validation loader.
        # Its CPU base-seed draw is part of the following epoch's dropout RNG.
        batches=DataLoader(range(len(high)),batch_size=cfg["batch_size"],shuffle=False)
    else:
        batches=(range(i,min(i+cfg["batch_size"],len(high)))
                 for i in range(0,len(high),cfg["batch_size"]))
    was_training=model.training;model.eval();correct=strict_count=ties=count=0;nll=0.
    try:
        with torch.no_grad():
            for positions in batches:
                indices=list(map(int,positions))
                def score(ids):
                    batch=moved(normalize(collate([data["corpus"].get(int(i)) for i in ids],run,scorer=True),run["bundle"]["units"]["training"]),device)
                    return model(*(batch[k] for k in ("users","actions","feedback","mask")))
                delta=score(high[indices])-score(low[indices])
                target=torch.as_tensor(pairs["validation_targets"][indices],dtype=delta.dtype,device=device)
                nll+=float(_public_preference_loss(delta,target,run["config"]["dataset"]))*len(indices)
                strict=target != .5
                correct+=int((((delta>0)&(target==1))|((delta<0)&(target==0))).sum())
                strict_count+=int(strict.sum());ties+=int((~strict).sum());count+=len(indices)
    finally:
        model.train(was_training)
    return dict(validation_pair_nll=nll/count if count else None,
                validation_pair_accuracy=correct/strict_count if strict_count else None,
                validation_pairs=count,validation_ties=ties)


def scorer(profile, data_root, output_root, *, seed=0, device="auto", stop_after_updates=None):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from .data import collate_trajectories
    from .objective import bradley_terry_loss
    from .training import choose_device, seed_everything, _restore_rng_state
    output = Path(output_root)
    with locked(output):
        run, data = open_run(profile, data_root, output, seed)
        result_path = output / "scorer.json"
        if result_path.exists():
            result = read(result_path)
            if result["identity"] != run["identity"] or digest(output / "scores.npz") != result["scores_sha256"]:
                raise ValueError("completed scorer score binding changed")
            return result
        if "scores" in run["bundle"]["assets"]:
            cached = load_asset(data_root, run["bundle"], "scores")
            _validate_scores(data, cached)
            with (output / "scores.npz").open("wb") as stream: np.savez_compressed(stream, **cached)
            result = dict(status="frozen_scorer_cache_reused", identity=run["identity"], seed=seed,
                scores_sha256=digest(output / "scores.npz"), new_scorer_updates=0,
                source_sha256=run["bundle"]["assets"]["scores"]["sha256"])
            put(result_path, result, immutable=True); return result
        dev = choose_device(device); seed_everything(seed)
        model = make_model(data, run["config"], dev, scorer=True); cfg = run["config"]["scorer"]
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=0.)
        gen = torch.Generator().manual_seed(seed); reg = torch.Generator().manual_seed(seed + 991)
        reg_np = np.random.default_rng(0)
        state = dict(epoch=0, batch_offset=0, order=None, updates=0, history=[], epoch_bt=0., epoch_square=0., epoch_batches=0)
        path = output / "scorer_latest.pt"; checkpoint = load_checkpoint(path, run, "scorer", dev)
        if checkpoint:
            model.load_state_dict(checkpoint["model"]); optimizer.load_state_dict(checkpoint["optimizer"])
            _restore_rng_state(checkpoint["rng"], gen); reg.set_state(checkpoint["regularizer_rng"].cpu()); state = checkpoint["state"]
            if checkpoint.get("regularizer_numpy_state") is not None: reg_np.bit_generator.state = checkpoint["regularizer_numpy_state"]
        pairs = data["pairs"]; high, low = pairs["train_preferred"], pairs["train_rejected"]
        rows = data["corpus"].split("train")
        if not len(high): raise ValueError("no frozen strict TRAIN pairs; no scorer can be fitted")
        while state["epoch"] < cfg["epochs"]:
            if state["order"] is None:
                if cfg["pair_order_rng"] == "numpy_epoch_seed":
                    state["order"] = torch.from_numpy(np.random.default_rng(seed + 10000 * (state["epoch"] + 1)).permutation(len(high)))
                    reg_np = np.random.default_rng(seed + 991 + 10000 * (state["epoch"] + 1))
                elif cfg["pair_order_rng"] == "torch_randperm":
                    state["order"] = torch.randperm(len(high), generator=gen)
                else:
                    state["order"] = torch.cat(list(DataLoader(range(len(high)), batch_size=cfg["batch_size"], shuffle=True, generator=gen)))
            order = state["order"]; model.train()
            for start in range(state["batch_offset"], len(order), cfg["batch_size"]):
                indices = order[start:start + cfg["batch_size"]].tolist()
                def inputs(ids):
                    return moved(normalize(collate([data["corpus"].get(int(i)) for i in ids], run, scorer=True), run["bundle"]["units"]["training"]), dev)
                high_batch, low_batch = inputs(high[indices]), inputs(low[indices])
                reg_rows = (reg_np.integers(0, len(rows), size=cfg["batch_size"]).tolist() if cfg["pair_order_rng"] == "numpy_epoch_seed"
                            else torch.randint(len(rows), (cfg["batch_size"],), generator=reg).tolist())
                regularizer = inputs([rows[i].index for i in reg_rows])
                def forward(b): return model(*(b[k] for k in ("users", "actions", "feedback", "mask")))
                if run["config"]["schema_version"] == PUBLIC_SCHEMA:
                    targets = torch.as_tensor(pairs["train_targets"][indices], dtype=torch.float32, device=dev)
                    bt = _public_preference_loss(forward(high_batch) - forward(low_batch), targets, run["config"]["dataset"])
                else:
                    bt = bradley_terry_loss(forward(high_batch), forward(low_batch))
                square = forward(regularizer).square().mean()
                loss = bt + cfg["lambda_score"] * square
                optimizer.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True); optimizer.step()
                if not torch.isfinite(loss): raise FloatingPointError("nonfinite scorer objective")
                state["updates"] += 1; state["batch_offset"] = start + len(indices)
                state["epoch_bt"] += float(bt.detach()); state["epoch_square"] += float(square.detach()); state["epoch_batches"] += 1
                stop = stop_after_updates is not None and state["updates"] >= stop_after_updates
                if stop or state["updates"] % cfg["checkpoint_interval_updates"] == 0:
                    save(path, _stage_state(run, "scorer", model, optimizer, gen, dev, state, regularizer_rng=reg.get_state(),
                        regularizer_numpy_state=deepcopy(reg_np.bit_generator.state) if cfg["pair_order_rng"] == "numpy_epoch_seed" else None))
                if stop: return dict(status="interrupted_for_fixture", updates=state["updates"])
            diagnostic=(_scorer_validation(run,data,model,dev)
                        if run["config"]["schema_version"] == PUBLIC_SCHEMA else {})
            state["history"].append(dict(epoch=state["epoch"] + 1,
                bradley_terry_nll=state["epoch_bt"] / state["epoch_batches"],
                uniform_score_square=state["epoch_square"] / state["epoch_batches"],**diagnostic))
            state.update(epoch=state["epoch"] + 1, batch_offset=0, order=None, epoch_bt=0., epoch_square=0., epoch_batches=0)
            save(path, _stage_state(run, "scorer", model, optimizer, gen, dev, state, regularizer_rng=reg.get_state(),
                regularizer_numpy_state=deepcopy(reg_np.bit_generator.state) if cfg["pair_order_rng"] == "numpy_epoch_seed" else None))
        _score_output(run, data, model, data_root, output, dev, gen)
        result = dict(status="scorer_complete", identity=run["identity"], seed=seed,
            epochs=state["epoch"], new_scorer_updates=state["updates"], history=state["history"],
            scores_sha256=digest(output / "scores.npz"), checkpoint_sha256=digest(path))
        put(result_path, result, immutable=True); return result


def _validate_scores(data, cached):
    import numpy as np
    if set(cached) != {"indices", "scores"} or cached["indices"].shape != cached["scores"].shape or cached["scores"].ndim != 1:
        raise ValueError("cached TRAIN scores require aligned indices/scores vectors")
    expected = [t.index for t in data["corpus"].split("train")]
    if (list(map(int, cached["indices"])) != expected or not np.isfinite(cached["scores"]).all()):
        raise ValueError("cached scores must cover all TRAIN episodes in canonical order")


def probabilities(run, data, data_root, output):
    import numpy as np
    import torch
    with np.load(Path(output) / "scores.npz", allow_pickle=False) as archive:
        cached = {k: archive[k].copy() for k in archive.files}
    _validate_scores(data, cached); w = run["config"]["weighting"]
    if w["precision"] == "native_float64_softmax":
        values = cached["scores"].astype(np.float64); labels = np.asarray([data["collectors"][int(i)] for i in cached["indices"]])
        q = np.zeros(len(values), dtype=np.float64)
        if w["rule"] == "fixed_collector_prior":
            if set(labels) != set(w["collector_prior"]): raise ValueError("native collector support differs from frozen prior")
            for label, mass in w["collector_prior"].items():
                choose = labels == label; local = np.exp(w["beta"] * (values[choose] - float(values[choose].max())))
                q[choose] = mass * local / local.sum(dtype=np.float64)
            q /= q.sum(dtype=np.float64)
        else:
            q = np.exp(w["beta"] * (values - float(values.max()))); q /= q.sum(dtype=np.float64)
        if not np.isfinite(q).all() or np.any(q <= 0): raise ValueError("native global softmax lost positive finite support")
        return torch.from_numpy(q), dict(rule=w["rule"], beta=w["beta"], collector_prior=w["collector_prior"],
            precision=w["precision"], implicit_sampler_mass=float(q.sum()), effective_sample_size=float(q.sum() ** 2 / np.square(q).sum()),
            final_float64_sha256=hashlib.sha256(q.tobytes()).hexdigest())
    if "weights" in run["bundle"]["assets"]:
        weights = load_asset(data_root, run["bundle"], "weights")
        if (set(weights) != {"indices", "weights"} or not np.array_equal(weights["indices"], cached["indices"])
                or weights["weights"].dtype != np.float32 or weights["weights"].shape != cached["scores"].shape):
            raise ValueError("frozen global weights must be original aligned float32 TRAIN masses")
        raw = torch.from_numpy(weights["weights"].copy())
    else:
        from .objective import global_softmax_weights
        raw = global_softmax_weights(torch.from_numpy(cached["scores"]).double(), w["beta"]).float()
    q = raw.double()
    if not torch.isfinite(q).all() or torch.any(q <= 0): raise ValueError("global softmax must retain positive finite TRAIN support")
    q /= q.sum(); source = q.clone(); ids = list(map(int, cached["indices"]))
    if w["rule"] == "fixed_collector_prior":
        labels = [data["collectors"][i] for i in ids]
        if set(labels) != set(w["collector_prior"]): raise ValueError("fixed collector support differs from the frozen prior")
        for label, mass in w["collector_prior"].items():
            choose = torch.tensor([x == label for x in labels]); q[choose] *= mass / source[choose].sum()
    return q, dict(rule=w["rule"], beta=w["beta"], collector_prior=w["collector_prior"],
        implicit_sampler_mass=float(q.sum()), effective_sample_size=float(q.sum().square() / q.square().sum()),
        original_global_float32_sha256=hashlib.sha256(raw.numpy().tobytes()).hexdigest(),
        final_float64_sha256=hashlib.sha256(q.numpy().tobytes()).hexdigest())


def policy(profile, data_root, output_root, *, seed=0, device="auto", stop_after_updates=None):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from .data import collate_trajectories
    from .loss import per_trajectory_mean_action_nll
    from .training import choose_device, seed_everything, _restore_rng_state
    output = Path(output_root)
    with locked(output):
        run, data = open_run(profile, data_root, output, seed)
        if not (output / "scorer.json").exists() or digest(output / "scores.npz") != read(output / "scorer.json")["scores_sha256"]:
            raise ValueError("fit or import this seed's frozen scorer scores before policy")
        result_path = output / "policy.json"
        if result_path.exists():
            result = read(result_path)
            if result["identity"] != run["identity"] or digest(output / "policy_final.pt") != result["checkpoint_sha256"]:
                raise ValueError("completed policy binding changed")
            return result
        dev = choose_device(device); seed_everything(seed)
        model = make_model(data, run["config"], dev); cfg = run["config"]["policy"]
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
        gen = torch.Generator().manual_seed(77 + seed)
        sampler_np = np.random.default_rng(77 + seed)
        q, weight_evidence = probabilities(run, data, data_root, output)
        state = dict(epoch=0, batch_offset=0, order=None, updates=0, draws=0, tokens=0, history=[], epoch_loss=0., epoch_norms=[], epoch_rates=[], epoch_tokens=0)
        path = output / "policy_latest.pt"; checkpoint = load_checkpoint(path, run, "policy", dev)
        if checkpoint:
            if not torch.equal(checkpoint["sampling_probabilities"], q): raise ValueError("saved weighted sampler law changed")
            model.load_state_dict(checkpoint["model"]); optimizer.load_state_dict(checkpoint["optimizer"])
            _restore_rng_state(checkpoint["rng"], gen); state = checkpoint["state"]
            if checkpoint.get("numpy_sampler_state") is not None: sampler_np.bit_generator.state = checkpoint["numpy_sampler_state"]
        else:
            save(output / "policy_initial.pt", _stage_state(run, "policy", model, optimizer, gen, dev, deepcopy(state), sampling_probabilities=q))
        rows = data["policy_corpus"].split("train")
        while state["epoch"] < cfg["epochs"]:
            if state["order"] is None:
                if cfg["sampling_rng"] == "numpy_choice_persistent":
                    state["order"] = torch.from_numpy(sampler_np.choice(len(rows), cfg["draws_per_epoch"], replace=True, p=q.numpy()))
                else:
                    sampler = WeightedRandomSampler(q, cfg["draws_per_epoch"], replacement=True, generator=gen)
                    state["order"] = torch.cat(list(DataLoader(range(len(rows)), batch_size=cfg["batch_size"], sampler=sampler, generator=gen)))
            order = state["order"]; model.train()
            for start in range(state["batch_offset"], len(order), cfg["batch_size"]):
                draw = order[start:start + cfg["batch_size"]].tolist()
                batch = moved(normalize(collate([rows[i] for i in draw], run), run["bundle"]["units"].get("policy_training", run["bundle"]["units"]["training"])), dev)
                rate = learning_rate(cfg, state["updates"])
                for group in optimizer.param_groups: group["lr"] = rate
                logits = model(*(batch[k] for k in ("users", "actions", "feedback", "mask")), initial_context_actions=batch["initial_context_actions"])
                loss = per_trajectory_mean_action_nll(logits, batch["actions"], batch["mask"]).mean()
                optimizer.zero_grad(set_to_none=True); loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True); optimizer.step()
                if not torch.isfinite(loss): raise FloatingPointError("nonfinite policy mean-action objective")
                count = int(batch["mask"].sum())
                state["updates"] += 1; state["draws"] += len(draw); state["tokens"] += count; state["epoch_tokens"] += count
                state["batch_offset"] = start + len(draw); state["epoch_loss"] += float(loss.detach())
                state["epoch_norms"].append(float(norm)); state["epoch_rates"].append(rate)
                stop = stop_after_updates is not None and state["updates"] >= stop_after_updates
                if stop or state["updates"] % cfg["checkpoint_interval_updates"] == 0:
                    save(path, _stage_state(run, "policy", model, optimizer, gen, dev, state, sampling_probabilities=q,
                        schedule=dict(total_updates=cfg["cosine_horizon_updates"], next_update=state["updates"]),
                        numpy_sampler_state=deepcopy(sampler_np.bit_generator.state) if cfg["sampling_rng"] == "numpy_choice_persistent" else None))
                if stop: return dict(status="interrupted_for_fixture", updates=state["updates"])
            global_ids = torch.tensor([rows[i].index for i in order.tolist()], dtype=torch.int64)
            state["history"].append(dict(epoch=state["epoch"] + 1, draws=len(order), updates=len(state["epoch_rates"]),
                valid_targets=state["epoch_tokens"], loss=state["epoch_loss"] / len(state["epoch_rates"]),
                preclip_gradient_norms=state["epoch_norms"], applied_learning_rates=state["epoch_rates"],
                global_draw_stream_sha256=hashlib.sha256(global_ids.numpy().tobytes()).hexdigest(),
                private_sampler_state_sha256=identity(sampler_np.bit_generator.state) if cfg["sampling_rng"] == "numpy_choice_persistent" else hashlib.sha256(gen.get_state().numpy().tobytes()).hexdigest()))
            state.update(epoch=state["epoch"] + 1, batch_offset=0, order=None, epoch_loss=0., epoch_norms=[], epoch_rates=[], epoch_tokens=0)
            save(path, _stage_state(run, "policy", model, optimizer, gen, dev, state, sampling_probabilities=q,
                schedule=dict(total_updates=cfg["cosine_horizon_updates"], next_update=state["updates"]),
                numpy_sampler_state=deepcopy(sampler_np.bit_generator.state) if cfg["sampling_rng"] == "numpy_choice_persistent" else None))
        save(output / "policy_final.pt", torch.load(path, map_location="cpu", weights_only=False))
        result = dict(status="policy_complete", identity=run["identity"], seed=seed, sampler_seed=77 + seed,
            epochs=state["epoch"], optimizer_updates=state["updates"], trajectory_draws=state["draws"],
            valid_action_targets=state["tokens"], history=state["history"], weighting=weight_evidence,
            checkpoint_sha256=digest(output / "policy_final.pt"), objective=cfg["loss"], final_evaluation=False)
        put(result_path, result, immutable=True); return result


def evaluate(profile, data_root, output_root, *, seed=0, device="auto", adapter_factory=None):
    """Closed DEV evaluation. No teacher forcing, repeat exclusion or terminal mask loss."""
    import numpy as np
    import torch
    from .training import choose_device, seed_everything
    output = Path(output_root)
    with locked(output):
        run, data = open_run(profile, data_root, output, seed)
        if not (output / "policy.json").exists(): raise ValueError("fixed policy endpoint must complete before DEV evaluation")
        policy_result = read(output / "policy.json")
        if digest(output / "policy_final.pt") != policy_result["checkpoint_sha256"]: raise ValueError("policy endpoint changed")
        result_path = output / "evaluation.json"
        if result_path.exists():
            result = read(result_path)
            if (result["identity"] != run["identity"] or digest(output / "development_raw.json") != result["raw_sha256"]
                    or digest(output / "development_logits.npz") != result["logits_sha256"]): raise ValueError("completed DEV evidence changed")
            return result
        attempt = output / "evaluation_attempt.json"
        if attempt.exists():
            raise ValueError("partial DEV attempt exists without complete result; preserve evidence and request direction, never silently reroll")
        dev = choose_device(device)
        if adapter_factory is None:
            adapter_factory = build_evaluation_adapter
        adapter = adapter_factory(run["bundle"], Path(data_root))
        seed_everything(seed); model = make_model(data, run["config"], dev)
        checkpoint = load_checkpoint(output / "policy_final.pt", run, "policy", dev)
        if checkpoint["state"]["epoch"] != run["config"]["policy"]["epochs"]: raise ValueError("incomplete fixed policy")
        model.load_state_dict(checkpoint["model"]); model.eval()
        for parameter in model.parameters(): parameter.requires_grad_(False)
        put(attempt, dict(identity=run["identity"], policy_sha256=policy_result["checkpoint_sha256"], phase="development"), immutable=True)
        episodes, catalog, positions = [], [], []
        cases = data["cases"]; evaluation = run["config"]["evaluation"]; horizon = evaluation["horizon"]
        case_rows = list(zip(cases["raw_users"], cases["encoded_users"], cases["initial_context_actions"]))
        with torch.no_grad():
            for block_start in range(0, len(case_rows), evaluation["reset_block_size"]):
                block = case_rows[block_start:block_start + evaluation["reset_block_size"]]
                actors, records = [], []
                for raw_user, user, context in block:
                    actor = adapter.fork()
                    actual_context = actor.reset(int(raw_user), int(context))
                    if actual_context != int(context): raise ValueError("native reset differs from the frozen saved context")
                    actors.append(actor)
                    records.append(dict(raw_user=int(raw_user), encoded_user=int(user), initial_context_action=int(context),
                        actions=[], rewards=[], normalized_feedback=[], native_info=[], terminated=False))
                active = list(range(len(block)))
                for t in range(horizon):
                    if not active: break
                    survivors = []
                    for start in range(0, len(active), evaluation["batch_size"]):
                        selected = active[start:start + evaluation["batch_size"]]
                        # Current placeholders are unobserved: DC shifts actual
                        # prior feedback; all active rows have the same t+1 shape.
                        a = torch.tensor([records[i]["actions"] + [0] for i in selected], dtype=torch.long, device=dev)
                        r = torch.tensor([records[i]["normalized_feedback"] + [0.] for i in selected], dtype=torch.float32, device=dev)
                        mask = torch.ones_like(a, dtype=torch.bool)
                        logits = model(torch.tensor([records[i]["encoded_user"] for i in selected], device=dev), a, r, mask,
                            initial_context_actions=torch.tensor([records[i]["initial_context_action"] for i in selected], device=dev))[:, -1, 2:]
                        if not torch.isfinite(logits).all(): raise FloatingPointError("nonfinite real catalog logits")
                        chosen = torch.argmax(logits, dim=-1).cpu().tolist()
                        for j, i in enumerate(selected):
                            record, actor, action = records[i], actors[i], int(chosen[j]) + 2
                            catalog.append(logits[j].float().cpu().numpy()); positions.append([record["raw_user"], t])
                            reward, terminated, info = actor.step(action)
                            if not math.isfinite(reward) or type(terminated) is not bool: raise ValueError("invalid native reward/termination")
                            record["actions"].append(action); record["rewards"].append(float(reward)); record["native_info"].append(info)
                            observed = float(actor.feedback(float(reward), action))
                            if not math.isfinite(observed): raise ValueError("nonfinite actual causal rollout feedback")
                            record["normalized_feedback"].append(observed); record["terminated"] = terminated
                            if not terminated: survivors.append(i)
                    active = survivors
                for record in records:
                    record.update(length=len(record["actions"]), cumulative_reward=sum(record["rewards"]), cap_hit=len(record["actions"]) == horizon)
                episodes.extend(records)
        put(output / "development_raw.json", dict(identity=run["identity"], phase="development", episodes=episodes), immutable=True)
        with (output / "development_logits.npz").open("wb") as stream:
            np.savez_compressed(stream, original_real_logits=np.asarray(catalog, dtype=np.float32),
                raw_user_and_past_executions=np.asarray(positions, dtype=np.int64))
        total_steps = sum(e["length"] for e in episodes); total_reward = sum(e["cumulative_reward"] for e in episodes)
        result = dict(status="development_complete", identity=run["identity"], users=len(episodes),
            mean_cumulative_reward=total_reward / len(episodes), pooled_reward_per_step=total_reward / total_steps,
            mean_length=total_steps / len(episodes), raw_sha256=digest(output / "development_raw.json"),
            logits_sha256=digest(output / "development_logits.npz"), policy_sha256=policy_result["checkpoint_sha256"],
            final_evaluation=False, adapter=run["bundle"]["evaluation"]["adapter"], batch_size=evaluation["batch_size"], reset_block_size=evaluation["reset_block_size"],
            terminal_reward_included=True, ordinary_greedy=True, no_repeat_mask=False)
        put(result_path, result, immutable=True); return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("list", "check", "prepare", "scorer", "policy", "evaluate", "resume"))
    parser.add_argument("--profile", default="movielens")
    parser.add_argument("--config", type=Path, help="shared config file; defaults to configs/prdt.json")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true", help="validate only; never construct models or write outputs")
    parser.add_argument("--through", choices=("scorer", "policy", "evaluate"), default="policy", help="resume only through this explicit stage")
    args = parser.parse_args(argv)
    try:
        if args.stage == "list": result = list_profiles(args.config)
        else:
            if args.config is not None: args.profile = f"{args.config}#{args.profile}"
            if args.data_root is None: parser.error("--data-root is required")
            if args.stage == "check" or args.dry_run:
                run = check(args.profile, args.data_root, seed=args.seed)
                deep_check(run, args.data_root)
                result = dict(status="preflight_pass", identity=run["identity"], profile=run["config"]["profile_id"], writes=False, models_constructed=False, payload_validated=True, native_environment_reset_or_step=False)
            else:
                if args.output_root is None: parser.error("--output-root is required")
                kw = dict(seed=args.seed, device=args.device)
                if args.stage == "prepare": result = prepare(args.profile, args.data_root, args.output_root, seed=args.seed)
                elif args.stage == "resume":
                    scorer(args.profile, args.data_root, args.output_root, **kw)
                    result = policy(args.profile, args.data_root, args.output_root, **kw) if args.through != "scorer" else read(args.output_root / "scorer.json")
                    if args.through == "evaluate": result = evaluate(args.profile, args.data_root, args.output_root, **kw)
                else: result = globals()[args.stage](args.profile, args.data_root, args.output_root, **kw)
        print(json.dumps(result, sort_keys=True, allow_nan=False)); return 0
    except (ValueError, RuntimeError, FileNotFoundError, KeyError) as error:
        print(json.dumps(dict(status="blocked", reason=str(error)), sort_keys=True)); return 2


if __name__ == "__main__":
    raise SystemExit(main())
