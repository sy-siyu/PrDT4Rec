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

from .data.bundles import load_asset, load_data, relative, validate_units

VERSION = "portable-plain-prdt-v1"
CONFIG_SCHEMA = "plain-prdt-config-v1"
PUBLIC_SCHEMA = "plain-prdt-config-v2"
BUNDLE_SCHEMA = "plain-prdt-bundle-v1"
PROFILES_SCHEMA = "plain-prdt-profiles-v1"
LOCK_NAME = ".plain.lock"
# Executable sources of the initial three-profile release (commit 160b9ac).
_INITIAL_RELEASE_SOURCES = {
    "plain.py": "423a2f9a50a61539826cb6db87422f8ab61856b6d8fa787c32c9f882db8ffea0",
    "plain_environments.py": "91cfc86ff25027f507f4117aeaabe4572f4e7d4dd625e4d2d8a10765c417c98e",
    "data.py": "c1ca07f2239a56d608aff78ce93fd892a41092724b14fbc64cc969da2fa79304",
    "models.py": "a3dfcff08a7fa5eed3b1fe7518565bd58a5aa7144d5ca504381cefc57cb5bbe5",
    "training.py": "7dfd7d32110879b262b5811ae89c1eb77106133225df99b697ed8e9eecb8188d",
    "objective.py": "8403f035c6141c2dd69fe143401a8654ba2b64350ea56ecde7edd840cf3eb88b",
    "features.py": "d7e11a665b8f4faf89d550d7364f860cc70e613d40ebb3dac6967031128148db",
    "loss.py": "f6720bde6dcca6cbe46ad7302170b535db68a692953dcf52c08be8bde57a03e5",
}
# Source identities before the environment package move (commit 4c423ef).
_PRE_MOVE_SOURCES = {
    **{name: _INITIAL_RELEASE_SOURCES[name] for name in ("data.py", "training.py")},
    "models.py": "3f38f965ffaeecd0259322c29d97bc8aba351cb654b2e855d0aade0d761eaf2d",
    "objective.py": "ccdda7a2824bc44e9c5578e884f3f75ffa43d4eae47d0ea5b9121612016b2db1",
    "runner.py": "b24e95c410458b9a1dd3402b7d4140a404ba6f995a14bb53659836940705bf10",
    "environments.py": "6319b30da9d26d1b1768b2f6e7db966e0b32e789f46fdcbf2733530b55004b07",
    "kuairand.py": "5441e443779ec86cf922ba9b600b8957b98ece15a560692fb09a41354e3f41e2",
    "yahoor3.py": "e151dd3323937ec7174dd09c14917f68e10d153b53995bf83bbd7f6a61f561ad",
    "virtualtb.py": "6d9f5a2ca8de95998b190864d6a3664757cc7354a71b4c326491d924c7486600",
    "virtualtb_native.py": "bf56dee03b242f4410b780c70cadca3c01a0e2e0a064bffea5da39017f154e33",
    "continuous.py": "0d9b07c1762d8fda99e9ee909610887a629148a9a52b94bce7dc9d331b005d82",
}
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


def _implementation_hashes(names):
    root = Path(__file__).parent
    return {name: digest(root / name) for name in names}


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




def merge_settings(defaults, overrides):
    """Recursively merge dataset overrides without modifying shared defaults."""
    result = deepcopy(defaults)
    for key, value in overrides.items():
        result[key] = (merge_settings(result[key], value)
                       if isinstance(result.get(key), dict) and isinstance(value, dict)
                       else deepcopy(value))
    return result


def dataset_settings(config_file=None):
    """Resolve the shared parameters and each dataset's overrides."""
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


def _validate_public(config):
    # Format metadata stays internal; preserve historical resolved identities.
    config.setdefault("schema_version", PUBLIC_SCHEMA)
    if config["schema_version"] != PUBLIC_SCHEMA:
        raise ValueError("shared config requires the public profile format")
    validate_config(config)
    return config


def public_profiles(config_file=None):
    """Return profiles supported by the prepared-bundle launcher."""
    return {name: _validate_public(config) for name, config in dataset_settings(config_file).items()}


def list_profiles(config_file=None):
    return [dict(profile=name, dataset=c["dataset"], runnable_recipe=True)
            for name, c in sorted(public_profiles(config_file).items())]


def load_config(profile):
    name = str(profile)
    if "#" in name:
        config_file, dataset = name.rsplit("#", 1)
        profiles = dataset_settings(config_file)
        if dataset not in profiles: raise ValueError(f"unknown dataset profile: {dataset}")
        return _validate_public(profiles[dataset])
    source = Path(profile)
    if not source.suffix and source.name == name:
        profiles = dataset_settings()
        if name in profiles: return _validate_public(profiles[name])
    path = source if source.suffix == ".json" else CONFIGS / f"{profile}.json"
    # Previously published per-dataset paths resolve to the same merged recipe.
    if not path.exists() and path.parent.resolve() == CONFIGS.resolve():
        profiles = dataset_settings()
        if path.stem in profiles: return _validate_public(profiles[path.stem])
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
    if public and _input_type(c) == "continuous":
        from .data.virtualtb import validate_config as validate_continuous_config
        validate_continuous_config(c)
        return
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
    optional = {
        "YahooR3": (set(), set(), {"gradient_clip", "selection"}, set(),
                    {"context_seed", "feedback_units", "leave_window", "leave_threshold"},
                    {"pairing", "representation"}),
        "KuaiRand": ({"catalog_items", "users"}, {"optimizer_updates", "trajectory_draws"},
                     {"gradient_clip", "selection"}, set(),
                     {"context_seed", "feedback_units", "native_quitting", "category_window",
                      "literal_python_slice", "matched_resets"},
                     {"bundle_identity", "pairing", "representation"}),
    }
    extras = optional.get(c["dataset"], (set(),) * len(keys)) if public else (set(),) * len(keys)
    extras = (extras[0] | {"input_type", "user_features", "catalog_items"}, *extras[1:])
    if any(not names <= obj.keys() or not obj.keys() <= names | extra
           for (obj, names), extra in zip(keys, extras, strict=True)):
        raise ValueError("unknown or missing executable recipe fields")
    if (_input_type(c) != "catalog" or m.get("user_features", "projected") not in ("projected", "zero")
            or (m.get("user_features") == "zero" and "catalog_items" not in m)):
        raise ValueError("invalid configured catalog model inputs")
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
            if rule in ("cumulative_logged_feedback", "cumulative_logged_native_feedback"):
                if threshold is not None:
                    raise ValueError("cumulative labels have no count threshold")
            elif not isinstance(threshold, (int, float)) or isinstance(threshold, bool) or not math.isfinite(threshold):
                raise ValueError("count labels need an explicit finite threshold")
    if (w["rule"] not in ("global_softmax", "fixed_collector_prior") or not math.isfinite(w["beta"]) or w["beta"] < 0
            or w["precision"] not in ("saved_global_float32_then_TRAIN_float64_normalized",
                                     "saved_global_float32_then_TRAIN_numpy_float64_normalized",
                                     "native_float64_softmax")):
        raise ValueError("declare the global softmax or fixed collector prior and its precision explicitly")
    if w["rule"] == "fixed_collector_prior":
        prior = w["collector_prior"]
        if not isinstance(prior, dict) or not prior or any(v <= 0 for v in prior.values()) or not math.isclose(sum(prior.values()), 1., abs_tol=1e-12):
            raise ValueError("collector prior must be a positive normalized fixed map")
    elif w["collector_prior"] is not None:
        raise ValueError("global sampling does not use a collector prior")


def _input_type(config):
    return config["model"].get("input_type", "catalog")


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
    adapter_aliases = {"movielens": "movielens_native_v1", "kuairec": "kuairec_native_v1",
                       "coat": "coat_native_v1", "yahoor3": "yahoor3_native_v1",
                       "kuairand": "kuairand_native_v1"}
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
              "Coat": ("coat_native_v1", {"world", "distance"}, 7, 6),
              "YahooR3": ("yahoor3_native_v1", {"world", "distance"}, 3, 120),
              "KuaiRand": ("kuairand_native_v1", {"world", "categories"}, 10, None)}
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
    numerical_sources = ["runner.py", "environments/common.py", "data/bundles.py",
                         "data/trajectories.py", "models.py", "training.py", "objective.py"]
    if config["dataset"] == "YahooR3":
        numerical_sources.extend(("data/yahoor3.py", "environments/yahoor3.py"))
    elif config["dataset"] == "KuaiRand":
        numerical_sources.extend(("data/kuairand.py", "environments/kuairand.py"))
    summary = {"version": VERSION, "config": config, "bundle": b, "bundle_sha256": digest(bundle_path), "seed": seed,
               "implementation_sha256": _implementation_hashes(numerical_sources)}
    summary["identity"] = identity(summary)
    return summary




@contextmanager
def locked(output):
    import fcntl
    path = Path(output); path.mkdir(parents=True, exist_ok=True)
    with (path / LOCK_NAME).open("a+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("this output is already owned by another launcher") from error
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _bound_run(run, saved):
    if saved == run:
        return run
    dataset = run["config"]["dataset"]
    names = ["runner.py", "models.py", "training.py", "objective.py"]
    if dataset == "VirtualTB":
        names.extend(("virtualtb.py", "virtualtb_native.py", "continuous.py"))
    else:
        names.extend(("environments.py", "data.py"))
        if dataset in ("YahooR3", "KuaiRand"):
            names.append(dataset.lower() + ".py")
    previous = {key: value for key, value in run.items() if key != "identity"}
    previous["implementation_sha256"] = {name: _PRE_MOVE_SOURCES[name] for name in names}
    previous["identity"] = identity(previous)
    if saved == previous:
        return {**run, "identity": previous["identity"]}
    if dataset in ("MovieLens", "KuaiRec", "Coat"):
        previous = deepcopy(run)
        geometry = previous["config"]["model"]
        user_mode = "zero" if dataset == "KuaiRec" else "projected"
        if (geometry.pop("input_type", "catalog") == "catalog"
                and geometry.pop("user_features", "projected") == user_mode
                and geometry.pop("catalog_items", None) == (3327 if user_mode == "zero" else None)):
            sources = _INITIAL_RELEASE_SOURCES.copy()
            if user_mode == "zero":
                sources["kuairec.py"] = "41d5a709246219bd4e537fca4b6c85bd2d072622d4575e449359ec63a727cbd6"
                sources["kuairec_models.py"] = sources["models.py"]
            previous["implementation_sha256"] = sources
            previous.pop("identity")
            previous["identity"] = identity(previous)
            if saved == previous:
                return {**run, "identity": previous["identity"], "initial_release_config": previous["config"]}
    raise ValueError("run identity collision or changed inputs; prepare this exact config/data/seed first")


def prepare(profile, data_root, output_root, *, seed=0):
    run = check(profile, data_root, seed=seed)
    with locked(output_root):
        binding = Path(output_root) / "run.json"
        if not binding.exists() and any(p.name != LOCK_NAME for p in Path(output_root).iterdir()):
            raise ValueError("foreign nonempty output directory; choose a fresh run root")
        deep_check(run, data_root)
        if binding.exists():
            run = _bound_run(run, read(binding))
        else:
            put(binding, run, immutable=True)
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
    if not path.exists():
        raise ValueError("run identity missing or changed; prepare this exact config/data/seed first")
    run = _bound_run(run, read(path))
    return run, load_data(run, data_root)




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
    if bundle["dataset"] == "YahooR3":
        from .environments.yahoor3 import build_adapter
        return build_adapter(bundle, root)
    if bundle["dataset"] == "KuaiRand":
        from .environments.kuairand import build_adapter
        return build_adapter(bundle, root)
    from .environments.common import build_adapter
    return build_adapter(bundle, root)


def deep_check(run, root):
    """Decode shapes/labels/units/native assets without a model or env reset."""
    data = load_data(run, root)
    validate_units(run["bundle"]["units"]["training"])
    build_evaluation_adapter(run["bundle"], root)
    if run["bundle"]["dataset"] in ("MovieLens", "Coat", "YahooR3"):
        units = run["bundle"]["units"]["evaluation"]
        validate_units(units)
        if units["mapping"] != "affine" or units["offset"] != 0. or units["scale"] != 5.:
            raise ValueError("rating rollout units must preserve native raw rating/5")
    elif run["bundle"]["dataset"] == "KuaiRec":
        if run["bundle"]["units"]["evaluation"] != {"mapping": "item_train_range_minus_response_min"}:
            raise ValueError("KuaiRec rollout requires its frozen item-specific TRAIN response transform")
    elif run["bundle"]["dataset"] == "KuaiRand":
        if run["bundle"]["units"]["evaluation"] != {"mapping": "identity"}:
            raise ValueError("KuaiRand rollout uses native feedback directly")
    return data




def evaluate(profile, data_root, output_root, *, seed=0, device="auto", adapter_factory=None):
    """Closed DEV evaluation. No teacher forcing, repeat exclusion or terminal mask loss."""
    import numpy as np
    import torch
    from .models import make_model
    from .training import choose_device, load_checkpoint, seed_everything
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
                    or (result["logits_sha256"] is not None and
                        digest(output / "development_logits.npz") != result["logits_sha256"])):
                raise ValueError("completed DEV evidence changed")
            return result
        attempt = output / "evaluation_attempt.json"
        if attempt.exists():
            raise ValueError("partial DEV attempt exists without complete result; preserve evidence and request direction, never silently reroll")
        dev = choose_device(device)
        if adapter_factory is None:
            adapter_factory = build_evaluation_adapter
        adapter = adapter_factory(run["bundle"], Path(data_root))
        capture_logits = run["config"]["dataset"] not in ("YahooR3", "KuaiRand")
        seed_everything(seed)
        model = make_model(run["config"], dev, corpus=data["corpus"], features=data["features"])
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
                            if capture_logits:
                                catalog.append(logits[j].float().cpu().numpy())
                                positions.append([record["raw_user"], t])
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
        logits_sha256 = None
        if capture_logits:
            with (output / "development_logits.npz").open("wb") as stream:
                np.savez_compressed(stream, original_real_logits=np.asarray(catalog, dtype=np.float32),
                    raw_user_and_past_executions=np.asarray(positions, dtype=np.int64))
            logits_sha256 = digest(output / "development_logits.npz")
        total_steps = sum(e["length"] for e in episodes); total_reward = sum(e["cumulative_reward"] for e in episodes)
        metrics = dict(
            mean_cumulative_reward=total_reward / len(episodes), pooled_reward_per_step=total_reward / total_steps,
            mean_length=total_steps / len(episodes))
        if hasattr(adapter, "summarize"):
            metrics = adapter.summarize(episodes)
        result = dict(status="development_complete", identity=run["identity"], users=len(episodes),
            **metrics, raw_sha256=digest(output / "development_raw.json"),
            logits_sha256=logits_sha256, policy_sha256=policy_result["checkpoint_sha256"],
            final_evaluation=False, adapter=run["bundle"]["evaluation"]["adapter"], batch_size=evaluation["batch_size"], reset_block_size=evaluation["reset_block_size"],
            terminal_reward_included=True, ordinary_greedy=True, no_repeat_mask=False)
        put(result_path, result, immutable=True); return result


def dispatch_virtualtb(stage, profile, data_root, output_root=None, *, seed=0, device="auto", through="policy",
             dry_run=False, simulator_root=None, stop_after_updates=None, environment_factory=None):
    from .data.virtualtb import check, load_data
    from .environments.virtualtb import evaluate
    from .environments.virtualtb_native import load_environment
    from .training import train_continuous

    if environment_factory is None:
        environment_factory = load_environment
    run = check(profile, data_root, seed)
    data, cases = load_data(run, data_root)
    if stage == "check" or dry_run:
        return dict(status="preflight_pass", profile="virtualtb", identity=run["identity"],
                    writes=False, models_constructed=False, payload_validated=True,
                    native_environment_reset_or_step=False)
    if output_root is None:
        raise ValueError("--output-root is required")
    output = Path(output_root)
    with locked(output):
        binding = output / "run.json"
        if stage == "prepare":
            if not binding.exists() and any(path.name != LOCK_NAME for path in output.iterdir()):
                raise ValueError("foreign nonempty output directory")
            if binding.exists():
                run = _bound_run(run, read(binding))
            else:
                put(binding, run, immutable=True)
            return dict(status="prepared", identity=run["identity"], profile="virtualtb")
        if not binding.exists():
            raise ValueError("run is unprepared")
        run = _bound_run(run, read(binding))
        if stage == "resume":
            result = train_continuous(run, data, output, "scorer", device)
            if through in {"policy", "evaluate"}:
                result = train_continuous(run, data, output, "policy", device)
            if through == "evaluate":
                result = evaluate(run, cases, output, device, simulator_root, environment_factory)
            return result
        if stage == "evaluate":
            return evaluate(run, cases, output, device, simulator_root, environment_factory)
        return train_continuous(run, data, output, stage, device, stop_after_updates)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("list", "check", "prepare", "scorer", "policy", "evaluate", "resume"))
    parser.add_argument("--profile", default="movielens")
    parser.add_argument("--config", type=Path, help="shared config file; defaults to configs/prdt.json")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--simulator-root", type=Path, help="VirtualTB simulator checkout for development evaluation")
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
            if str(args.profile).rsplit("#", 1)[-1] == "virtualtb":
                if args.stage != "check" and not args.dry_run and args.output_root is None:
                    parser.error("--output-root is required")
                result = dispatch_virtualtb(args.stage, args.profile, args.data_root, args.output_root,
                                            seed=args.seed, device=args.device, through=args.through,
                                            dry_run=args.dry_run, simulator_root=args.simulator_root)
            elif args.stage == "check" or args.dry_run:
                run = check(args.profile, args.data_root, seed=args.seed)
                deep_check(run, args.data_root)
                result = dict(status="preflight_pass", identity=run["identity"], profile=run["config"]["profile_id"], writes=False, models_constructed=False, payload_validated=True, native_environment_reset_or_step=False)
            else:
                if args.output_root is None: parser.error("--output-root is required")
                kw = dict(seed=args.seed, device=args.device)
                if args.stage == "prepare": result = prepare(args.profile, args.data_root, args.output_root, seed=args.seed)
                elif args.stage == "evaluate":
                    result = evaluate(args.profile, args.data_root, args.output_root, **kw)
                else:
                    from .training import policy, scorer
                    run_args = (args.profile, args.data_root, args.output_root)
                    if args.stage == "resume":
                        result = scorer(*run_args, **kw)
                        if args.through in ("policy", "evaluate"):
                            result = policy(*run_args, **kw)
                        if args.through == "evaluate":
                            result = evaluate(*run_args, **kw)
                    else:
                        result = {"scorer": scorer, "policy": policy}[args.stage](*run_args, **kw)
        print(json.dumps(result, sort_keys=True, allow_nan=False)); return 0
    except (ValueError, RuntimeError, FileNotFoundError, KeyError) as error:
        print(json.dumps(dict(status="blocked", reason=str(error)), sort_keys=True)); return 2


if __name__ == "__main__":
    raise SystemExit(main())
