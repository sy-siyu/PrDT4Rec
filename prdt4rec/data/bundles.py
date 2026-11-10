"""Validated catalog bundle loading and feedback preprocessing."""
import json
import math
from pathlib import Path


def relative(root, name):
    """Reject absolute paths, parent escapes and symlinks outside the bundle."""
    root = Path(root).resolve(); name = Path(name)
    if name.is_absolute() or ".." in name.parts:
        raise ValueError("bundle asset paths must be root-relative without parent traversal")
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError("bundle symlink escapes its data root")
    return path


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


def load_environment_asset(bundle, root, name):
    """Decode a native environment array from a validated bundle path."""
    import numpy as np

    with np.load(relative(root, bundle["assets"][name]["path"]), allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def load_data(run, root):
    """Decode only the supplied TRAIN/offline-VAL assets after strict preflight."""
    import numpy as np
    from ..runner import PUBLIC_SCHEMA, REGISTRY, read
    from .trajectories import Trajectory, TrajectoryCorpus
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
            or int(a["n_items"]) - 2 != c["model"].get("catalog_items", int(a["n_items"]) - 2)
            or int(a["max_length"]) != horizon or not np.all(np.isin(a["splits"], [0, 1]))
            or len(np.unique(a["indices"])) != n or np.any(a["indices"] < 0)
            or not np.all(mask == (np.arange(horizon)[None, :] < mask.sum(1)[:, None]))
            or np.any(mask.sum(1) == 0) or not np.isfinite(feedback[mask]).all()
            or np.any(actions[mask] < 2) or np.any(actions[mask] >= int(a["n_items"]))):
        raise ValueError("invalid complete, contiguous, finite TRAIN/VAL corpus arrays")
    for name in ("users", "indices", "splits", "initial_context_actions") + (("raw_users", "preference_score") if public else ("positive_count",)):
        if a[name].shape != (n,):
            raise ValueError("corpus row metadata must be aligned")
    user_mode = c["model"].get("user_features", "projected")
    user_lower = 0 if user_mode == "zero" else 1
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
            if rule in ("cumulative_logged_feedback", "cumulative_logged_native_feedback"):
                label = a.get("label_feedback")
                if label is None or label.shape != actions.shape or not np.isfinite(label[mask]).all():
                    raise ValueError("cumulative labels require separate finite label_feedback on all executed actions")
                if c["dataset"] == "YahooR3":
                    from .yahoor3 import preference_scores
                    scores[split] = preference_scores(label, mask)
                elif c["dataset"] == "KuaiRand":
                    from .kuairand import preference_scores
                    scores[split] = preference_scores(label, mask)
                else:
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
    required_features = {"item_vectors"} if user_mode == "zero" else {"user_vectors", "item_vectors"}
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
            if c["dataset"] in ("YahooR3", "KuaiRand") and value != 1.:
                raise ValueError("native cumulative-feedback pairs must be strict and stored preferred first")
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
    roles = json.loads(relative(root, asset["path"]).read_text())
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
