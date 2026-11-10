from __future__ import annotations
from typing import Any
import random
from copy import deepcopy
import hashlib
import math
from pathlib import Path
import numpy as np
import torch
from .data.bundles import load_asset, normalize
from .runner import PUBLIC_SCHEMA, VERSION, digest, identity, locked, open_run, put, read, save

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def choose_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but torch.backends.mps.is_available() is false")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no CUDA device is available")
    return device


def _rng_state(loader_generator: torch.Generator) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "loader": loader_generator.get_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "mps": torch.mps.get_rng_state() if torch.backends.mps.is_available() else None,
    }


def _restore_rng_state(state: dict[str, Any], loader_generator: torch.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    loader_generator.set_state(state["loader"].cpu())
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([tensor.cpu() for tensor in state["cuda"]])
    if state.get("mps") is not None and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"].cpu())


def moved(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def collate(rows, run, *, scorer=False):
    import torch
    from .data.trajectories import collate_trajectories
    batch = collate_trajectories(rows)
    if run["config"]["model"].get("user_features") == "zero":
        batch["users"].zero_()
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
    valid_configs = [run["config"]]
    if "initial_release_config" in run:
        valid_configs.append(run["initial_release_config"])
    if (saved.get("version") != VERSION or saved.get("identity") != run["identity"] or saved.get("config") not in valid_configs
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
    return dict(version=VERSION, identity=run["identity"], stage=stage, device_type=device.type,
        model=model.state_dict(), optimizer=optimizer.state_dict(), rng=_rng_state(generator),
        state=state, config=run["config"], **extra)


def _score_output(run, data, model, root, output, device, generator):
    import numpy as np
    import torch
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
    if dataset in ("YahooR3", "KuaiRand"):
        return F.softplus(-logits).mean()
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
    if run["config"]["dataset"] in ("Coat", "YahooR3", "KuaiRand"):
        # These recipes iterate a non-shuffled, unseeded validation loader.
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
        if run["config"]["dataset"] not in ("YahooR3", "KuaiRand"):
            model.train(was_training)
    return dict(validation_pair_nll=nll/count if count else None,
                validation_pair_accuracy=correct/strict_count if strict_count else None,
                validation_pairs=count,validation_ties=ties)


def scorer(profile, data_root, output_root, *, seed=0, device="auto", stop_after_updates=None):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from .models import make_model
    from .objective import bradley_terry_loss
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
        model = make_model(run["config"], dev, scorer=True, corpus=data["corpus"], features=data["features"])
        cfg = run["config"]["scorer"]
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
        # These retained scorers enter training mode at invocation, then stay
        # in evaluation mode after the first validation pass of that invocation.
        native_validation_mode = run["config"]["dataset"] in ("YahooR3", "KuaiRand")
        while state["epoch"] < cfg["epochs"]:
            if state["order"] is None:
                if cfg["pair_order_rng"] == "numpy_epoch_seed":
                    state["order"] = torch.from_numpy(np.random.default_rng(seed + 10000 * (state["epoch"] + 1)).permutation(len(high)))
                    reg_np = np.random.default_rng(seed + 991 + 10000 * (state["epoch"] + 1))
                elif cfg["pair_order_rng"] == "torch_randperm":
                    state["order"] = torch.randperm(len(high), generator=gen)
                else:
                    state["order"] = torch.cat(list(DataLoader(range(len(high)), batch_size=cfg["batch_size"], shuffle=True, generator=gen)))
            order = state["order"]
            if not native_validation_mode:
                model.train()
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
    if w["precision"] == "saved_global_float32_then_TRAIN_numpy_float64_normalized":
        values = q.numpy()
        values /= values.sum(dtype=np.float64)
    else:
        q /= q.sum()
    source = q.clone(); ids = list(map(int, cached["indices"]))
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
    from .models import make_model
    from .objective import per_trajectory_mean_action_nll
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
        model = make_model(run["config"], dev, corpus=data["corpus"], features=data["features"])
        cfg = run["config"]["policy"]
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


def scorer_schedule(seed, scorer, ids, first, second):
    from .data.virtualtb import rank_pair
    if seed == 0:
        order = sorted(range(len(first)), key=lambda i: rank_pair(
            scorer["exposure_digest_domain"], str(ids[first[i]]), str(ids[second[i]])))
        order = np.asarray(order, dtype=np.int64)
    else:
        order = np.random.default_rng(scorer["pair_order_seed_offset"] + seed).permutation(len(first))
    return order[:scorer["pair_draws"]].reshape(scorer["updates"], scorer["batch_size"])


def policy_schedule(seed, policy, n):
    permutation = np.random.default_rng(policy["sampler_seed_offset"] + seed).permutation(n)
    draws = policy["draws_per_epoch"]
    return np.stack([permutation[(np.arange(draws) + block * draws) % n]
                     for block in range(policy["epochs"])])


def pair_scores(model, corpus, pairs, indices, device):
    first, second, targets = (array[indices] for array in pairs)
    scores = model(*corpus.batch(np.concatenate((first, second)), device))
    return scores[:len(first)], scores[len(first):], torch.from_numpy(targets).to(device)


def validation_loss(model, corpus, pairs, device, batch_size):
    from .objective import bradley_terry_target_loss
    model.eval()
    count, total = len(pairs[0]), 0.0
    with torch.no_grad():
        for offset in range(0, count, batch_size):
            batch = slice(offset, offset + batch_size)
            loss = bradley_terry_target_loss(*pair_scores(model, corpus, pairs, batch, device))
            total += float(loss) * len(pairs[0][batch])
    return total / count


def train_continuous(run, data, output, stage, device_name, stop_after_updates=None):
    from . import runner
    from .continuous import per_trajectory_continuous_nll
    from .models import make_model
    from .objective import bradley_terry_target_loss
    config, seed = run["config"], run["seed"]
    device, values = choose_device(device_name), config[stage]
    path = output / f"{stage}.pt"
    torch.manual_seed(seed)
    model = make_model(config, device, scorer=stage == "scorer")
    optimizer = torch.optim.AdamW(model.parameters(), lr=values["learning_rate"],
                                  weight_decay=values["weight_decay"])
    scheduler = (None if stage == "scorer" else torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=values["cosine_horizon_updates"], eta_min=values["minimum_learning_rate"]))
    generator = torch.Generator(device="cpu")
    state = dict(identity=run["identity"], device=str(device), stage=stage, updates=0, draws=0,
                 tokens=0, best=None, best_loss=math.inf, best_update=None, validation_history=[])
    if path.exists():
        state = torch.load(path, map_location="cpu", weights_only=False)
        if (state["identity"], state["device"], state["stage"]) != (run["identity"], str(device), stage):
            raise ValueError("VirtualTB checkpoint does not match this run and device")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(state["scheduler"])
        _restore_rng_state(state["rng"], generator)
        model.train(state["model_training"])

    corpus, pairs = data["train"]
    n = len(corpus.episode_ids)
    if stage == "scorer":
        schedule = scorer_schedule(seed, values, corpus.episode_ids, *pairs[:2])
    else:
        frozen = torch.load(output / "scores.pt", map_location="cpu", weights_only=True)
        if (frozen["identity"] != run["identity"] or frozen["weights"].shape != (n,)
                or not torch.isfinite(frozen["weights"]).all()):
            raise ValueError("VirtualTB weights do not match this run")
        weights = frozen["weights"]
        schedule = [block[i:i + values["batch_size"]]
                    for block in policy_schedule(seed, values, n)
                    for i in range(0, len(block), values["batch_size"])]

    def checkpoint():
        runner.save(path, {**state, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                          "rng": _rng_state(generator), "model_training": model.training,
                          "scheduler": scheduler.state_dict() if scheduler else None})

    cap = values["updates"] if stop_after_updates is None else min(values["updates"], stop_after_updates)
    for offset in range(state["updates"], cap):
        batch = schedule[offset]
        if stage == "scorer":
            left, right, targets = pair_scores(model, corpus, pairs, batch, device)
            loss = (bradley_terry_target_loss(left, right, targets)
                    + values["lambda_score"] * .5 * (left.square().mean() + right.square().mean()))
        else:
            obs, actions, _, mask = corpus.batch(batch, device)
            nll = per_trajectory_continuous_nll(model(obs, actions, mask), actions, mask)
            batch_weights = weights.index_select(0, torch.from_numpy(batch)).to(device=device, dtype=nll.dtype)
            loss = (n * batch_weights * nll).mean()
            state["tokens"] += int(mask.sum())
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite VirtualTB {stage} loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), values["gradient_clip"])
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(f"nonfinite VirtualTB {stage} gradient")
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        state["updates"] = offset + 1
        state["draws"] += len(batch)
        boundary = state["updates"] % values["checkpoint_interval_updates"] == 0
        if stage == "scorer" and boundary:
            # The retained scorer stays in evaluation mode after its first
            # offline validation; subsequent updates retain disabled dropout.
            val = validation_loss(model, *data["validation"], device, values["validation_batch_size"])
            if not math.isfinite(val):
                raise FloatingPointError("nonfinite VirtualTB offline validation loss")
            state["validation_history"].append({"update": offset + 1, "loss": val})
            if val < state["best_loss"]:
                state["best_loss"], state["best_update"] = val, offset + 1
                state["best"] = deepcopy({key: tensor.detach().cpu() for key, tensor in model.state_dict().items()})
        if boundary or state["updates"] == cap:
            checkpoint()
    complete = state["updates"] == values["updates"]
    summary = dict(status="complete" if complete else "paused", updates=state["updates"],
                   draws=state["draws"], valid_action_tokens=state["tokens"])
    if complete and stage == "scorer":
        model.load_state_dict(state["best"])
        model.eval()
        scores = []
        with torch.no_grad():
            for start in range(0, n, 256):
                ids = np.arange(start, min(start + 256, n))
                scores.append(model(*corpus.batch(ids, device)).cpu())
        scores = torch.cat(scores).double()
        if not torch.isfinite(scores).all():
            raise FloatingPointError("nonfinite VirtualTB frozen scorer output")
        weights = torch.softmax(config["weighting"]["beta"] * (scores - scores.max()), dim=0)
        runner.save(output / "scores.pt", {"identity": run["identity"], "scores": scores, "weights": weights})
        summary.update(best_update=state["best_update"], validation_history=state["validation_history"])
    runner.put(output / f"{stage}.json", summary)
    return summary

