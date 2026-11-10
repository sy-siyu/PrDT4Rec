"""Deterministic policy rollouts in the native VirtualTB simulator."""
import numpy as np
import torch

from .. import runner
from ..models import make_model
from ..training import choose_device
from .virtualtb_native import reset_seed


def rollout(environment, model, profile, seed, device):
    observation = environment.reset(profile, seed)
    observations, actions, rewards = [], [], []
    terminated = truncated = False
    while not (terminated or truncated):
        observations.append(np.asarray(observation, dtype=np.float32).copy())
        length = len(observations)
        action_history = np.zeros((length, 27), dtype=np.float32)
        if actions:
            action_history[:length - 1] = np.stack(actions)
        obs = torch.from_numpy(np.stack(observations)).unsqueeze(0).to(device)
        act = torch.from_numpy(action_history).unsqueeze(0).to(device)
        mask = torch.ones(1, length, dtype=torch.bool, device=device)
        with torch.no_grad():
            action = model(obs, act, mask).mean[0, -1].cpu().numpy().astype(np.float32)
        actions.append(action.copy())
        observation, reward, terminated, truncated = environment.step(action)
        rewards.append(reward)
    return dict(actions=np.asarray(actions).tolist(), rewards=rewards, length=len(rewards),
                return_raw_clicks=float(sum(rewards)), terminated=terminated, truncated=truncated)


def evaluate(run, cases, output, device_name, simulator_root, environment_factory):
    config = run["config"]
    policy_path = output / "policy.pt"
    rows_path = output / "development_episodes.json"
    result_path = output / "evaluation.json"
    attempt = output / "evaluation_attempt.json"
    binding = dict(identity=run["identity"], policy_sha256=runner.digest(policy_path))
    if result_path.exists():
        result = runner.read(result_path)
        if (any(result.get(key) != value for key, value in binding.items())
                or result.get("raw_sha256") != runner.digest(rows_path)):
            raise ValueError("completed VirtualTB development evidence changed")
        return result
    if attempt.exists():
        raise ValueError("partial VirtualTB development evaluation exists; use a fresh output for a new run")
    device = choose_device(device_name)
    state = torch.load(policy_path, map_location="cpu", weights_only=False)
    if state["identity"] != run["identity"] or state["updates"] != config["policy"]["updates"]:
        raise ValueError("VirtualTB evaluation requires the complete policy checkpoint for this run")
    model = make_model(config, device)
    model.load_state_dict(state["model"])
    model.eval()
    environment = environment_factory(simulator_root)
    runner.put(attempt, binding, immutable=True)
    rows = []
    domain = config["evaluation"]["reset_seed_domain"]
    for ordinal, (profile, digest) in enumerate(zip(cases["profiles"], cases["profile_sha256"])):
        seed = reset_seed(domain, str(digest))
        row = rollout(environment, model, profile, seed, device)
        rows.append(dict(row, profile_ordinal=ordinal, profile_sha256=str(digest), environment_seed=seed))
    returns = np.asarray([r["return_raw_clicks"] for r in rows], dtype=np.float64)
    lengths = np.asarray([r["length"] for r in rows], dtype=np.float64)
    pooled = float(returns.sum() / lengths.sum())
    result = dict(status="complete", episodes=len(rows), pages=int(lengths.sum()),
                  mean_cumulative_raw_clicks=float(returns.mean()), pooled_clicks_per_page=pooled,
                  mean_pages=float(lengths.mean()), CTR=pooled / 10.0,
                  natural_terminations=sum(r["terminated"] for r in rows),
                  h30_truncations=sum(r["truncated"] for r in rows), final_rollouts=0)
    runner.put(rows_path, rows, immutable=True)
    result.update(binding, raw_sha256=runner.digest(rows_path))
    runner.put(result_path, result, immutable=True)
    return result
