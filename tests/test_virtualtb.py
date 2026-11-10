import json

import numpy as np
import pytest
import torch

from prdt4rec import runner
from prdt4rec.data.virtualtb import export_bundle, profile_digest
from prdt4rec.environments.virtualtb_native import VirtualTBH30


@pytest.fixture
def bundle(tmp_path):
    config = runner.read(runner.PUBLIC_CONFIG)
    fixture = config["datasets"]["virtualtb"]
    fixture["model"].update(hidden_dim=8, layers=1, heads=2, conv_window=2)
    fixture["scorer"].update(hidden_dim=8, layers=1, heads=2, updates=3, pair_draws=3,
                             batch_size=1, validation_batch_size=1, checkpoint_interval_updates=1)
    fixture["policy"].update(epochs=2, batch_size=2, draws_per_epoch=3, updates=4,
                             trajectory_draws=6, cosine_horizon_updates=4, checkpoint_interval_updates=2)
    fixture["data"].update(split_counts={"train": 3, "validation": 2, "test": 0},
                           pair_counts={"train": 3, "validation": 1}, train_steps=9)
    fixture["evaluation"].update(users=2)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))

    def source(name, count, profile, clicks):
        obs = np.zeros((count, 30, 91), dtype=np.float32)
        obs[:, :3, :88] = profile
        actions = np.zeros((count, 30, 27), dtype=np.float32)
        actions[:, :3] = 0.25
        mask = np.zeros((count, 30), dtype=bool)
        mask[:, :3] = True
        rewards = np.zeros((count, 30), dtype=np.float32)
        rewards[:, :3] = np.asarray(clicks)[:, None]
        return dict(observations=obs, actions=actions, mask=mask, rewards=rewards,
                    episode_ids=np.asarray([f"{name}-{i}" for i in range(count)]))

    train = tmp_path / "train.npz"
    np.savez(train, **source("train", 3, 0.2, [1, 1, 3]))
    validation = tmp_path / "val.npz"
    np.savez(validation, **source("val", 2, 0.3, [2, 1]),
             first=np.array([0], np.int64), second=np.array([1], np.int64),
             targets=np.array([1], np.float32))
    development = tmp_path / "dev.npz"
    profiles = np.broadcast_to(np.array([[0.1], [0.7]], np.float32), (2, 88))
    np.savez(development, development_profiles=profiles,
             development_sha256=np.asarray([profile_digest(p) for p in profiles]),
             final_profiles=np.asarray([{"must not decode": True}], dtype=object))
    output = tmp_path / "bundle"
    assert export_bundle(train, validation, development, output, config_path)["status"] == "preflight_pass"
    return f"{config_path}#virtualtb", output


class NativeSimulator:
    """Legacy four-return adapter fixture with native reset on termination."""

    def __init__(self):
        self.user_leave_model = self

    def predict(self, profile):
        torch.rand(1)
        return 2 if profile[0, 0] < 0.5 else 40

    def step(self, action):
        assert np.all(np.abs(action) <= 1)
        torch.rand(1)
        self.total_c += 1
        self.rend_action = torch.FloatTensor([3, 0])
        done = self.total_c >= self._VirtualTB__leave
        if done:
            self.cur_user = np.zeros(88, dtype=np.float32)
        return np.concatenate((self.cur_user, np.zeros(3, dtype=np.float32))), 3, done, {}


def test_virtualtb_stages_checkpoint_resume_and_native_evaluation(bundle, tmp_path):
    profile, data_root = bundle
    assert runner.dispatch_virtualtb("check", profile, data_root)["models_constructed"] is False
    full, resumed = tmp_path / "full", tmp_path / "resume"
    for path in (full, resumed):
        runner.dispatch_virtualtb("prepare", profile, data_root, path)
    assert runner.dispatch_virtualtb("resume", profile, data_root, full, device="cpu")["updates"] == 4
    assert runner.dispatch_virtualtb("scorer", profile, data_root, resumed, device="cpu",
                              stop_after_updates=1)["status"] == "paused"
    assert runner.dispatch_virtualtb("resume", profile, data_root, resumed, device="cpu",
                              through="scorer")["updates"] == 3
    assert runner.dispatch_virtualtb("policy", profile, data_root, resumed, device="cpu",
                              stop_after_updates=1)["status"] == "paused"
    assert runner.dispatch_virtualtb("resume", profile, data_root, resumed, device="cpu")["draws"] == 6
    for name in ("scorer", "policy"):
        lhs, rhs = (torch.load(path / f"{name}.pt", weights_only=False) for path in (full, resumed))
        assert all(torch.equal(lhs["model"][key], rhs["model"][key]) for key in lhs["model"])
        assert torch.equal(lhs["rng"]["torch"], rhs["rng"]["torch"])
    weights = torch.load(resumed / "scores.pt", weights_only=True)["weights"]
    assert weights.dtype == torch.float64 and weights.sum().item() == pytest.approx(1)
    result = runner.dispatch_virtualtb("evaluate", profile, data_root, resumed, device="cpu",
                                environment_factory=lambda _: VirtualTBH30(NativeSimulator()))
    assert result["episodes"] == 2
    assert result["natural_terminations"] == result["h30_truncations"] == 1
    assert result["mean_cumulative_raw_clicks"] == 48
    assert result["pooled_clicks_per_page"] == 3
    assert result["final_rollouts"] == 0
    assert runner.dispatch_virtualtb("evaluate", profile, data_root, resumed, device="cpu",
                              environment_factory=lambda _: pytest.fail("repeated evaluation")) == result


def test_virtualtb_adapter_preserves_private_rng_and_terminal_clicks():
    before = torch.get_rng_state()
    profile = np.full(88, 0.1, np.float32)
    env = VirtualTBH30(NativeSimulator())
    np.testing.assert_array_equal(env.reset(profile, 4)[:88], profile)
    action = np.zeros(27, dtype=np.float32)
    env.step(action)
    observation, reward, terminal, truncated = env.step(action)
    assert reward == 3 and terminal and not truncated
    np.testing.assert_array_equal(observation[:88], profile)
    assert torch.equal(torch.get_rng_state(), before)
    with pytest.raises(RuntimeError, match="outside"):
        env.step(action)
