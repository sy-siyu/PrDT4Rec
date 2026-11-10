"""Synthetic checks for the retained KuaiRand execution contract."""

from copy import deepcopy

import numpy as np
import pytest
import torch

from prdt4rec import runner, training
from prdt4rec.data.kuairand import convert_bundle, preference_scores
from prdt4rec.environments.kuairand import KuaiRandDEV


@pytest.fixture
def small_bundle(tmp_path):
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    config = deepcopy(runner.load_config("kuairand"))
    config["model"].update(hidden_dim=8, layers=1, heads=2, conv_window=3, users=5, catalog_items=4)
    config["scorer"].update(hidden_dim=8, layers=1, heads=2, batch_size=2, epochs=2,
                            checkpoint_interval_updates=2)
    config["policy"].update(batch_size=2, draws_per_epoch=5, epochs=2, optimizer_updates=6,
                            trajectory_draws=10, cosine_horizon_updates=12, checkpoint_interval_updates=3)
    config["data"].update(bundle_identity="synthetic", split_counts=dict(train=3, validation=2, test=0),
                          pair_counts=dict(train=3, validation=1))
    config["evaluation"].update(users=2, batch_size=2, reset_block_size=2)
    profile = tmp_path / "profile.json"
    runner.put(profile, config)

    source = tmp_path / "retained"
    source.mkdir()
    feedback = np.zeros((5, 30), dtype=np.float32)
    actions = np.zeros((5, 30), dtype=np.int64)
    mask = np.zeros((5, 30), dtype=bool)
    for i, prefix in enumerate([[.8, .7], [.6, .3], [.2, .2], [.8, .5], [.1, .3]]):
        feedback[i, :len(prefix)] = prefix
        actions[i, :len(prefix)] = [2, 3]
        mask[i, :len(prefix)] = True
    np.savez(source / "offline.npz", actions=actions, feedback=feedback, mask=mask,
             users=np.array([1, 1, 1, 2, 2]), indices=np.arange(5),
             splits=np.array([0, 0, 0, 1, 1]), initial_context_actions=np.full(5, 5),
             n_users=np.asarray(6), n_items=np.asarray(6), max_length=np.asarray(30))
    rng = np.random.default_rng(19)
    users = rng.normal(size=(6, 40)).astype(np.float32)
    users[0] = 0
    np.savez(source / "representation.npz", user_vectors=users,
             item_vectors=rng.normal(size=(4, 40)).astype(np.float32))
    np.savez(source / "pairs.npz", train_preferred=np.array([0, 0, 1]), train_rejected=np.array([1, 2, 2]),
             validation_preferred=np.array([3]), validation_rejected=np.array([4]))
    np.savez(source / "lineage.npz", global_episode_ids=np.arange(5), collector_codes=np.arange(5) % 4)
    np.savez(source / "dev_cases.npz", encoded_user_ids=np.array([2, 3]),
             initial_context_item_columns=np.array([1, 0]))
    runner.put(source / "bundle.json", dict(identity="synthetic",
        assets={path.name: dict(sha256=runner.digest(path)) for path in source.glob("*.npz")}))

    cohorts = tmp_path / "cohorts.json"
    runner.put(cohorts, dict(schema="prdt4rec-kuairand-cohorts-v1", frozen_before_new_simulator_outcomes=True,
        roles={key: {"encoded_zero_based": values} for key, values in
               dict(policy_train=[0], offline_validation=[1], policy_dev=[2, 3], final_sealed=[4]).items()}))
    world = tmp_path / "dev_only.npz"
    np.savez(world, raw_users=np.array([2, 3]), raw_items=np.arange(4),
             rewards=np.array([[.1, .2, .3, .4], [.6, .7, .8, .9]], dtype=np.float32))
    tags = tmp_path / "items.csv"
    tags.write_text('video_id,tag\n0,"1,2"\n1,3\n2,\n3,\n', encoding="utf-8")
    root = convert_bundle(source, tmp_path / "data", cohorts=cohorts, world=world, categories=tags, profile=profile)
    try:
        yield profile, root, tmp_path
    finally:
        torch.set_num_threads(before)


def test_native_label_prefix_sum():
    values = np.asarray([[.5, .2, .8, .7, .1, 0., .6, .3, .2, .7]], dtype=np.float32)
    mask = np.arange(10)[None, :] < 7
    score = preference_scores(values, mask)
    assert score[0] == float(values[0, :7].sum())
    with pytest.raises(ValueError, match="float32"):
        preference_scores(values.astype(np.float64), mask)


def test_native_category_suffix_includes_missing_tag_and_terminal_reward():
    rewards = np.arange(31, dtype=np.float32)[None, :] / 30
    tags = np.full((31, 3), -2, dtype=np.int64)
    tags[:, 0] = np.arange(31)
    tags[0, 0] = tags[7, 0] = -1
    cases = dict(raw_users=np.array([9]), initial_context_actions=np.array([32]))
    adapter = KuaiRandDEV(rewards, np.array([9]), np.arange(31), cases, tags)
    with pytest.raises(ValueError, match="DEV"):
        adapter.reset(10, 32)
    assert adapter.reset(9, 32) == 32
    for encoded in range(2, 10):
        _, done, _ = adapter.step(encoded)
        assert not done
    # t=7 has previous[-3:7], so row 0 is excluded. At t=8 the
    # literal suffix includes the just-executed row 7 with its missing tag.
    reward, done, info = adapter.step(9)
    assert done and reward == float(rewards[0, 7]) and info["actual_window"] == [6, 7]
    with pytest.raises(RuntimeError, match="terminal"):
        adapter.step(2)


def test_native_pipeline_resume_and_eval(small_bundle):
    profile, data, tmp = small_bundle
    direct, resumed = tmp / "direct", tmp / "resumed"
    assert runner.main(["check", "--dry-run", "--profile", str(profile), "--data-root", str(data)]) == 0
    for output in (direct, resumed):
        runner.prepare(profile, data, output, seed=2)
    interrupted = training.scorer(profile, data, resumed, seed=2, device="cpu", stop_after_updates=1)
    assert interrupted["updates"] == 1
    for output in (direct, resumed):
        training.scorer(profile, data, output, seed=2, device="cpu")
    with np.load(direct / "scores.npz") as expected, np.load(resumed / "scores.npz") as actual:
        assert np.array_equal(expected["scores"], actual["scores"])
        from prdt4rec.objective import global_softmax_weights
        stored = global_softmax_weights(torch.from_numpy(expected["scores"]).double(), .5).float().numpy()
        probabilities = stored.astype(np.float64)
        probabilities /= probabilities.sum(dtype=np.float64)
    training.policy(profile, data, resumed, seed=2, device="cpu", stop_after_updates=2)
    for output in (direct, resumed):
        result = training.policy(profile, data, output, seed=2, device="cpu")
        assert (result["optimizer_updates"], result["trajectory_draws"], result["valid_action_targets"]) == (6, 10, 20)
    expected = torch.load(direct / "policy_final.pt", weights_only=False)
    actual = torch.load(resumed / "policy_final.pt", weights_only=False)
    assert np.array_equal(actual["sampling_probabilities"].numpy(), probabilities)
    assert all(torch.equal(tensor, actual["model"][name]) for name, tensor in expected["model"].items())
    evaluation = runner.evaluate(profile, data, resumed, seed=2, device="cpu")
    assert evaluation["users"] == 2 and evaluation["final_evaluation"] is False
    assert evaluation["adapter"] == "kuairand_native_v1" and evaluation["terminal_reward_included"] is True
    episodes = runner.read(resumed / "development_raw.json")["episodes"]
    assert [episode["raw_user"] for episode in episodes] == [2, 3]
    assert all(2 <= episode["length"] <= 30 for episode in episodes)
