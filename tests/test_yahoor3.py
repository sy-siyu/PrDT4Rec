"""Synthetic YahooR3 fixtures exercising source preparation and the shared launcher."""
import json

import numpy as np
import pytest
import torch

from prdt4rec import runner
from prdt4rec.objective import global_softmax_weights
from prdt4rec.yahoor3 import YahooR3DEV, convert_bundle, preference_scores


def test_native_distance_boundary_and_literal_negative_slice():
    users = np.array([4])
    items = np.arange(1, 5)
    cases = dict(raw_users=users, encoded_users=users, initial_context_actions=np.array([5]))
    matrix = np.array([[1, 2, 3, 4]], dtype=np.float32)
    distance = np.full((4, 4), 120.)
    distance[2, 0] = distance[0, 2] = 119.
    actor = YahooR3DEV(matrix, users, items, cases, distance)
    assert actor.reset(4, 5) == 5
    for action in (2, 3, 4):
        reward, done, info = actor.step(action)
        assert reward == action - 1 and not done
    # t=2, previous[-1:2] ignores item 0, where the distance is only 119.
    assert info["actual_window"] == [1]
    reward, done, info = actor.step(4)
    assert (reward, done, info["actual_window"]) == (3., True, [0, 1, 2])
    assert actor.feedback(reward, 4) == .6
    with pytest.raises(ValueError, match="DEV"):
        actor.reset(3, 5)


def _source(tmp_path):
    config = runner.read(runner.PUBLIC_CONFIG)
    recipe = config["datasets"]["yahoor3"]
    recipe["model"].update(hidden_dim=8, layers=1, heads=2, dropout=.2, conv_window=3)
    recipe["scorer"].update(hidden_dim=8, layers=1, heads=2, dropout=.2, epochs=2, batch_size=2,
                            checkpoint_interval_updates=2)
    recipe.setdefault("policy", {}).update(epochs=2, draws_per_epoch=5, batch_size=2,
                                           cosine_horizon_updates=8, checkpoint_interval_updates=2)
    recipe["evaluation"].update(users=1, batch_size=1, reset_block_size=1)
    recipe["data"].update(split_counts=dict(train=5, validation=2), pair_counts=dict(train=3, validation=1))
    config_path = tmp_path / "config.json"
    runner.put(config_path, config)
    profile = f"{config_path}#yahoor3"

    source = tmp_path / "source"
    source.mkdir()
    inputs = [[2, 5], [3, 4], [1, 2, 5], [1, 1], [5, 5, 4], [4, 5], [1, 2]]
    normalized = np.zeros((len(inputs), 30), dtype=np.float32)
    actions = np.zeros_like(normalized, dtype=np.int64)
    mask = np.zeros_like(normalized, dtype=bool)
    for row, rewards in enumerate(inputs):
        mask[row, :len(rewards)] = True
        normalized[row, :len(rewards)] = np.asarray(rewards, dtype=np.float32) / 5.
        actions[row, :len(rewards)] = np.arange(len(rewards)) + 2
    labels = np.asarray([row[:length].sum() for row, length in zip(normalized, mask.sum(axis=1))],
                        dtype=np.float32)
    assert labels[0] < labels[1]  # exactly tied raw sums, strictly ordered executed float32 sums
    np.savez(source / "offline.npz", actions=actions, feedback=normalized, mask=mask,
             users=np.array([1] * 5 + [2] * 2), indices=np.arange(7), splits=np.array([0] * 5 + [1] * 2),
             initial_context_actions=np.full(7, 5), n_users=np.asarray(5), n_items=np.asarray(6),
             max_length=np.asarray(30), positive_count=labels)
    rng = np.random.default_rng(120)
    features = rng.normal(size=(5, 40)).astype(np.float32)
    features[0] = 0.
    np.savez(source / "representation.npz", user_vectors=features,
             item_vectors=rng.normal(size=(4, 40)).astype(np.float32))
    np.savez(source / "pairs.npz", train_preferred=np.array([1, 2, 4]), train_rejected=np.array([0, 3, 3]),
             validation_preferred=np.array([5]), validation_rejected=np.array([6]))
    np.savez(source / "lineage.npz", global_episode_ids=np.arange(7), collector_codes=np.arange(7) % 3)
    contexts = np.random.default_rng(recipe["evaluation"]["context_seed"]).integers(0, 4, size=1, dtype=np.int32)
    np.savez(source / "dev_cases.npz", raw_user_ids=np.array([3]), encoded_user_ids=np.array([2]),
             initial_context_item_columns=contexts)
    matrix = tmp_path / "ratings.ascii"
    # A deliberately unparsable row for every user except DEV. Passing the
    # converter proves it did not decode TRAIN or FINAL response rows.
    matrix.write_text("sealed\nsealed\n1 3 5 4\n" + "sealed\n" * (15400 - 3))
    distance = tmp_path / "distance.npy"
    np.save(distance, np.zeros((4, 4), dtype=np.float64))
    runner.put(source / "bundle.json", dict(status="shared_inputs_frozen", frozen_before_fit=True,
              assets={p.name: dict(sha256=runner.digest(p)) for p in source.glob("*.npz")},
              source=dict(matrix_sha256=runner.digest(matrix))))
    cohorts = tmp_path / "cohorts.json"
    runner.put(cohorts, dict(schema="prdt4rec-yahoor3-cohorts-v1", frozen_before_simulator_outcomes=True,
              roles={name: dict(encoded_zero_based=[i]) for i, name in enumerate(
                  ("policy_train", "offline_validation", "policy_dev", "final_sealed"))}))
    return profile, source, dict(cohorts=cohorts, matrix=matrix, distance=distance)


def test_yahoor3_conversion_training_resume_and_evaluation(tmp_path):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        profile, source, assets = _source(tmp_path)
        data = tmp_path / "bundle"
        convert_bundle(source, data, profile=profile, **assets)
        with pytest.raises(FileExistsError):
            convert_bundle(source, data, profile=profile, **assets)
        checked = runner.check(profile, data, seed=2)
        loaded = runner.deep_check(checked, data)
        assert loaded["corpus"].get(1).positive_count > loaded["corpus"].get(0).positive_count
        bundle = checked["bundle"]
        label_data = runner.load_asset(data, bundle, "corpus")
        assert np.array_equal(label_data["preference_score"], preference_scores(label_data["feedback"], label_data["mask"]))
        assert runner.load_asset(data, bundle, "world")["raw_users"].tolist() == [3]
        common = dict(seed=2, device="cpu")
        full, resumed = tmp_path / "full", tmp_path / "resumed"
        for output in (full, resumed):
            runner.prepare(profile, data, output, seed=2)
        uninterrupted = runner.scorer(profile, data, full, **common)
        partial = runner.scorer(profile, data, resumed, stop_after_updates=1, **common)
        assert partial["updates"] == 1
        restarted = runner.scorer(profile, data, resumed, **common)
        assert uninterrupted["new_scorer_updates"] == restarted["new_scorer_updates"] == 4
        with np.load(full / "scores.npz") as reference, np.load(resumed / "scores.npz") as restored:
            assert np.array_equal(reference["scores"], restored["scores"])
            stored = global_softmax_weights(torch.from_numpy(reference["scores"]).double(), .1).float().numpy()
        expected = stored.astype(np.float64)
        expected /= expected.sum(dtype=np.float64)
        q, _ = runner.probabilities(checked, loaded, data, full)
        assert np.array_equal(q.numpy(), expected)
        runner.policy(profile, data, full, **common)
        assert runner.policy(profile, data, resumed, stop_after_updates=4, **common)["updates"] == 4
        result = runner.policy(profile, data, resumed, **common)
        for key, value in torch.load(full / "policy_final.pt", weights_only=False)["model"].items():
            assert torch.equal(value, torch.load(resumed / "policy_final.pt", weights_only=False)["model"][key])
        state = torch.load(resumed / "policy_final.pt", weights_only=False)["state"]
        assert state["updates"] == 6 and state["draws"] == 10
        assert result["status"] == "policy_complete"
        evaluated = runner.evaluate(profile, data, resumed, **common)
        assert evaluated["users"] == 1 and evaluated["mean_length"] == 2
        assert evaluated["final_evaluation"] is False and evaluated["terminal_reward_included"] is True
        raw = json.loads((resumed / "development_raw.json").read_text())["episodes"][0]
        assert raw["raw_user"] == 3 and len(raw["rewards"]) == 2
        assert evaluated["mean_cumulative_reward"] == sum(raw["rewards"])
    finally:
        torch.set_num_threads(old_threads)
