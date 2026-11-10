"""Native YahooR3 development transitions."""
import numpy as np

from ..data.bundles import load_environment_asset as _asset
from .common import MovieLensDEV


class YahooR3DEV(MovieLensDEV):
    """Native distance quitting, with the released Python slicing semantics."""

    def __init__(self, rewards, raw_users, raw_items, cases, distance, *, horizon=30):
        if not np.array_equal(raw_users, cases["raw_users"]) or not np.array_equal(
            cases["raw_users"], cases["encoded_users"]
        ):
            raise ValueError("YahooR3 world must contain exactly the ordered one-based DEV users")
        if not np.array_equal(raw_items, np.arange(1, len(raw_items) + 1)):
            raise ValueError("YahooR3 catalog must retain its one-based native column order")
        if rewards.dtype != np.float32 or np.any(rewards < 1) or np.any(rewards != np.rint(rewards)):
            raise ValueError("YahooR3 DEV rewards must be native float32 ratings in {1,2,3,4,5}")
        super().__init__(rewards, raw_users, raw_items, cases, distance, horizon=horizon, leave_threshold=120.)

    def _leave(self, action, t):
        window = self.history[t - 3:t]
        leaves = t > 0 and any(self.distance[action, previous] < self.leave_threshold for previous in window)
        return leaves, window

    def summarize(self, episodes):
        # Match the retained float32 padded reward reduction and int16 lengths.
        lengths = np.asarray([row["length"] for row in episodes], dtype=np.int16)
        rewards = np.zeros((len(episodes), self.horizon), dtype=np.float32)
        for row, episode in enumerate(episodes):
            rewards[row, :lengths[row]] = episode["rewards"]
        return dict(mean_cumulative_reward=float(rewards.sum(axis=1).mean()),
                    pooled_reward_per_step=float(rewards.sum() / int(lengths.sum())),
                    mean_length=float(lengths.mean()))


def build_adapter(bundle, root):
    evaluation = bundle["evaluation"]
    if (evaluation.get("adapter") != "yahoor3_native_v1"
            or evaluation.get("phase") != "development" or evaluation.get("horizon") != 30
            or evaluation.get("leave_window") != 3 or evaluation.get("leave_threshold") != 120.):
        raise ValueError("YahooR3 requires native H30 DEV evaluation with distance threshold 120 and window 3")
    world = _asset(bundle, root, "world")
    cases = _asset(bundle, root, evaluation["cases_asset"])
    if set(map(int, world["raw_users"])) & set(bundle["sealed_final_users"]):
        raise ValueError("YahooR3 world overlaps the sealed FINAL cohort")
    return YahooR3DEV(world["rewards"], world["raw_users"], world["raw_items"], cases,
                      _asset(bundle, root, "distance")["distance"], horizon=evaluation["horizon"])
