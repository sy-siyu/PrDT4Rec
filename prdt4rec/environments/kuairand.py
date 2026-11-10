"""Native KuaiRand development transitions."""
import numpy as np

from ..data.bundles import load_environment_asset as _asset
from .common import MatrixDEV


class KuaiRandDEV(MatrixDEV):
    """Native FreeB transitions with the original literal category-history slice."""

    def __init__(self, rewards, raw_users, raw_items, cases, categories, *, horizon=30):
        super().__init__(rewards, raw_users, raw_items, cases, horizon=horizon)
        if self.rewards.dtype != np.float32 or np.any(self.rewards > 1):
            raise ValueError("KuaiRand development requires native float32 click-proxy rewards in [0, 1]")
        if not np.array_equal(self.raw_users, cases["raw_users"]):
            raise ValueError("KuaiRand response matrix must contain only the ordered DEV rows")
        if not np.array_equal(self.raw_items, np.arange(len(self.raw_items))):
            raise ValueError("KuaiRand native item columns must be contiguous and zero based")
        categories = np.asarray(categories)
        if (categories.shape != (len(raw_items), 3) or categories.dtype.kind not in "iu"
                or np.any(categories < -2)):
            raise ValueError("KuaiRand tags require three aligned integer slots with -2 padding")
        # -1 represents a missing native tag and participates in overlap.
        self.categories = tuple(frozenset(map(int, row[row != -2])) for row in categories)
        if any(not tags for tags in self.categories):
            raise ValueError("each KuaiRand item requires a native tag set")

    def _leave(self, action, t):
        history = self.history[t - 10:t]
        tags = self.categories[action]
        return bool(t and any(tags & self.categories[old] for old in history)), history

    def feedback(self, raw_reward, encoded_action):
        return float(raw_reward)

    def summarize(self, episodes):
        rewards = np.zeros((len(episodes), self.horizon), dtype=np.float32)
        lengths = np.asarray([episode["length"] for episode in episodes], dtype=np.int16)
        for i, episode in enumerate(episodes):
            rewards[i, :lengths[i]] = episode["rewards"]
        returns = rewards.sum(axis=1)
        return dict(mean_cumulative_reward=float(returns.mean()),
                    pooled_reward_per_step=float(rewards.sum() / int(lengths.sum())),
                    mean_length=float(lengths.mean()))


def build_adapter(bundle, root):
    settings = bundle["evaluation"]
    if (bundle["dataset"] != "KuaiRand" or settings["adapter"] != "kuairand_native_v1"
            or settings.get("phase") != "development" or settings.get("horizon") != 30
            or settings.get("leave_window") != 10):
        raise ValueError("KuaiRand requires its H30 development-only native adapter")
    world = _asset(bundle, root, "world")
    cases = _asset(bundle, root, settings["cases_asset"])
    categories = _asset(bundle, root, "categories")["categories"]
    return KuaiRandDEV(world["rewards"], world["raw_users"], world["raw_items"], cases,
                       categories, horizon=settings["horizon"])
