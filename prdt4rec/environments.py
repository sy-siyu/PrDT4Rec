"""Native FreeB transition adapters for catalog development evaluation.

Small source-equivalent adapters: ML distance75/window7; Coat distance6/window7;
Kuai category count>1/window9. Literal negative Python slices and terminal rewards are intentional.
The case table supplies retained native reset states; no corpus or feature fit.
"""
from collections import Counter
from pathlib import Path
import numpy as np


def _asset(bundle, root, key):
    path = (Path(root) / bundle['assets'][key]['path']).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError('environment asset escapes data root')
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


class MatrixDEV:
    def __init__(self, rewards, raw_users, raw_items, cases, *, horizon=40):
        self.rewards = np.asarray(rewards)
        self.raw_users = np.asarray(raw_users)
        self.raw_items = np.asarray(raw_items)
        if (self.rewards.shape != (len(self.raw_users), len(self.raw_items)) or
                not np.isfinite(self.rewards).all() or np.any(self.rewards < 0) or
                np.any(self.rewards > 5) or type(horizon) is not int or not 1 <= horizon <= 40 or
                len(np.unique(self.raw_users)) != len(self.raw_users) or
                len(np.unique(self.raw_items)) != len(self.raw_items)):
            raise ValueError('finite raw0..5 response matrix, unique IDs and explicit native horizon <=40 required')
        self.user_index = {int(u): i for i, u in enumerate(self.raw_users)}
        users = cases['raw_users']; contexts = cases['initial_context_actions']
        if len(users) != len(contexts) or len(np.unique(users)) != len(users):
            raise ValueError('one immutable reset context per DEV user required')
        self.contexts = dict(zip(map(int, users), map(int, contexts)))
        if any(u not in self.user_index for u in self.contexts) or any(
                c < 2 or c >= len(self.raw_items) + 2 for c in self.contexts.values()):
            raise ValueError('DEV cases do not align with response/catalog IDs')
        self.horizon = horizon
        self.done = True

    def fork(self):
        """Independent episode state sharing read-only response assets."""
        from copy import copy
        other = copy(self)
        other.done = True
        other.history = []
        return other

    def reset(self, raw_user, saved_context):
        if int(raw_user) not in self.contexts or self.contexts[int(raw_user)] != int(saved_context):
            raise ValueError('only hash-bound DEV users and retained reset states are accessible')
        self.user = self.user_index[int(raw_user)]
        self.history = []
        self.done = False
        return int(saved_context)

    def step(self, encoded_action):
        if self.done:
            raise RuntimeError('reset required; terminal action cannot be repeated after termination')
        action = int(encoded_action) - 2
        if isinstance(encoded_action, bool) or not 0 <= action < len(self.raw_items):
            raise ValueError('only real encoded catalog actions are allowed')
        t = len(self.history)
        natural, window = self._leave(action, t)
        cap = t >= self.horizon - 1
        reward = float(self.rewards[self.user, action])
        self.history.append(action)
        self.done = bool(natural or cap)
        return reward, self.done, dict(past_count=t, actual_window=list(window),
            reason=f'H{self.horizon}_cap' if cap else 'natural' if natural else 'continue',
            raw_item=int(self.raw_items[action]))


class MovieLensDEV(MatrixDEV):
    def __init__(self, rewards, raw_users, raw_items, cases, distance, *, horizon=40, leave_threshold=75):
        super().__init__(rewards, raw_users, raw_items, cases, horizon=horizon)
        self.leave_threshold = leave_threshold
        self.distance = np.asarray(distance)
        if self.distance.shape != (len(raw_items), len(raw_items)) or not np.isfinite(self.distance).all() or np.any(self.distance < 0):
            raise ValueError('aligned finite native item-distance matrix required')

    def _leave(self, action, t):
        window = self.history[t-7:t] if t else []
        return bool(t and any(self.distance[action, prior] < self.leave_threshold for prior in window)), window

    def feedback(self, raw_reward, encoded_action):
        return float(raw_reward) / 5.


class CoatDEV(MovieLensDEV):
    def __init__(self, rewards, raw_users, raw_items, cases, distance, *, horizon=30):
        super().__init__(rewards, raw_users, raw_items, cases, distance, horizon=horizon, leave_threshold=6)


class KuaiRecDEV(MatrixDEV):
    def __init__(self, rewards, raw_users, raw_items, cases, categories, transform, *, horizon=40):
        super().__init__(rewards, raw_users, raw_items, cases, horizon=horizon)
        cats = np.asarray(categories)
        if cats.ndim != 2 or len(cats) != len(raw_items) or not np.issubdtype(cats.dtype, np.integer):
            raise ValueError('aligned integer categories with negative padding required')
        self.categories = [list(map(int, row[row >= 0])) for row in cats]
        if any(not c for c in self.categories):
            raise ValueError('native categories required for every catalog item')
        self.minimum = np.asarray(transform['watch_min'], dtype=np.float64)
        maximum = np.asarray(transform['watch_max'], dtype=np.float64)
        self.response_minimum = float(np.asarray(transform['response_matrix_minimum']).item())
        if (not np.array_equal(transform['raw_items'], raw_items) or
                self.minimum.shape != (len(raw_items),) or maximum.shape != self.minimum.shape or
                not np.isfinite(self.minimum).all() or not np.isfinite(maximum).all() or
                not np.isfinite(self.response_minimum) or np.any(maximum < self.minimum)):
            raise ValueError('frozen TRAIN-only per-item transform does not align with catalog')
        self.range = maximum - self.minimum

    def _leave(self, action, t):
        window = self.history[t-9:t] if t else []
        counts = Counter(c for prior in window for c in self.categories[prior])
        return bool(t and any(counts[c] > 1 for c in self.categories[action])), window

    def feedback(self, raw_reward, encoded_action):
        i = int(encoded_action) - 2
        value = (float(raw_reward) - self.minimum[i]) / self.range[i] if self.range[i] != 0 else 0.
        return float(value - self.response_minimum)  # intentionally no clipping


def build_adapter(bundle_manifest, data_root):
    b = bundle_manifest; e = b['evaluation']
    if e.get('phase') != 'development' or e.get('horizon') not in (30, 40):
        raise ValueError('only explicitly bound native H30/H40 DEV evaluation is implemented')
    world = _asset(b, data_root, 'world'); cases = _asset(b, data_root, e.get('cases_asset', 'dev_cases'))
    args = [world[k] for k in ('rewards', 'raw_users', 'raw_items')] + [cases]
    if b['dataset'] == 'MovieLens' and e['adapter'] == 'movielens_native_v1':
        if e.get('leave_window') != 7 or e.get('leave_threshold') != 75:
            raise ValueError('MovieLens native window7/threshold75 required')
        return MovieLensDEV(*args, _asset(b, data_root, 'distance')['distance'], horizon=e['horizon'])
    if b['dataset'] == 'KuaiRec' and e['adapter'] == 'kuairec_native_v1':
        if e.get('leave_window') != 9 or e.get('leave_threshold') != 1:
            raise ValueError('KuaiRec native window9/categorycount threshold1 required')
        return KuaiRecDEV(*args, _asset(b, data_root, 'categories')['categories'], _asset(b, data_root, 'response_transform'), horizon=e['horizon'])
    if b['dataset'] == 'Coat' and e['adapter'] == 'coat_native_v1':
        if e.get('leave_window') != 7 or e.get('leave_threshold') != 6:
            raise ValueError('Coat native window7/threshold6 required')
        return CoatDEV(*args, _asset(b, data_root, 'distance')['distance'], horizon=e['horizon'])
    raise ValueError('unsupported environment or adapter; no fallback stopping or label rule')
