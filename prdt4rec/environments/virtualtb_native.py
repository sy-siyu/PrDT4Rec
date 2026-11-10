"""Profile-conditioned access to the external VirtualTaobao simulator at H30."""

from contextlib import contextmanager
from copy import deepcopy
import hashlib
from importlib import import_module
from pathlib import Path
import sys

import numpy as np
import torch




def reset_seed(domain, digest):
    key = hashlib.sha256(domain.encode("ascii") + b"\0" + bytes.fromhex(digest)).digest()
    return int.from_bytes(key[:8], "big") % (2**31 - 1)


class PrivateTorchRNG:
    def __init__(self, seed):
        self.generator = torch.Generator(device="cpu").manual_seed(seed)

    @contextmanager
    def active(self):
        outer = torch.random.get_rng_state()
        torch.random.set_rng_state(self.generator.get_state())
        try:
            yield
        finally:
            self.generator.set_state(torch.random.get_rng_state())
            torch.random.set_rng_state(outer)

    def call(self, function, *args):
        with self.active():
            return function(*args)


class VirtualTBH30:
    def __init__(self, native):
        self.native = native
        self._finished = True
        self._rng = PrivateTorchRNG(0)

    def _observation(self):
        dynamic = np.asarray(self.native.rend_action, dtype=np.float32).reshape(2)
        return np.concatenate((self._profile, dynamic, np.asarray([self.native.total_c], np.float32)))

    def reset(self, profile, seed):
        values = np.asarray(profile, dtype=np.float32)
        if values.shape != (88,) or not np.isfinite(values).all():
            raise ValueError("VirtualTB reset requires a finite 88D profile")
        self._profile = values.copy()
        self._rng.generator.manual_seed(seed)

        def reset_profile():
            self.native.total_a = 0
            self.native.total_c = 0
            self.native.cur_user = self._profile.copy()
            self.native.lst_action = torch.FloatTensor([0, 0])
            self.native.rend_action = deepcopy(self.native.lst_action)
            # A fresh native leave-model draw is private to the simulator.
            self.native._VirtualTB__leave = self.native.user_leave_model.predict(
                torch.FloatTensor(self._profile).unsqueeze(0)
            )

        self._rng.call(reset_profile)
        self._steps = 0
        self._finished = False
        return self._observation()

    def step(self, action):
        if self._finished:
            raise RuntimeError("step called outside an active VirtualTB episode")
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (27,) or not np.isfinite(action).all() or np.any(np.abs(action) > 1.0):
            raise ValueError("VirtualTB action must be a finite 27D vector in [-1,1]")
        source_observation, reward, done, _ = self._rng.call(self.native.step, action.copy())
        self._steps += 1
        terminated = bool(done)
        truncated = self._steps == 30 and not terminated
        self._finished = terminated or truncated
        if np.shape(source_observation) != (91,):
            raise RuntimeError("native VirtualTB observation geometry changed")
        # The legacy simulator replaces cur_user on done. Preserve the current
        # profile in the terminal observation while retaining the native reward.
        return self._observation(), float(reward), terminated, truncated


def load_environment(source_root):
    if source_root is None:
        raise ValueError("VirtualTB evaluation requires --simulator-root")
    root = Path(source_root).resolve()
    expected = root / "virtualTB" / "envs" / "virtualTB.py"
    for path in (expected, *(root / "virtualTB" / "data" / f"{name}_model.pt"
                             for name in ("action", "generator", "leave"))):
        if not path.is_file():
            raise FileNotFoundError(f"missing external VirtualTaobao file: {path.relative_to(root)}")
    sys.path.insert(0, str(root))
    try:
        try:
            module = import_module("virtualTB.envs.virtualTB")
        except ModuleNotFoundError as error:
            raise RuntimeError("install the legacy Gym dependency for the external VirtualTaobao simulator") from error
        if Path(module.__file__).resolve() != expected:
            raise RuntimeError("VirtualTaobao imported from a different source directory")
        return VirtualTBH30(PrivateTorchRNG(0).call(module.VirtualTB))
    finally:
        sys.path.remove(str(root))
