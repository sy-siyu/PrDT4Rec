"""Trajectory records, storage and padded batches."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import json
import numpy as np
import torch
SPLIT_TO_CODE={"train":0,"validation":1,"test":2}
CODE_TO_SPLIT={v:k for k,v in SPLIT_TO_CODE.items()}
PAD_ITEM_ID=0
UNK_ITEM_ID=1

@dataclass(frozen=True)
class Trajectory:
    index: int
    user_id: int
    actions: np.ndarray
    feedback: np.ndarray
    timestamps: np.ndarray
    source_ids: np.ndarray
    split: str
    positive_count: float
    source_session_id: int = -1
    # Optional unrewarded reset observation. It is neither a chosen action nor
    # part of feedback/RTG; zero means that the source has no reset context.
    initial_context_action: int = 0

    @property
    def length(self) -> int:
        return int(self.actions.shape[0])


class TrajectoryCorpus:
    def __init__(
        self,
        trajectories: Sequence[Trajectory],
        n_users: int,
        n_items: int,
        max_length: int,
        metadata: dict | None = None,
    ) -> None:
        self.trajectories = list(trajectories)
        self.n_users = int(n_users)
        self.n_items = int(n_items)
        self.max_length = int(max_length)
        self.metadata = metadata or {}
        self._by_index = {t.index: t for t in self.trajectories}
        if len(self._by_index) != len(self.trajectories):
            raise ValueError("trajectory indices must be unique")
        if any(
            isinstance(trajectory.initial_context_action, (bool, np.bool_))
            or not isinstance(trajectory.initial_context_action, (int, np.integer))
            or (trajectory.initial_context_action != 0
                and not 2 <= trajectory.initial_context_action < self.n_items)
            for trajectory in self.trajectories
        ):
            raise ValueError("initial context actions must be zero or valid real item IDs")

    def split(self, name: str) -> list[Trajectory]:
        return [trajectory for trajectory in self.trajectories if trajectory.split == name]

    def get(self, index: int) -> Trajectory:
        return self._by_index[int(index)]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        count = len(self.trajectories)
        actions = np.zeros((count, self.max_length), dtype=np.int64)
        # Preserve canonical float64 simulator outcomes when present. Existing
        # rating/simulator corpora remain float32, and model collation below
        # still makes its deliberate float32 compute copy.
        feedback_dtype = np.float64 if any(
            np.asarray(trajectory.feedback).dtype == np.dtype("float64")
            for trajectory in self.trajectories
        ) else np.float32
        feedback = np.zeros((count, self.max_length), dtype=feedback_dtype)
        timestamps = np.zeros((count, self.max_length), dtype=np.int64)
        source_ids = np.zeros((count, self.max_length), dtype=np.int64)
        mask = np.zeros((count, self.max_length), dtype=np.bool_)
        indices = np.empty(count, dtype=np.int64)
        users = np.empty(count, dtype=np.int64)
        splits = np.empty(count, dtype=np.int8)
        proxy = np.empty(count, dtype=np.float32)
        sessions = np.empty(count, dtype=np.int64)
        initial_context_actions = np.empty(count, dtype=np.int64)
        for row, trajectory in enumerate(self.trajectories):
            length = trajectory.length
            actions[row, :length] = trajectory.actions
            feedback[row, :length] = trajectory.feedback
            timestamps[row, :length] = trajectory.timestamps
            source_ids[row, :length] = trajectory.source_ids
            mask[row, :length] = True
            indices[row] = trajectory.index
            users[row] = trajectory.user_id
            splits[row] = SPLIT_TO_CODE[trajectory.split]
            proxy[row] = trajectory.positive_count
            sessions[row] = trajectory.source_session_id
            initial_context_actions[row] = trajectory.initial_context_action
        np.savez_compressed(
            path,
            actions=actions,
            feedback=feedback,
            timestamps=timestamps,
            source_ids=source_ids,
            mask=mask,
            indices=indices,
            users=users,
            splits=splits,
            positive_count=proxy,
            source_session_ids=sessions,
            initial_context_actions=initial_context_actions,
            n_users=np.asarray(self.n_users, dtype=np.int64),
            n_items=np.asarray(self.n_items, dtype=np.int64),
            max_length=np.asarray(self.max_length, dtype=np.int64),
            metadata=np.asarray(json.dumps(self.metadata, sort_keys=True)),
        )

    @classmethod
    def load(cls, path: str | Path) -> "TrajectoryCorpus":
        with np.load(path, allow_pickle=False) as archive:
            # NpzFile access decompresses an entire member on every lookup.
            # Read each member exactly once before constructing row objects.
            data = {name: archive[name] for name in archive.files}
            trajectories: list[Trajectory] = []
            for row in range(data["actions"].shape[0]):
                length = int(data["mask"][row].sum())
                trajectories.append(
                    Trajectory(
                        index=int(data["indices"][row]),
                        user_id=int(data["users"][row]),
                        actions=data["actions"][row, :length].copy(),
                        feedback=data["feedback"][row, :length].copy(),
                        timestamps=data["timestamps"][row, :length].copy(),
                        source_ids=data["source_ids"][row, :length].copy(),
                        split=CODE_TO_SPLIT[int(data["splits"][row])],
                        positive_count=float(data["positive_count"][row]),
                        source_session_id=int(data["source_session_ids"][row]) if "source_session_ids" in data else -1,
                        initial_context_action=(int(data["initial_context_actions"][row])
                                                if "initial_context_actions" in data else 0),
                    )
                )
            metadata = json.loads(str(data["metadata"].item()))
            return cls(
                trajectories,
                n_users=int(data["n_users"].item()),
                n_items=int(data["n_items"].item()),
                max_length=int(data["max_length"].item()),
                metadata=metadata,
            )


def collate_trajectories(batch: Sequence[Trajectory]) -> dict[str, torch.Tensor]:
    max_length = max(trajectory.length for trajectory in batch)
    batch_size = len(batch)
    actions = torch.zeros((batch_size, max_length), dtype=torch.long)
    feedback = torch.zeros((batch_size, max_length), dtype=torch.float32)
    mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
    users = torch.empty(batch_size, dtype=torch.long)
    indices = torch.empty(batch_size, dtype=torch.long)
    initial_context_actions = torch.empty(batch_size, dtype=torch.long)
    for row, trajectory in enumerate(batch):
        length = trajectory.length
        actions[row, :length] = torch.from_numpy(trajectory.actions)
        feedback[row, :length] = torch.from_numpy(trajectory.feedback)
        mask[row, :length] = True
        users[row] = trajectory.user_id
        indices[row] = trajectory.index
        initial_context_actions[row] = trajectory.initial_context_action
    return {"actions": actions, "feedback": feedback, "mask": mask, "users": users, "indices": indices,
            "initial_context_actions": initial_context_actions}
