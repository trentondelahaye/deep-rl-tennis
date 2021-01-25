import random
from collections import deque
from typing import Iterable, NamedTuple

import numpy as np

import torch
from agents.base import Experiences, device


class Experience(NamedTuple):
    state: np.ndarray
    action: int
    reward: np.float32
    next_state: np.ndarray
    done: np.float32


class ReplayBuffer:
    def __init__(self, buffer_size: int, batch_size: int):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add_experience(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        experience = Experience(
            state.astype(np.float32),
            action,
            np.float32(reward),
            next_state.astype(np.float32),
            np.float32(done),
        )
        self.memory.append(experience)

    # noinspection PyTypeChecker
    def sample_experiences(self) -> Experiences:
        experiences = self._sample()
        return tuple(
            (
                torch.from_numpy(
                    np.vstack([getattr(e, field) for e in experiences])
                ).to(device)
            )
            for field in Experience._fields
        )

    def _sample(self) -> Iterable[Experience]:
        return random.sample(self.memory, k=self.batch_size)

    def __len__(self):
        return len(self.memory)
