from configparser import ConfigParser

import numpy as np

from .base import AgentEnsemble


class RandomAgentEnsemble(AgentEnsemble):
    def act(self, state: np.ndarray, add_noise: bool = True) -> int:
        return np.random.uniform(-1, 1, size=(self.number_of_agents, self.action_size))

    def step(self, *args):
        pass

    def reset(self):
        pass

    def save(self, *args, filename: str = "", **kwargs):
        pass

    def load(self, *args, filename: str = "", **kwargs):
        pass

    @classmethod
    def from_config(
        cls,
        config: ConfigParser,
        number_of_agents: int,
        state_size: int,
        action_size: int,
    ):
        return cls(number_of_agents, state_size, action_size)
