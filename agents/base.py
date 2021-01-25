import configparser
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

States = Actions = Rewards = NextStates = Dones = torch.Tensor
Experiences = Tuple[States, Actions, Rewards, NextStates, Dones]


class AgentEnsemble(ABC):
    def __init__(self, number_of_agents: int, state_size: int, action_size: int):
        self.number_of_agents = number_of_agents
        self.state_size = state_size
        self.action_size = action_size
        self.train = False

    def set_train_mode(self, train_mode: bool):
        self.train = train_mode

    @abstractmethod
    def act(self, state: np.ndarray, add_noise: bool = True) -> np.ndarray:
        pass

    @abstractmethod
    def step(
        self,
        state: np.ndarray,
        actions: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def save(self, *args, filename: str = "", **kwargs):
        pass

    @abstractmethod
    def load(self, *args, filename: str = "", **kwargs):
        pass

    @classmethod
    @abstractmethod
    def from_config(
        cls,
        config: configparser.SectionProxy,
        number_of_agents: int,
        state_size: int,
        action_size: int,
    ):
        pass
