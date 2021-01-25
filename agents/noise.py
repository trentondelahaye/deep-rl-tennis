from typing import Tuple

import numpy as np


class OrnsteinUhlenbeckNoise:
    def __init__(
        self,
        size: Tuple[int, ...],
        mu: float = 0.,
        theta: float = 0.15,
        sigma: float = 0.2
    ):
        self.mu = mu
        self.size = size
        self.theta = theta
        self.sigma = sigma
        self.state = self.mu * np.ones(size)

    def reset(self):
        self.state = self.mu * np.ones(self.size)

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x)
        dx += self.sigma * np.random.randn(*self.size)
        self.state = x + dx
        return self.state
