import logging
import os
from configparser import ConfigParser
from typing import Any, Dict, Iterable

import numpy as np

import torch
import torch.nn.functional as F
from agents.networks import Actor, Critic
from torch.optim import Adam

from .base import AgentEnsemble, Experiences
from .noise import OrnsteinUhlenbeckNoise
from .replay_buffers import ReplayBuffer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

log = logging.getLogger()


class DDPGAgentEnsemble(AgentEnsemble):
    """Implements the DDPG algorithm to be used with multiple agents sharing the same
    actor and critic.
    """

    train_with_noise = True
    torch_states = (
        "actor",
        "actor_target",
        "actor_optimizer",
        "critic",
        "critic_target",
        "critic_optimizer",
    )

    def __init__(
        self,
        *args,
        gamma: float,
        buffer_size: int,
        batch_size: int,
        update_every: int,
        actor_tau: float,
        actor_lr: float,
        actor_fc_layers: Iterable[int],
        critic_tau: float,
        critic_lr: float,
        critic_fc_layers: Iterable[int],
    ):
        super().__init__(*args)

        # Q learning params
        self.gamma = gamma
        self.t_step = 0

        # replay buffer params
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)
        self.update_every = update_every

        # Exploration noise
        self.noise = OrnsteinUhlenbeckNoise((self.number_of_agents, self.action_size))

        # Initialise local and target actor networks
        self.actor_tau = actor_tau
        self.actor_lr = actor_lr
        self.actor_fc_layers = actor_fc_layers
        self.actor = Actor(self.state_size, self.action_size, self.actor_fc_layers).to(
            device
        )
        self.actor_target = Actor(
            self.state_size, self.action_size, self.actor_fc_layers
        ).to(device)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.actor_lr)

        # Initialise local and target critic networks
        self.critic_tau = critic_tau
        self.critic_lr = critic_lr
        self.critic_fc_layers = critic_fc_layers
        self.critic = Critic(
            self.state_size, self.action_size, self.critic_fc_layers
        ).to(device)
        self.critic_target = Critic(
            self.state_size, self.action_size, self.critic_fc_layers
        ).to(device)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.critic_lr)

    def reset(self):
        self.noise.reset()

    def save(self, *args, filename: str = "", **kwargs):
        """Function for saving the model weights of all neural networks
        in the agent
        """
        if not len(filename):
            log.warning("Please provide a filename")
            return

        dir_name = os.path.dirname(__file__)
        path = os.path.join(dir_name, f"checkpoints/{filename}")
        state = {
            torch_state: getattr(self, torch_state).state_dict()
            for torch_state in self.torch_states
        }
        torch.save(state, path)

    def load(self, *args, filename: str = "", **kwargs):
        """Function for loading the saved model weights of all neural networks
        in the agent
        """
        if not len(filename):
            log.warning("Please provide a filename")
            return

        dir_name = os.path.dirname(__file__)
        path = os.path.join(dir_name, f"checkpoints/{filename}")

        try:
            state = torch.load(path)
        except FileNotFoundError as e:
            log.warning(f"Unable to load agent state: {e}")
            return

        try:
            for torch_state in self.torch_states:
                getattr(self, torch_state).load_state_dict(state[torch_state])
        except RuntimeError as e:
            log.warning(f"Unable to load agent state: {e}")

    def act(self, state: np.ndarray, add_noise: bool = True) -> np.ndarray:
        # take the state turn into a torch tensor
        state = torch.from_numpy(state).float().to(device)
        self.actor.eval()
        with torch.no_grad():
            actions = self.actor(state).cpu().data.numpy()
        self.actor.train()

        # add noise to encourage exploration
        if add_noise and self.train_with_noise:
            actions += self.noise.sample()
        return np.clip(actions, -1, 1)

    def step(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
    ):
        # for each agent add the experiences to the global buffer
        for i in range(self.number_of_agents):
            self.memory.add_experience(
                state[i, :], action[i, :], reward[i], next_state[i, :], done[i]
            )

        # train the model
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0 and len(self.memory) > self.memory.batch_size:
            experiences = self.memory.sample_experiences()
            self._learn(experiences)

    def _learn(self, experiences: Experiences) -> None:
        states, actions, rewards, next_states, dones = experiences

        # work out the expected q of the next states then discount
        # and add reward for current states
        actions_next = self.actor_target(next_states)
        q_targets_next = self.critic_target(next_states, actions_next)
        q_targets = rewards + (self.gamma * q_targets_next * (1 - dones))
        q_expected = self.critic(states, actions)

        # work out the loss of expected q and optimise the critic
        critic_loss = F.mse_loss(q_expected, q_targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # predict the best actions with the actor
        actions_pred = self.actor(states)
        actor_loss = -self.critic(states, actions_pred).mean()

        # optimise the actions to increase the score (minimise negative loss)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # perform a soft update between local and target networks
        self._soft_update()

    def _soft_update(self):
        # soft update by mixing local and target params on the target actor network
        for target_param, param in zip(
            self.actor_target.parameters(), self.actor.parameters()
        ):
            target_param.data.copy_(
                self.actor_tau * param.data + (1.0 - self.actor_tau) * target_param.data
            )

        # soft update by mixing local and target params on the target critic network
        for target_param, param in zip(
            self.critic_target.parameters(), self.critic.parameters()
        ):
            target_param.data.copy_(
                self.critic_tau * param.data
                + (1.0 - self.critic_tau) * target_param.data
            )

    # noinspection PyArgumentList
    @classmethod
    def _get_config_params(cls, config: ConfigParser) -> Dict[str, Any]:
        return dict(
            gamma=config.getfloat("gamma"),
            buffer_size=config.getint("buffer_size"),
            batch_size=config.getint("batch_size"),
            update_every=config.getint("update_every"),
            actor_tau=config.getfloat("actor_tau"),
            actor_lr=config.getfloat("actor_lr"),
            actor_fc_layers=tuple(
                int(layer) for layer in config.get("actor_fc_layers").split(",")
            ),
            critic_tau=config.getfloat("critic_tau"),
            critic_lr=config.getfloat("critic_lr"),
            critic_fc_layers=tuple(
                int(layer) for layer in config.get("critic_fc_layers").split(",")
            ),
        )

    @classmethod
    def from_config(
        cls,
        config: ConfigParser,
        number_of_agents: int,
        state_size: int,
        action_size: int,
    ):
        config_params = cls._get_config_params(config)
        return cls(number_of_agents, state_size, action_size, **config_params)
