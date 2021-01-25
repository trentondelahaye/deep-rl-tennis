import numpy as np

from agents.base import AgentEnsemble
from unityagents import UnityEnvironment


def watch_episode(env: UnityEnvironment, agent: AgentEnsemble, **kwargs):
    agent.set_train_mode(False)
    brain_name = env.brain_names[0]
    env_info = env.reset(train_mode=False)[brain_name]
    states = env_info.vector_observations
    dones = env_info.local_done
    while not np.any(dones):
        actions = agent.act(states, False)
        env_info = env.step(actions)[brain_name]
        next_states = env_info.vector_observations
        dones = env_info.local_done
        states = next_states
