import inspect
from configparser import ConfigParser
from typing import Any, Callable, Dict, List

import agents
from agents import AgentEnsemble
from env_interface.observe import watch_episode
from env_interface.train import AgentTrainer
from unityagents import UnityEnvironment


def load_env(env_path: str, no_graphics: bool = False) -> UnityEnvironment:
    return UnityEnvironment(file_name=env_path, no_graphics=no_graphics)


def load_agent(env: UnityEnvironment, agent_cfg: str) -> agents.AgentEnsemble:
    config = ConfigParser()
    config.read("./agents/configs.cfg")

    try:
        section = config[agent_cfg]
    except KeyError:
        raise Exception("Config section not found")

    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]

    number_of_agents = len(env_info.agents)
    action_size = brain.vector_action_space_size
    state_size = (
        brain.num_stacked_vector_observations * brain.vector_observation_space_size
    )

    agent_name = section.get("agent")

    try:
        agent: AgentEnsemble = getattr(agents, agent_name)
        return agent.from_config(section, number_of_agents, state_size, action_size)
    except AttributeError:
        raise Exception(f"Unrecognised agent {agent_name}")


def build_commands(trainer: AgentTrainer, agent: AgentEnsemble) -> Dict[str, Callable]:
    return {
        "exit": lambda *args: True,
        "watch": watch_episode,
        "train": trainer.train_agent,
        "plot": trainer.plot_progress,
        "save-agent": agent.save,
        "load-agent": agent.load,
    }


class KwargError(BaseException):
    pass


def build_kwargs(raw_kwargs: List[str], sig: inspect.Signature) -> Dict[str, Any]:
    kwargs = {}
    for kwarg in raw_kwargs:
        try:
            key, value = kwarg.split("=")
        except ValueError:
            raise KwargError(f"Please format kwarg {kwarg} like key=value")

        try:
            param = sig.parameters[key]
        except KeyError:
            raise KwargError(f"Unrecognised kwarg {key}")

        if param.default is param.empty:
            raise KwargError(f"Setting positional argument {key}")

        try:
            param_type = type(param.default)
            if param_type == bool:
                if value.lower() in {"true", "t"}:
                    value = True
                elif value.lower() in {"false", "f"}:
                    value = False
                else:
                    raise KwargError(
                        f"Unable to convert type for kwarg {kwarg}"
                        f" inferred type: {type(param.default).__name__}"
                    )
            else:
                value = type(param.default)(value)
        except ValueError:
            raise KwargError(
                f"Unable to convert type for kwarg {kwarg}"
                f" inferred type: {type(param.default).__name__}"
            )
        kwargs[key] = value

    return kwargs
