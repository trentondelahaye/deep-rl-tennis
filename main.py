import inspect
import logging
import random

import click
import numpy as np

import torch
from env_interface.load import (
    KwargError,
    build_commands,
    build_kwargs,
    load_agent,
    load_env,
)
from env_interface.train import AgentTrainer

log = logging.getLogger()


@click.command()
@click.option("--unity-env", default="./Tennis.app", help="Path to UnityEnvironment")
@click.option(
    "--agent-cfg", default="Random", help="Section of config used to load agent"
)
@click.option(
    "--no-graphics/--graphics", default=False, help="Load environment without graphics"
)
def main(unity_env, agent_cfg, no_graphics):
    env = load_env(unity_env, no_graphics)
    agent = load_agent(env, agent_cfg)

    #  TODO: get this information for environments other than Tennis,
    #   currently not done as to not clutter the click options
    trainer = AgentTrainer(score_window_size=100, score_threshold=0.5)

    command_to_func = build_commands(trainer, agent)

    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)

    to_exit = False

    while not to_exit:
        inputs = input("\nInput command: ").lower().split(" ")
        command = inputs[0]
        if command not in command_to_func:
            log.info(f"Unrecognised command, select from {set(command_to_func.keys())}")
            continue

        func = command_to_func[command]
        sig = inspect.signature(func)

        if any(arg == "--help" for arg in inputs):
            print("Valid kwargs")
            for param in sig.parameters.values():
                if param.default is not param.empty:
                    print(
                        f"{param.name}: {type(param.default).__name__} = {param.default}"
                    )
            continue

        try:
            kwargs = build_kwargs(inputs[1:], sig)
        except KwargError as e:
            log.warning(e)
            continue

        to_exit = func(env, agent, **kwargs)

    env.close()


if __name__ == "__main__":
    main()
