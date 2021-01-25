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
    """Main loop of the program, used to interact with the agent through
    a series of commands
    """
    # load the environment
    env = load_env(unity_env, no_graphics)
    # load the agent
    agent = load_agent(env, agent_cfg)

    #  TODO: get this information for environments other than Tennis,
    #   currently not done as to not clutter the click options
    trainer = AgentTrainer(score_window_size=100, score_threshold=0.5)

    # builds commands, pointing strings to methods on the trainer object
    command_to_func = build_commands(trainer, agent)

    # seed the algorithm for determinism
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)

    to_exit = False

    # loop until a function returns an exit bool
    while not to_exit:
        # parse the users command
        inputs = input("\nInput command: ").lower().split(" ")
        command = inputs[0]
        if command not in command_to_func:
            log.info(f"Unrecognised command, select from {set(command_to_func.keys())}")
            continue

        # obtain the relevant function
        func = command_to_func[command]
        sig = inspect.signature(func)

        # display valid kwargs and types if the user requires help
        if any(arg == "--help" for arg in inputs):
            print("Valid kwargs")
            for param in sig.parameters.values():
                if param.default is not param.empty:
                    print(
                        f"{param.name}: {type(param.default).__name__} = {param.default}"
                    )
            continue

        # try to cast to the inferred types, else fail and show error message
        try:
            kwargs = build_kwargs(inputs[1:], sig)
        except KwargError as e:
            log.warning(e)
            continue

        to_exit = func(env, agent, **kwargs)

    env.close()


if __name__ == "__main__":
    main()
