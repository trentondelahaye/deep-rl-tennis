import matplotlib.pyplot as plt
import numpy as np

from agents import AgentEnsemble
from unityagents import UnityEnvironment


class AgentTrainer:
    """Used to train the agent for either a specified number of episodes or
    until the environment is solved. Stores scores and is able to plot them.
    """
    def __init__(self, score_window_size: int, score_threshold: float):
        self.score_window_size = score_window_size
        self.score_threshold = score_threshold
        self.scores = []

    def train_agent(
        self,
        env: UnityEnvironment,
        agent: AgentEnsemble,
        verbose: bool = True,
        exit_when_solved: bool = True,
        number_episodes: int = 1000,
        **kwargs,
    ) -> None:
        """Train the agent for either a specified number of episodes or
        until the environment is solved
        """
        agent.set_train_mode(True)
        brain_name = env.brain_names[0]
        number_of_agents = agent.number_of_agents

        # training loop over the number of episodes
        for _ in range(number_episodes):
            episode_number = len(self.scores) + 1

            # reset the environment and agent
            env_info = env.reset(train_mode=True)[brain_name]
            agent.reset()

            # get initial state and reset scores
            states = env_info.vector_observations
            scores = np.zeros(number_of_agents)
            while True:
                # select actions
                actions = agent.act(states, True)
                # take actions
                env_info = env.step(actions)[brain_name]
                # observe actions
                rewards = env_info.rewards
                # observe next states
                next_states = env_info.vector_observations
                # see if the episode is completed
                dones = env_info.local_done
                # give the experiences to the agent to store and potentially learn from
                agent.step(states, actions, rewards, next_states, dones)
                # update score
                scores += rewards
                states = next_states
                # if the episode is over, break
                if np.all(dones):
                    break

            self.scores.append(np.max(scores))
            average_score_window = np.mean(self.scores[-self.score_window_size :])

            if verbose:
                print(
                    f"\rEpisode {episode_number}\tAverage Score: {average_score_window:.2f}",
                    end="",
                )
                if episode_number % 100 == 0:
                    print(
                        f"\rEpisode {episode_number}\tAverage Score: {average_score_window:.2f}",
                    )

            if (
                exit_when_solved
                and episode_number >= self.score_window_size
                and average_score_window >= self.score_threshold
            ):
                if verbose:
                    print(
                        f"\rEnvironment solved in {len(self.scores)} episodes!          ",
                    )
                break

    def plot_progress(self, *args, **kwargs):
        # plot the scores
        fig, ax = plt.subplots(figsize=(15, 10))
        plt.plot(np.arange(len(self.scores)), self.scores)
        plt.ylabel("Score")
        plt.xlabel("Episode")
        plt.title("Agent training progress")
        plt.show()
