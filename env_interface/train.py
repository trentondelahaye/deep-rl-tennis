import matplotlib.pyplot as plt
import numpy as np

from agents import AgentEnsemble
from unityagents import UnityEnvironment


class AgentTrainer:
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
        agent.set_train_mode(True)
        brain_name = env.brain_names[0]
        number_of_agents = agent.number_of_agents
        max_avg = -100
        for _ in range(number_episodes):
            episode_number = len(self.scores) + 1
            env_info = env.reset(train_mode=True)[brain_name]
            agent.reset()
            states = env_info.vector_observations
            scores = np.zeros(number_of_agents)
            while True:
                actions = agent.act(states, True)
                env_info = env.step(actions)[brain_name]
                rewards = env_info.rewards
                next_states = env_info.vector_observations
                dones = env_info.local_done
                agent.step(states, actions, rewards, next_states, dones)
                scores += rewards
                states = next_states
                if np.all(dones):
                    break

            self.scores.append(np.max(scores))
            average_score_window = np.mean(self.scores[-self.score_window_size :])

            if average_score_window > max_avg:
                max_avg = average_score_window
                agent.save(filename="ddpg_optimal.pth")

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
        fig, ax = plt.subplots(figsize=(15, 10))
        plt.plot(np.arange(len(self.scores)), self.scores)
        plt.ylabel("Score")
        plt.xlabel("Episode")
        plt.title("Agent training progress")
        plt.show()
