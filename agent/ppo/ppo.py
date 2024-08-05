"""PPO Agent for the RoomEnv2 environment.

This should be inherited. This itself can not be used.
"""

import os

import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
from humemai.utils import is_running_notebook, write_yaml
from room_env.envs.room2 import RoomEnv2

from ..handcrafted import HandcraftedAgent
from .nn import LSTM
from .utils import (plot_results, save_final_results,
                    save_states_actions_probs_values, save_validation)


class PPOAgent(HandcraftedAgent):
    """PPO Agent interacting with environment.

    Based on https://github.com/MrSyee/pg-is-all-you-need
    """

    def __init__(
        self,
        env_str: str = "room_env:RoomEnv-v2",
        num_episodes: int = 10,
        num_rollouts: int = 2,
        epoch_per_rollout: int = 64,
        batch_size: int = 128,
        gamma: float = 0.9,
        lam: float = 0.8,
        epsilon: float = 0.2,
        entropy_weight: float = 0.005,
        capacity: dict = {
            "episodic": 12,
            "semantic": 12,
            "short": 1,
        },
        pretrain_semantic: str | bool = False,
        nn_params: dict = {
            "hidden_size": 64,
            "num_layers": 2,
            "embedding_dim": 64,
            "make_categorical_embeddings": False,
            "memory_of_interest": [
                "episodic",
                "semantic",
                "short",
            ],
            "fuse_information": "sum",
            "include_positional_encoding": True,
            "max_timesteps": 100,
            "max_strength": 100,
        },
        run_test: bool = True,
        num_samples_for_results: dict = {"val": 10, "test": 10},
        train_seed: int = 5,
        test_seed: int = 0,
        device: str = "cpu",
        mm_policy: str = "generalize",
        qa_function: str = "episodic_semantic",
        explore_policy: str = "avoid_walls",
        env_config: dict = {
            "question_prob": 1.0,
            "terminates_at": 99,
            "randomize_observations": "objects",
            "room_size": "l",
            "rewards": {"correct": 1, "wrong": 0, "partial": 0},
            "make_everything_static": False,
            "num_total_questions": 1000,
            "question_interval": 5,
            "include_walls_in_observations": True,
        },
        default_root_dir: str = "./training-results/PPO/",
        run_handcrafted_baselines: bool = False,
    ) -> None:
        """Initialization.

        Args:
            env_str: environment string. This has to be "room_env:RoomEnv-v2"
            num_episodes: number of episodes
            num_rollouts: number of rollouts
            epoch_per_rollout: number of epochs per rollout
            batch_size: batch size
            gamma: discount factor
            lam: GAE lambda parameter
            epsilon: PPO clip parameter
            entropy_weight: entropy weight
            capacity: The capacity of each human-like memory systems
            pretrain_semantic: whether to pretrain the semantic memory system.
            nn_params: neural network parameters
            run_test: whether to run test
            num_samples_for_results: The number of samples to validate / test the agent.
            train_seed: seed for training
            test_seed: seed for testing
            device: This is either "cpu" or "cuda".
            mm_policy: memory management policy. Choose one of "generalize", "random",
                "rl", or "neural"
            qa_function: question answering policy Choose one of "episodic_semantic",
                "random", or "neural". qa_function shouldn't be trained with RL. There is
                no sequence of states / actions to learn from.
            explore_policy: The room exploration policy. Choose one of "random",
                "avoid_walls", "rl", or "neural"
            env_config: The configuration of the environment.
                question_prob: The probability of a question being asked at every
                    observation.
                terminates_at: The maximum number of steps to take in an episode.
                seed: seed for env
                room_size: The room configuration to use. Choose one of "dev", "xxs",
                    "xs", "s", "m", or "l".
            default_root_dir: default root directory to save results
            run_handcrafted_baselines: whether to run handcrafted baselines

        """
        self.train_seed = train_seed
        self.test_seed = test_seed
        env_config["seed"] = self.train_seed

        super().__init__(
            env_str=env_str,
            env_config=env_config,
            mm_policy=mm_policy,
            qa_function=qa_function,
            explore_policy=explore_policy,
            num_samples_for_results=num_samples_for_results,
            capacity=capacity,
            pretrain_semantic=pretrain_semantic,
            default_root_dir=default_root_dir,
        )
        self.num_steps_in_episode = self.env.unwrapped.terminates_at + 1
        self.total_maximum_episode_rewards = (
            self.env.unwrapped.total_maximum_episode_rewards
        )
        self.device = torch.device(device)
        print(f"Running on {self.device}")

        self.nn_params = nn_params
        self.nn_params["capacity"] = self.capacity
        self.nn_params["device"] = self.device
        self.nn_params["entities"] = self.env.unwrapped.entities
        self.nn_params["relations"] = self.env.unwrapped.relations

        self.val_filenames = []
        self.is_notebook = is_running_notebook()

        self.num_episodes = num_episodes
        self.num_rollouts = num_rollouts
        self.epoch_per_rollout = epoch_per_rollout
        self.batch_size = batch_size
        self.gamma = gamma
        self.lam = lam
        self.epsilon = epsilon
        self.entropy_weight = entropy_weight

        self.run_test = run_test

        assert (self.num_rollouts % self.num_episodes) == 0 or (
            self.num_episodes % self.num_rollouts
        ) == 0

        self.num_steps_per_rollout = int(
            self.num_episodes / self.num_rollouts * self.num_steps_in_episode
        )

        self.actor = LSTM(**self.nn_params, is_actor=True, is_critic=False)
        self.critic = LSTM(**self.nn_params, is_actor=False, is_critic=True)

        # optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters())
        self.critic_optimizer = optim.Adam(self.critic.parameters())

        # global stats to save
        self.actor_losses, self.critic_losses = [], []  # training loss
        self.states_all = {"train": [], "val": [], "test": []}
        self.scores_all = {"train": [], "val": [], "test": None}
        self.actions_all = {"train": [], "val": [], "test": []}
        self.actor_probs_all = {"train": [], "val": [], "test": []}
        self.critic_values_all = {"train": [], "val": [], "test": []}

        if run_handcrafted_baselines:
            self.run_handcrafted_baselines()

    def run_handcrafted_baselines(self) -> None:
        """Run and save the handcrafted baselines."""

        env = RoomEnv2(**self.env_config)
        observations, info = env.reset()
        env.render("image", save_fig_dir=self.default_root_dir)
        del env

        policies = [
            {
                "mm": mm,
                "qa": qa,
                "explore": explore,
                "pretrain_semantic": pretrain_semantic,
            }
            for mm in ["random", "episodic", "semantic"]
            for qa in ["episodic_semantic"]
            for explore in ["random", "avoid_walls"]
            for pretrain_semantic in [False, "exclude_walls"]
        ]

        results = {}
        for policy in policies:
            results[str(policy)] = []
            for test_seed in [0, 1, 2, 3, 4]:
                agent_handcrafted = HandcraftedAgent(
                    env_str="room_env:RoomEnv-v2",
                    env_config={**self.env_config, "seed": test_seed},
                    mm_policy=policy["mm"],
                    qa_function=policy["qa"],
                    explore_policy=policy["explore"],
                    num_samples_for_results=self.num_samples_for_results,
                    capacity=self.capacity,
                    pretrain_semantic=policy["pretrain_semantic"],
                    default_root_dir=self.default_root_dir,
                )
                agent_handcrafted.test()
                results[str(policy)].append(
                    agent_handcrafted.scores["test_score"]["mean"]
                )
                agent_handcrafted.remove_results_from_disk()
            results[str(policy)] = {
                "mean": np.mean(results[str(policy)]).item(),
                "std": np.std(results[str(policy)]).item(),
            }
        write_yaml(results, os.path.join(self.default_root_dir, "handcrafted.yaml"))

    def create_empty_rollout_buffer(self) -> tuple[list, list, list, list, list, list]:
        """Create empty buffer for training.

        Make sure to call this before each rollout.

        Returns:
            states_buffer: The states.
            actions_buffer: The actions.
            rewards_buffer: The rewards.
            values_buffer: The values.
            masks_buffer: The masks.
            log_probs_buffer: The log probabilities.

        """
        # memory for training
        states_buffer: list[dict] = []  # this has to be a list of dictionaries
        actions_buffer: list[torch.Tensor] = []
        rewards_buffer: list[torch.Tensor] = []
        values_buffer: list[torch.Tensor] = []
        masks_buffer: list[torch.Tensor] = []
        log_probs_buffer: list[torch.Tensor] = []

        return (
            states_buffer,
            actions_buffer,
            rewards_buffer,
            values_buffer,
            masks_buffer,
            log_probs_buffer,
        )

    def step(self) -> None:
        """Interact with the actual gymnasium environment by taking a step."""

    def train(self) -> None:
        """Code for training"""

    def validate(self) -> None:
        """Validate the agent."""
        self.actor.eval()
        self.critic.eval()

        scores = self.validate_test_middle("val")

        save_validation(
            scores=scores,
            scores_all_val=self.scores_all["val"],
            default_root_dir=self.default_root_dir,
            num_validation=self.num_validation,
            val_filenames=self.val_filenames,
            actor=self.actor,
            critic=self.critic,
        )

        start = self.num_validation * self.num_steps_in_episode
        end = (self.num_validation + 1) * self.num_steps_in_episode

        save_states_actions_probs_values(
            self.states_all["val"][start:end],
            self.actions_all["val"][start:end],
            self.actor_probs_all["val"][start:end],
            self.critic_values_all["val"][start:end],
            self.default_root_dir,
            "val",
            self.num_validation,
        )

        self.env.close()
        self.num_validation += self.validation_interval
        self.actor.train()
        self.critic.train()

    def test(self, checkpoint: str = None) -> None:
        self.env_config["seed"] = self.test_seed
        self.env = gym.make(self.env_str, **self.env_config)
        self.actor.eval()
        self.critic.eval()

        assert len(self.val_filenames) == 1
        self.actor.load_state_dict(
            torch.load(os.path.join(self.val_filenames[0], "actor.pt"))
        )
        self.critic.load_state_dict(
            torch.load(os.path.join(self.val_filenames[0], "critic.pt"))
        )
        if checkpoint is not None:
            self.actor.load_state_dict(os.path.join(torch.load(checkpoint), "actor.pt"))
            self.critic.load_state_dict(
                os.path.join(torch.load(checkpoint), "critic.pt")
            )

        scores = self.validate_test_middle("test")

        self.scores_all["test"] = scores

        save_states_actions_probs_values(
            self.states_all["test"],
            self.actions_all["test"],
            self.actor_probs_all["test"],
            self.critic_values_all["test"],
            self.default_root_dir,
            "test",
        )

        save_final_results(
            self.scores_all,
            self.actor_losses,
            self.critic_losses,
            self.default_root_dir,
            self,
        )

        self.plot_results("all", True)
        self.env.close()
        self.actor.train()
        self.critic.train()

    def plot_results(self, to_plot: str = "all", save_fig: bool = False) -> None:
        """Plot things for ppo training.

        Args:
            to_plot: what to plot:
                all: everything
                actor_loss: actor loss
                critic_loss: critic loss
                scores: train, val, and test scores
                actor_probs_train: actor probabilities for training
                actor_probs_val: actor probabilities for validation
                actor_probs_test: actor probabilities for test
                critic_values_train: critic values for training
                critic_values_val: critic values for validation
                critic_values_test: critic values for test

        """
        plot_results(
            self.scores_all,
            self.actor_losses,
            self.critic_losses,
            self.actor_probs_all,
            self.critic_values_all,
            self.num_validation,
            self.action_space.n.item(),
            self.num_episodes,
            self.total_maximum_episode_rewards,
            self.default_root_dir,
            to_plot,
            save_fig,
        )
