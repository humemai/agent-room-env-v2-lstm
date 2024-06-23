"""DQN explore Agent for the RoomEnv2 environment."""

import os
from copy import deepcopy

import gymnasium as gym
import numpy as np
import torch
from humemai.policy import (
    answer_question,
    encode_observation,
    explore,
    manage_memory,
    argmax,
)
from humemai.utils import read_pickle, read_yaml, write_yaml

from .dqn import DQNAgent
from .utils import select_action, target_hard_update, update_epsilon, update_model


class DQNExploreAgent(DQNAgent):
    """DQN explore Agent interacting with environment.

    Based on https://github.com/Curt-Park/rainbow-is-all-you-need/
    """

    def __init__(
        self,
        env_str: str = "room_env:RoomEnv-v2",
        num_iterations: int = 10000,
        replay_buffer_size: int = 10000,
        warm_start: int = 1000,
        batch_size: int = 32,
        target_update_interval: int = 10,
        epsilon_decay_until: float = 10000,
        max_epsilon: float = 1.0,
        min_epsilon: float = 0.1,
        gamma: float = 0.9,
        capacity: dict = {
            "episodic": 16,
            "episodic_agent": 0,
            "semantic": 16,
            "semantic_map": 0,
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
            ],
            "fuse_information": "sum",
            "include_positional_encoding": True,
            "max_timesteps": 100,
            "max_strength": 100,
        },
        run_test: bool = True,
        num_samples_for_results: int = 10,
        plotting_interval: int = 10,
        train_seed: int = 5,
        test_seed: int = 0,
        device: str = "cpu",
        mm_policy: str = "neural",
        mm_agent_path: (
            str | None
        ) = "trained-agents/mm/2023-12-28 18:13:03.001952/agent.pkl",
        qa_policy: str = "episodic_semantic",
        env_config: dict = {
            "question_prob": 1.0,
            "terminates_at": 99,
            "randomize_observations": "objects",
            "room_size": "l",
            "rewards": {"correct": 1, "wrong": 0, "partial": 0},
            "make_everything_static": False,
            "num_total_questions": 1000,
            "question_interval": 1,
            "include_walls_in_observations": True,
        },
        ddqn: bool = True,
        dueling_dqn: bool = True,
        default_root_dir: str = "./training-results/explore",
        run_neural_baseline: bool = False,
    ) -> None:
        """Initialization.

        Args:
            env_str: environment string. This has to be "room_env:RoomEnv-v2"
            num_iterations: number of iterations to train
            replay_buffer_size: size of replay buffer
            warm_start: number of steps to fill the replay buffer, before training
            batch_size: This is the amount of samples sampled from the replay buffer.
            target_update_interval: interval to update target network
            epsilon_decay_until: until which iteration to decay epsilon
            max_epsilon: maximum epsilon
            min_epsilon: minimum epsilon
            gamma: discount factor
            capacity: The capacity of each human-like memory systems
            pretrain_semantic: whether to pretrain the semantic memory system.
            nn_params: parameters for the neural network (DQN)
            run_test: whether to run test
            num_samples_for_results: The number of samples to validate / test the agent.
            plotting_interval: interval to plot results
            train_seed: seed for training
            test_seed: seed for testing
            device: This is either "cpu" or "cuda".
            mm_policy: memory management policy. Choose one of "generalize", "random",
                "rl", or "neural"
            mm_agent_path: memory management agent path
            qa_policy: question answering policy Choose one of "episodic_semantic",
                "random", or "neural". qa_policy shouldn't be trained with RL. There is
                no sequence of states / actions to learn from.
            env_config: The configuration of the environment.
                question_prob: The probability of a question being asked at every
                    observation.
                terminates_at: The maximum number of steps to take in an episode.
                seed: seed for env
                room_size: The room configuration to use. Choose one of "dev", "xxs",
                    "xs", "s", "m", or "l".
            ddqn: whether to use double DQN
            dueling_dqn: whether to use dueling DQN
            default_root_dir: default root directory to save results
            run_neural_baseline: whether to run neural baseline

        """
        all_params = deepcopy(locals())
        del all_params["self"]
        del all_params["__class__"]
        all_params["run_handcrafted_baselines"] = all_params["run_neural_baseline"]
        self.all_params = deepcopy(all_params)
        del all_params["mm_agent_path"]
        del all_params["run_neural_baseline"]

        self.action2str = {0: "north", 1: "east", 2: "south", 3: "west", 4: "stay"}
        self.action_space = gym.spaces.Discrete(len(self.action2str))

        all_params["nn_params"]["n_actions"] = len(self.action2str)
        all_params["explore_policy"] = "rl"
        super().__init__(**all_params)
        write_yaml(self.all_params, os.path.join(self.default_root_dir, "train.yaml"))

        if self.mm_policy == "neural":
            self.mm_agent = read_pickle(mm_agent_path)
            self.mm_agent.dqn.eval()
            self.mm_policy_model = self.mm_agent.dqn
            self.mm_action2str = self.mm_agent.action2str

        if run_neural_baseline:
            with torch.no_grad():
                test_mean, test_std = self.run_neural_baseline()

            handcrafted = read_yaml(
                os.path.join(self.default_root_dir, "handcrafted.yaml")
            )
            handcrafted[
                "{"
                "mm"
                ": "
                "neural"
                ", "
                "qa"
                ": "
                "episodic_semantic"
                ", "
                "explore"
                ": "
                "avoid_walls"
                "}"
            ] = {"mean": test_mean, "std": test_std}
            write_yaml(
                handcrafted, os.path.join(self.default_root_dir, "handcrafted.yaml")
            )

        self.env_config["seed"] = self.train_seed
        self.env = gym.make(self.env_str, **self.env_config)

    def process_room_observations(self, observations_room: list):
        """Process room observations.

        Args:
            observations_room: observations["room"] from the environment

        """
        observations_room = self.manage_agent_and_map_memory(observations_room)
        for obs in observations_room:
            encode_observation(self.memory_systems, obs)

            state = self.memory_systems.return_as_a_dict_list()

            if self.mm_policy == "neural":

                with torch.no_grad():
                    q_values = (
                        self.mm_policy_model(np.array([state]))
                        .detach()
                        .cpu()
                        .tolist()[0]
                    )
                    selected_action = argmax(q_values)

                mm_action = self.mm_action2str[selected_action]

            else:
                mm_action = self.mm_policy

            manage_memory(
                memory_systems=self.memory_systems,
                policy=mm_action,
                split_possessive=False,
            )

    def run_neural_baseline(self) -> None:
        """Run the neural baseline."""
        self.env_config["seed"] = self.test_seed
        self.env = gym.make(self.env_str, **self.env_config)
        scores = []

        for _ in range(self.num_samples_for_results):
            score = 0

            self.init_memory_systems()
            observations, info = self.env.reset()

            while True:

                self.process_room_observations(observations["room"])

                action_explore = explore(self.memory_systems, "avoid_walls")
                actions_qa = [
                    answer_question(
                        self.memory_systems,
                        self.qa_policy,
                        question,
                        split_possessive=False,
                    )
                    for question in observations["questions"]
                ]

                action_pair = (actions_qa, action_explore)
                (
                    observations,
                    reward,
                    done,
                    truncated,
                    info,
                ) = self.env.step(action_pair)
                score += reward
                done = done or truncated

                if done:
                    break

            scores.append(score)

        return np.mean(scores).item(), np.std(scores).item()

    def step(
        self, state: dict, questions: list, greedy: bool
    ) -> tuple[dict, int, float, bool, list]:
        """Step through the environment.

        This is the only place where env.step() is called. Make sure to call this
        function to interact with the environment.

        Args:
            state: state of the memory systems
            questions: questions to answer
            greedy: whether to act greedily

        Returns:
            observations: observations from the environment
            action_explore: exploration action taken
            reward: reward received
            done: whether the episode is done
            q_values: q values (list of floats)

        """
        # select explore action
        action_explore, q_values = select_action(
            state=state,
            greedy=greedy,
            dqn=self.dqn,
            epsilon=self.epsilon,
            action_space=self.action_space,
        )
        actions_qa = [
            answer_question(
                self.memory_systems,
                self.qa_policy,
                question,
                split_possessive=False,
            )
            for question in questions
        ]

        action_pair = (actions_qa, self.action2str[action_explore])
        (
            observations,
            reward,
            done,
            truncated,
            info,
        ) = self.env.step(action_pair)
        done = done or truncated

        return observations, action_explore, reward, done, q_values

    def fill_replay_buffer(self) -> None:
        """Make the replay buffer full in the beginning with the uniformly-sampled
        actions. The filling continues until it reaches the warm start size.

        For explore_policy == "rl"

        """
        new_episode_starts = True
        while len(self.replay_buffer) < self.warm_start:

            if new_episode_starts:
                self.init_memory_systems()
                observations, info = self.env.reset()
                done = False
                self.process_room_observations(observations["room"])
                new_episode_starts = False

            else:
                state = self.get_deepcopied_memory_state()
                observations, action, reward, done, q_values = self.step(
                    state, observations["questions"], greedy=False
                )
                self.process_room_observations(observations["room"])
                next_state = self.get_deepcopied_memory_state()
                self.replay_buffer.store(*[state, action, reward, next_state, done])

            if done:
                new_episode_starts = True

    def train(self) -> None:
        """Train the explore agent."""
        self.fill_replay_buffer()  # fill up the buffer till warm start size
        self.num_validation = 0

        self.epsilons = []
        self.training_loss = []
        self.scores = {"train": [], "val": [], "test": None}

        self.dqn.train()

        new_episode_starts = True
        score = 0

        self.iteration_idx = 0

        while True:
            if new_episode_starts:
                self.init_memory_systems()
                observations, info = self.env.reset()
                done = False
                self.process_room_observations(observations["room"])
                new_episode_starts = False

            else:
                state = self.get_deepcopied_memory_state()
                observations, action, reward, done, q_values = self.step(
                    state, observations["questions"], greedy=False
                )
                self.process_room_observations(observations["room"])
                next_state = self.get_deepcopied_memory_state()
                self.replay_buffer.store(*[state, action, reward, next_state, done])
                self.q_values["train"].append(q_values)
                score += reward
                self.iteration_idx += 1

            if done:
                new_episode_starts = True

                self.scores["train"].append(score)
                score = 0
                with torch.no_grad():
                    self.validate()

            if not new_episode_starts:
                loss = update_model(
                    replay_buffer=self.replay_buffer,
                    optimizer=self.optimizer,
                    device=self.device,
                    dqn=self.dqn,
                    dqn_target=self.dqn_target,
                    ddqn=self.ddqn,
                    gamma=self.gamma,
                )

                self.training_loss.append(loss)

                # linearly decay epsilon
                self.epsilon = update_epsilon(
                    self.epsilon,
                    self.max_epsilon,
                    self.min_epsilon,
                    self.epsilon_decay_until,
                )
                self.epsilons.append(self.epsilon)

                # if hard update is needed
                if self.iteration_idx % self.target_update_interval == 0:
                    target_hard_update(self.dqn, self.dqn_target)

                # plotting & show training results
                if (
                    self.iteration_idx == self.num_iterations
                    or self.iteration_idx % self.plotting_interval == 0
                ):
                    self.plot_results("all", save_fig=True)

                if self.iteration_idx == self.num_iterations:
                    break

        with torch.no_grad():
            self.test()

        self.env.close()

    def validate_test_middle(self, val_or_test: str) -> tuple[list, list, list, list]:
        """A function shared by explore validation and test in the middle.

        Args:
            val_or_test: "val" or "test"

        Returns:
            scores_local: a list of total episode rewards
            states_local: memory states
            q_values_local: q values
            actions_local: greey actions taken

        """
        scores_local = []
        states_local = []
        q_values_local = []
        actions_local = []

        for idx in range(self.num_samples_for_results):
            new_episode_starts = True
            score = 0
            while True:
                if new_episode_starts:
                    self.init_memory_systems()
                    observations, info = self.env.reset()
                    done = False
                    self.process_room_observations(observations["room"])
                    new_episode_starts = False

                else:
                    state = self.get_deepcopied_memory_state()
                    observations, action, reward, done, q_values = self.step(
                        state, observations["questions"], greedy=True
                    )
                    if not done:
                        self.process_room_observations(observations["room"])
                    score += reward

                    if idx == self.num_samples_for_results - 1:
                        states_local.append(state)
                        q_values_local.append(q_values)
                        self.q_values[val_or_test].append(q_values)
                        actions_local.append(action)

                if done:
                    break

            scores_local.append(score)

        return scores_local, states_local, q_values_local, actions_local
