"""DQN memory management Agent for the RoomEnv2 environment."""

import os
from copy import deepcopy

import gymnasium as gym
import torch
from humemai.policy import answer_question, encode_observation, explore, manage_memory
from humemai.utils import write_yaml

from .dqn import DQNAgent
from .utils import select_action, target_hard_update, update_epsilon, update_model


class DQNMMAgent(DQNAgent):
    """DQN memory management Agent interacting with environment.

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
                "short",
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
        qa_policy: str = "episodic_semantic",
        explore_policy: str = "avoid_walls",
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
        default_root_dir: str = "./training-results/stochastic-objects/DQN/mm",
        run_handcrafted_baselines: bool = False,
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
            qa_policy: question answering policy Choose one of "episodic_semantic",
                "random", or "neural". qa_policy shouldn't be trained with RL. There is
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
            ddqn: whether to use double DQN
            dueling_dqn: whether to use dueling DQN
            default_root_dir: default root directory to save results
            run_handcrafted_baselines: whether to run handcrafted baselines

        """
        all_params = deepcopy(locals())
        del all_params["self"]
        del all_params["__class__"]
        self.all_params = deepcopy(all_params)

        # action: 1. move to episodic, 2. move to semantic, 3. forget
        if capacity["episodic"] > 0 and capacity["semantic"] > 0:
            self.action2str = {0: "episodic", 1: "semantic", 2: "forget"}
        elif capacity["episodic"] > 0:
            self.action2str = {0: "episodic", 1: "forget"}
        elif capacity["semantic"] > 0:
            self.action2str = {0: "semantic", 1: "forget"}
        self.action_space = gym.spaces.Discrete(len(self.action2str))

        all_params["nn_params"]["n_actions"] = len(self.action2str)
        all_params["mm_policy"] = "rl"
        super().__init__(**all_params)
        write_yaml(self.all_params, os.path.join(self.default_root_dir, "train.yaml"))

    def process_first_observation(self, observations_room: list) -> list:
        """Process observations.

        Args:
            observations_room: observations["room"] from the environment

        Returns:
            observations_room[1:]: remaining observations

        """
        observations_room = self.manage_agent_and_map_memory(observations_room)
        encode_observation(self.memory_systems, observations_room[0])

        return observations_room[1:]

    def process_remaining_observations(
        self, remaining: list, greedy: bool
    ) -> tuple[list, list, list]:
        """Process remaining observations.

        Args:
            remaining: observations["room"] from the environment
            greedy: whether to act greedily

        Returns:
            states: consecutive states by taking consecutive actions.
            actions: actions taken (list of ints)
            q_values: q values (list of lists of floats)

        """
        states = []
        actions = []
        q_values = []
        state = self.get_deepcopied_memory_state()
        states.append(state)

        for obs in remaining:
            # select memory management action
            action, q_values_ = select_action(
                state=state,
                greedy=greedy,
                dqn=self.dqn,
                epsilon=self.epsilon,
                action_space=self.action_space,
            )
            manage_memory(
                self.memory_systems, self.action2str[action], split_possessive=False
            )
            encode_observation(self.memory_systems, obs)
            state = self.get_deepcopied_memory_state()
            states.append(deepcopy(state))
            actions.append(action)
            q_values.append(q_values_)

        assert len(states) == len(actions) + 1 == len(q_values) + 1

        return states, actions, q_values

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
            action_mm: memory management action taken
            reward: reward received
            done: whether the episode is done
            q_values: q values (list of floats)

        """
        # select memory management action
        action_mm, q_values = select_action(
            state=state,
            greedy=greedy,
            dqn=self.dqn,
            epsilon=self.epsilon,
            action_space=self.action_space,
        )
        manage_memory(
            self.memory_systems, self.action2str[action_mm], split_possessive=False
        )
        action_explore = explore(self.memory_systems, self.explore_policy)
        actions_qa = [
            answer_question(self.memory_systems, self.qa_policy, q) for q in questions
        ]
        action_pair = (actions_qa, action_explore)
        (
            observations,
            reward,
            done,
            truncated,
            info,
        ) = self.env.step(action_pair)
        done = done or truncated

        return observations, action_mm, reward, done, q_values

    def fill_replay_buffer(self) -> None:
        """Make the replay buffer full in the beginning with the uniformly-sampled
        actions. The filling continues until it reaches the warm start size.

        """
        new_episode_starts = True
        while len(self.replay_buffer) < self.warm_start:

            if new_episode_starts:
                self.init_memory_systems()
                observations, info = self.env.reset()
                done = False
                remaining = self.process_first_observation(observations["room"])
                new_episode_starts = False

            else:
                state = self.get_deepcopied_memory_state()
                observations, action, reward, done, q_values = self.step(
                    state, observations["questions"], greedy=False
                )
                remaining = self.process_first_observation(observations["room"])
                next_state = self.get_deepcopied_memory_state()
                self.replay_buffer.store(*[state, action, reward, next_state, done])

            if done:
                new_episode_starts = True

            else:
                states_, actions_, q_values_ = self.process_remaining_observations(
                    remaining, greedy=False
                )
                reward = 0
                done = False
                for state, next_state, action in zip(
                    states_[:-1], states_[1:], actions_
                ):
                    self.replay_buffer.store(*[state, action, reward, next_state, done])

    def train(self) -> None:
        """Train the memory management agent."""
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
                remaining = self.process_first_observation(observations["room"])
                new_episode_starts = False

            else:
                state = self.get_deepcopied_memory_state()
                observations, action, reward, done, q_values = self.step(
                    state, observations["questions"], greedy=False
                )
                remaining = self.process_first_observation(observations["room"])
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

            else:
                states_, actions_, q_values_ = self.process_remaining_observations(
                    remaining, greedy=False
                )
                reward = 0
                for state, next_state, action in zip(
                    states_[:-1], states_[1:], actions_
                ):
                    self.replay_buffer.store(*[state, action, reward, next_state, done])
                self.q_values["train"].extend(q_values_)

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
        """A function shared by validation and test in the middle.

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
                    remaining = self.process_first_observation(observations["room"])
                    new_episode_starts = False

                else:
                    state = self.get_deepcopied_memory_state()
                    observations, action, reward, done, q_values = self.step(
                        state, observations["questions"], greedy=True
                    )
                    remaining = self.process_first_observation(observations["room"])
                    score += reward

                    if idx == self.num_samples_for_results - 1:
                        states_local.append(state)
                        q_values_local.append(q_values)
                        self.q_values[val_or_test].append(q_values)
                        actions_local.append(action)

                if done:
                    break

                else:
                    states_, actions_, q_values_ = self.process_remaining_observations(
                        remaining, greedy=True
                    )

                    if idx == self.num_samples_for_results - 1:
                        states_local.extend(states_[:-1])
                        q_values_local.extend(q_values_)
                        self.q_values[val_or_test].extend(q_values_)
                        actions_local.extend(actions_)

            scores_local.append(score)

        return scores_local, states_local, q_values_local, actions_local
