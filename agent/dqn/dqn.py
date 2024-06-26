"""DQN Agent for the RoomEnv2 environment.

This should be inherited. This itself should not be used.
"""

import datetime
import os
from copy import deepcopy
import shutil
from typing import Literal

import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim

from humemai.memory import EpisodicMemory, MemorySystems, SemanticMemory, ShortMemory
from humemai.policy import (
    answer_question,
    encode_observation,
    explore,
    manage_memory,
    argmax,
)
from humemai.utils import write_yaml, is_running_notebook


from .nn import LSTM, MLP
from .utils import (
    ReplayBuffer,
    plot_results,
    save_final_results,
    save_states_q_values_actions,
    save_validation,
    select_action,
    target_hard_update,
    update_epsilon,
    update_model,
)


class DQNAgent:
    """DQN Agent interacting with environment.

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
        gamma: dict[str, float] = {"mm": 0.99, "explore": 0.9},
        capacity: dict = {
            "episodic": 16,
            "semantic": 16,
            "short": 1,
        },
        pretrain_semantic: str | bool = False,
        lstm_params: dict = {
            "hidden_size": 64,
            "num_layers": 2,
            "embedding_dim": 64,
            "max_timesteps": 100,
            "max_strength": 100,
        },
        mlp_params: dict = {"hidden_size": 64},
        num_samples_for_results: int = 10,
        plotting_interval: int = 10,
        train_seed: int = 5,
        test_seed: int = 0,
        device: Literal["cpu", "cuda"] = "cpu",
        qa_function: Literal[
            "episodic_semantic", "episodic", "semantic", "random"
        ] = "episodic_semantic",
        explore_policy_heuristic: Literal["avoid_walls", "random"] = "avoid_walls",
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
        default_root_dir: str = "./training-results/",
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
            lstm_params: parameters for the neural network (DQN)
            num_samples_for_results: The number of samples to validate / test the agent.
            plotting_interval: interval to plot results
            train_seed: seed for training
            test_seed: seed for testing
            device: This is either "cpu" or "cuda".
            qa_function: question answering function. Choose one of "episodic_semantic",
                "episodic", "semantic", or "random". This is the reward function.
            env_config: The configuration of the environment.
            ddqn: whether to use double DQN
            dueling_dqn: whether to use dueling DQN
            default_root_dir: default root directory to save results

        """
        params_to_save = deepcopy(locals())
        if "self" in params_to_save:
            del params_to_save["self"]
        if "__class__" in params_to_save:
            del params_to_save["__class__"]
        self.default_root_dir = os.path.join(
            default_root_dir, str(datetime.datetime.now())
        )
        self._create_directory(params_to_save)

        self.qa_function = qa_function
        self.explore_policy_heuristic = explore_policy_heuristic

        self.train_seed = train_seed
        self.test_seed = test_seed
        self.env_config = env_config
        self.env_config["seed"] = self.train_seed
        self.env_str = env_str
        self.num_samples_for_results = num_samples_for_results
        self.capacity = capacity
        self.pretrain_semantic = pretrain_semantic
        self.env = gym.make(self.env_str, **self.env_config)

        self.action2str = {
            "explore": {
                0: "north",
                1: "east",
                2: "south",
                3: "west",
                4: "stay",
            },
        }

        if capacity["episodic"] > 0 and capacity["semantic"] > 0:
            self.action2str["mm"] = {0: "episodic", 1: "semantic", 2: "forget"}
            self.memory_types = {
                "mm": ["short", "episodic", "semantic"],
                "explore": ["episodic", "semantic"],
            }
        elif capacity["episodic"] > 0:
            self.action2str["mm"] = {0: "episodic", 1: "forget"}
            self.memory_types = {"mm": ["short", "episodic"], "explore": ["episodic"]}
        elif capacity["semantic"] > 0:
            self.action2str["mm"] = {0: "semantic", 1: "forget"}
            self.memory_types = {"mm": ["short", "semantic"], "explore": ["semantic"]}

        else:
            raise ValueError(
                "At least one of episodic or semantic memory should be > 0"
            )

        self.action_space = {
            mt: gym.spaces.Discrete(len(self.action2str[mt]))
            for mt in ["mm", "explore"]
        }

        self.device = torch.device(device)
        print(f"Running on {self.device}")

        self.ddqn = ddqn
        self.dueling_dqn = dueling_dqn

        self.val_dir_names = {"mm": [], "explore": []}
        self.is_notebook = is_running_notebook()
        self.num_iterations = num_iterations
        self.plotting_interval = plotting_interval

        self.replay_buffer_size = replay_buffer_size
        self.batch_size = batch_size
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.epsilon = self.max_epsilon
        self.epsilon_decay_until = epsilon_decay_until
        self.target_update_interval = target_update_interval
        self.gamma = gamma
        self.warm_start = warm_start
        assert self.batch_size <= self.warm_start <= self.replay_buffer_size

        self.device = torch.device(device)
        self.lstm_params = lstm_params
        self.lstm_params["capacity"] = capacity
        self.lstm_params["device"] = self.device
        self.lstm_params["entities"] = self.env.unwrapped.entities
        self.lstm_params["relations"] = self.env.unwrapped.relations

        self.lstm_mm = LSTM(**self.lstm_params)
        self.lstm_mm_target = LSTM(**self.lstm_params)
        self.lstm_mm_target.load_state_dict(self.lstm_mm.state_dict())
        self.lstm_mm_target.eval()

        self.lstm_explore = LSTM(**self.lstm_params)
        self.lstm_explore_target = LSTM(**self.lstm_params)
        self.lstm_explore_target.eval()

        self.mlp_params = mlp_params
        self.mlp_params["device"] = self.device
        self.mlp_params["dueling_dqn"] = self.dueling_dqn

        self.mlp_mm = MLP(n_actions=len(self.action2str["mm"]), **self.mlp_params)
        self.mlp_mm_target = MLP(
            n_actions=len(self.action2str["mm"]), **self.mlp_params
        )
        self.mlp_mm_target.load_state_dict(self.mlp_mm.state_dict())
        self.mlp_mm_target.eval()

        self.mlp_explore = MLP(
            n_actions=len(self.action2str["explore"]), **self.mlp_params
        )
        self.mlp_explore_target = MLP(
            n_actions=len(self.action2str["explore"]), **self.mlp_params
        )
        self.mlp_explore_target.load_state_dict(self.mlp_explore.state_dict())
        self.mlp_explore_target.eval()

        self._save_number_of_parameters()

        self.q_values = {
            "mm": {"train": [], "val": [], "test": []},
            "explore": {"train": [], "val": [], "test": []},
        }
        self.epsilons = {"mm": [], "explore": []}
        self.training_loss = {"mm": [], "explore": []}
        self.scores = {
            "mm": {"train": [], "val": [], "test": None},
            "explore": {"train": [], "val": [], "test": None},
        }

    def _create_directory(self, params_to_save: dict) -> None:
        """Create the directory to store the results."""
        os.makedirs(self.default_root_dir, exist_ok=True)
        write_yaml(params_to_save, os.path.join(self.default_root_dir, "train.yaml"))

    def _save_number_of_parameters(self) -> None:
        """Save the number of parameters in the model."""
        write_yaml(
            {
                "lstm_mm": sum(p.numel() for p in self.lstm_mm.parameters()),
                "mlp_mm": sum(p.numel() for p in self.mlp_mm.parameters()),
                "lstm_explore": sum(p.numel() for p in self.lstm_explore.parameters()),
                "mlp_explore": sum(p.numel() for p in self.mlp_explore.parameters()),
            },
            os.path.join(self.default_root_dir, "num_params.yaml"),
        )

    def remove_results_from_disk(self) -> None:
        """Remove the results from the disk."""
        shutil.rmtree(self.default_root_dir)

    def init_memory_systems(self) -> None:
        """Initialize the agent's memory systems. This has nothing to do with the
        replay buffer."""
        self.memory_systems = MemorySystems(
            episodic=EpisodicMemory(
                capacity=self.capacity["episodic"], remove_duplicates=False
            ),
            semantic=SemanticMemory(capacity=self.capacity["semantic"]),
            short=ShortMemory(capacity=self.capacity["short"]),
        )

        assert self.pretrain_semantic in [False, "exclude_walls", "include_walls"]
        if self.pretrain_semantic in ["exclude_walls", "include_walls"]:
            if self.pretrain_semantic == "exclude_walls":
                exclude_walls = True
            else:
                exclude_walls = False
            room_layout = self.env.unwrapped.return_room_layout(exclude_walls)

            assert self.capacity["semantic"] > 0
            _ = self.memory_systems.semantic.pretrain_semantic(
                semantic_knowledge=room_layout,
                return_remaining_space=False,
                freeze=False,
            )

    def get_deepcopied_memory_state(self) -> dict:
        """Get a deepcopied memory state.

        This is necessary because the memory state is a list of dictionaries, which is
        mutable.

        Returns:
            deepcopied memory_state
        """
        return deepcopy(self.memory_systems.return_as_a_dict_list())

    def process_first_observation(self, observations_room: list) -> list:
        """Encode the first observation into the short-term memory system.

        This is necessary since we are dealing with the observations one by one.

        Args:
            observations_room: observations["room"] from the environment

        Returns:
            observations_room[1:]: remaining observations

        """
        encode_observation(self.memory_systems, observations_room[0])

        return observations_room[1:]

    def process_remaining_observations(
        self, remaining: list, greedy: bool
    ) -> tuple[list, list, list]:
        """Process remaining observations.

        This is necessary since we are dealing with the observations one by one.

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
                memory_types=self.memory_types["mm"],
                state=state,
                greedy=greedy,
                lstm=self.lstm_mm,
                mlp=self.mlp_mm,
                epsilon=self.epsilon,
                action_space=self.action_space["mm"],
            )
            manage_memory(
                self.memory_systems,
                self.action2str["mm"][action],
                split_possessive=False,
            )
            encode_observation(self.memory_systems, obs)
            state = self.get_deepcopied_memory_state()
            states.append(deepcopy(state))
            actions.append(action)
            q_values.append(q_values_)

        assert len(states) == len(actions) + 1 == len(q_values) + 1

        return states, actions, q_values

    def step_mm(
        self, state: dict, questions: list, greedy: bool
    ) -> tuple[dict, int, float, bool, list]:
        """Step through the environment with the memory management policy.

        env.step() is called here. Make sure to call this function to interact with
        the environment.

        Args:
            state: state of the memory systems
            questions: questions to answer
            greedy: whether to act greedily

        Returns:
            observations: observations from the environment
            action: memory management action taken
            reward: reward received
            done: whether the episode is done
            q_values: q values (list of floats)

        """
        # select memory management action
        action, q_values = select_action(
            memory_types=self.memory_types["mm"],
            state=state,
            greedy=greedy,
            lstm=self.lstm_mm,
            mlp=self.mlp_mm,
            epsilon=self.epsilon,
            action_space=self.action_space["mm"],
        )
        manage_memory(
            self.memory_systems,
            self.action2str["mm"][action],
            split_possessive=False,
        )
        action_explore = explore(self.memory_systems, self.explore_policy_heuristic)
        answers = [
            answer_question(
                self.memory_systems, self.qa_function, q, split_possessive=False
            )
            for q in questions
        ]
        action_pair = (answers, action_explore)
        (
            observations,
            reward,
            done,
            truncated,
            info,
        ) = self.env.step(action_pair)
        done = done or truncated

        return observations, action, reward, done, q_values

    def fill_replay_buffer_mm(self) -> None:
        """Make the replay buffer full in the beginning with the uniformly-sampled
        actions. The filling continues until it reaches the warm start size.

        """
        self.replay_buffer = ReplayBuffer(
            observation_type="dict",
            size=self.replay_buffer_size,
            batch_size=self.batch_size,
        )

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
                observations, action, reward, done, q_values = self.step_mm(
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

    def train_mm(self) -> None:
        """Train the memory management policy."""
        os.makedirs(os.path.join(self.default_root_dir, "mm"), exist_ok=True)
        self.optimizer = optim.Adam(
            list(self.lstm_mm.parameters()) + list(self.mlp_mm.parameters())
        )

        self.fill_replay_buffer_mm()
        self.num_validation = 0
        self.epsilon = self.max_epsilon

        self.lstm_mm.train()
        self.mlp_mm.train()

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
                observations, action, reward, done, q_values = self.step_mm(
                    state, observations["questions"], greedy=False
                )
                remaining = self.process_first_observation(observations["room"])
                next_state = self.get_deepcopied_memory_state()
                self.replay_buffer.store(*[state, action, reward, next_state, done])
                self.q_values["mm"]["train"].append(q_values)
                score += reward
                self.iteration_idx += 1

            if done:
                new_episode_starts = True

                self.scores["mm"]["train"].append(score)
                score = 0
                with torch.no_grad():
                    self.validate(self.lstm_mm, self.mlp_mm, "mm")

            else:
                states_, actions_, q_values_ = self.process_remaining_observations(
                    remaining, greedy=False
                )
                reward = 0
                for state, next_state, action in zip(
                    states_[:-1], states_[1:], actions_
                ):
                    self.replay_buffer.store(*[state, action, reward, next_state, done])
                self.q_values["mm"]["train"].extend(q_values_)

            if not new_episode_starts:
                loss = update_model(
                    memory_types=self.memory_types["mm"],
                    replay_buffer=self.replay_buffer,
                    optimizer=self.optimizer,
                    device=self.device,
                    lstm=self.lstm_mm,
                    lstm_target=self.lstm_mm_target,
                    mlp=self.mlp_mm,
                    mlp_target=self.mlp_mm_target,
                    ddqn=self.ddqn,
                    gamma=self.gamma["mm"],
                )

                self.training_loss["mm"].append(loss)

                # linearly decay epsilon
                self.epsilon = update_epsilon(
                    self.epsilon,
                    self.max_epsilon,
                    self.min_epsilon,
                    self.epsilon_decay_until,
                )
                self.epsilons["mm"].append(self.epsilon)

                # if hard update is needed
                if self.iteration_idx % self.target_update_interval == 0:
                    target_hard_update(self.lstm_mm, self.lstm_mm_target)
                    target_hard_update(self.mlp_mm, self.mlp_mm_target)

                # plotting & show training results
                if (
                    self.iteration_idx == self.num_iterations
                    or self.iteration_idx % self.plotting_interval == 0
                ):
                    self.plot_results("mm", "all", save_fig=True)

                if self.iteration_idx == self.num_iterations:
                    break

        with torch.no_grad():
            self.test(self.lstm_mm, self.mlp_mm, "mm")

        self.env.close()

    def validate_test_middle_mm(
        self, val_or_test: str
    ) -> tuple[list, list, list, list]:
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
                    observations, action, reward, done, q_values = self.step_mm(
                        state, observations["questions"], greedy=True
                    )
                    remaining = self.process_first_observation(observations["room"])
                    score += reward

                    if idx == self.num_samples_for_results - 1:
                        states_local.append(state)
                        q_values_local.append(q_values)
                        self.q_values["mm"][val_or_test].append(q_values)
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
                        self.q_values["mm"][val_or_test].extend(q_values_)
                        actions_local.extend(actions_)

            scores_local.append(score)

        return scores_local, states_local, q_values_local, actions_local

    def validate(
        self,
        lstm: torch.nn.Module,
        mlp: torch.nn.Module,
        policy_type: Literal["mm", "explore"],
    ) -> None:
        """Validate the memory management agent.

        Args:
            lstm: lstm model
            mlp: mlp model
            policy_type: "mm" or "explore"

        """
        lstm.eval()
        mlp.eval()

        if policy_type == "mm":
            middle_function = self.validate_test_middle_mm
        elif policy_type == "explore":
            middle_function = self.validate_test_middle_explore
        else:
            raise ValueError("policy_type should be either 'mm' or 'explore'")

        scores_temp, states, q_values, actions = middle_function("val")

        save_validation(
            policy=policy_type,
            scores_temp=scores_temp,
            scores=self.scores[policy_type],
            default_root_dir=self.default_root_dir,
            num_validation=self.num_validation,
            val_dir_names=self.val_dir_names[policy_type],
            lstm=lstm,
            mlp=mlp,
        )
        save_states_q_values_actions(
            policy_type,
            states,
            q_values,
            actions,
            self.default_root_dir,
            "val",
            self.num_validation,
        )
        self.env.close()
        self.num_validation += 1
        lstm.train()
        mlp.train()

    def test(
        self,
        lstm: torch.nn.Module,
        mlp: torch.nn.Module,
        policy_type: Literal["mm", "explore"],
        checkpoint_lstm: str | None = None,
        checkpoint_mlp: str | None = None,
    ) -> None:
        """Test the memory management agent.

        Args:
            lstm: lstm model
            mlp: mlp model
            checkpoint_lstm: checkpoint for the lstm
            checkpoint_mlp: checkpoint for the mlp
        """
        lstm.eval()
        mlp.eval()
        self.env_config["seed"] = self.test_seed
        self.env = gym.make(self.env_str, **self.env_config)

        assert len(self.val_dir_names[policy_type]) == 1
        lstm.load_state_dict(
            torch.load(os.path.join(self.val_dir_names[policy_type][0], "lstm.pt"))
        )
        mlp.load_state_dict(
            torch.load(os.path.join(self.val_dir_names[policy_type][0], "mlp.pt"))
        )

        if checkpoint_lstm is not None:
            lstm.load_state_dict(torch.load(checkpoint_lstm))
        if checkpoint_mlp is not None:
            mlp.load_state_dict(torch.load(checkpoint_mlp))

        if policy_type == "mm":
            middle_function = self.validate_test_middle_mm
        elif policy_type == "explore":
            middle_function = self.validate_test_middle_explore
        else:
            raise ValueError("policy_type should be either mm or explore")

        scores, states, q_values, actions = middle_function("test")
        self.scores[policy_type]["test"] = scores

        save_final_results(
            self.scores[policy_type],
            self.training_loss[policy_type],
            self.default_root_dir,
            self.q_values[policy_type],
            self,
            policy_type,
        )
        save_states_q_values_actions(
            policy_type,
            states,
            q_values,
            actions,
            self.default_root_dir,
            "test",
        )

        self.plot_results(policy_type, "all", save_fig=True)
        self.env.close()

    def process_room_observations(self, observations_room: list):
        """Process room observations.

        Args:
            observations_room: observations["room"] from the environment

        """
        for obs in observations_room:
            encode_observation(self.memory_systems, obs)

            state = self.memory_systems.return_as_a_dict_list()

            with torch.no_grad():
                q_values = (
                    self.mlp_mm(
                        self.lstm_mm(np.array([state]), self.memory_types["mm"])
                    )
                    .detach()
                    .cpu()
                    .tolist()[0]
                )
                action = argmax(q_values)

            manage_memory(
                memory_systems=self.memory_systems,
                policy=self.action2str["mm"][action],
                split_possessive=False,
            )

    def step_explore(
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
            action: exploration action taken
            reward: reward received
            done: whether the episode is done
            q_values: q values (list of floats)

        """
        # select explore action
        action, q_values = select_action(
            memory_types=self.memory_types["explore"],
            state=state,
            greedy=greedy,
            lstm=self.lstm_explore,
            mlp=self.mlp_explore,
            epsilon=self.epsilon,
            action_space=self.action_space["explore"],
        )
        answers = [
            answer_question(
                self.memory_systems, self.qa_function, q, split_possessive=False
            )
            for q in questions
        ]

        action_pair = (answers, self.action2str["explore"][action])
        (
            observations,
            reward,
            done,
            truncated,
            info,
        ) = self.env.step(action_pair)
        done = done or truncated

        return observations, action, reward, done, q_values

    def fill_replay_buffer_explore(self) -> None:
        """Make the replay buffer full in the beginning with the uniformly-sampled
        actions. The filling continues until it reaches the warm start size.

        """
        self.replay_buffer = ReplayBuffer(
            observation_type="dict",
            size=self.replay_buffer_size,
            batch_size=self.batch_size,
        )

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
                observations, action, reward, done, q_values = self.step_explore(
                    state, observations["questions"], greedy=False
                )
                self.process_room_observations(observations["room"])
                next_state = self.get_deepcopied_memory_state()
                self.replay_buffer.store(*[state, action, reward, next_state, done])

            if done:
                new_episode_starts = True

    def train_explore(self) -> None:
        """Train the exploration agent.

        The exploration agent is initialized with the memory management agent's
        best weights. Consider it as finetuing the exploration agent. Of cousre the
        mlp_explore is randomly initialized.

        """
        os.makedirs(os.path.join(self.default_root_dir, "explore"), exist_ok=True)

        # Assuming that self.lstm_mm is already trained and the best weights are loaded.
        self.lstm_explore.load_state_dict(self.lstm_mm.state_dict())
        self.lstm_explore_target.load_state_dict(self.lstm_mm.state_dict())

        # Freeze the weights of self.lstm_mm and self.mlp_mm
        self.lstm_mm.eval()
        for param in self.lstm_mm.parameters():
            param.requires_grad = False

        self.mlp_mm.eval()
        for param in self.mlp_mm.parameters():
            param.requires_grad = False

        # optimizer
        self.optimizer = optim.Adam(
            list(self.lstm_explore.parameters()) + list(self.mlp_explore.parameters())
        )

        self.fill_replay_buffer_explore()
        self.num_validation = 0
        self.epsilon = self.max_epsilon

        self.lstm_explore.train()
        self.mlp_explore.train()

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
                observations, action, reward, done, q_values = self.step_explore(
                    state, observations["questions"], greedy=False
                )
                self.process_room_observations(observations["room"])
                next_state = self.get_deepcopied_memory_state()
                self.replay_buffer.store(*[state, action, reward, next_state, done])
                self.q_values["explore"]["train"].append(q_values)
                score += reward
                self.iteration_idx += 1

            if done:
                new_episode_starts = True

                self.scores["explore"]["train"].append(score)
                score = 0
                with torch.no_grad():
                    self.validate(self.lstm_explore, self.mlp_explore, "explore")

            if not new_episode_starts:
                loss = update_model(
                    memory_types=self.memory_types["explore"],
                    replay_buffer=self.replay_buffer,
                    optimizer=self.optimizer,
                    device=self.device,
                    lstm=self.lstm_explore,
                    lstm_target=self.lstm_explore_target,
                    mlp=self.mlp_explore,
                    mlp_target=self.mlp_explore_target,
                    ddqn=self.ddqn,
                    gamma=self.gamma["explore"],
                )

                self.training_loss["explore"].append(loss)

                # linearly decay epsilon
                self.epsilon = update_epsilon(
                    self.epsilon,
                    self.max_epsilon,
                    self.min_epsilon,
                    self.epsilon_decay_until,
                )
                self.epsilons["explore"].append(self.epsilon)

                # if hard update is needed
                if self.iteration_idx % self.target_update_interval == 0:
                    target_hard_update(self.lstm_explore, self.lstm_explore_target)
                    target_hard_update(self.mlp_explore, self.mlp_explore_target)

                # plotting & show training results
                if (
                    self.iteration_idx == self.num_iterations
                    or self.iteration_idx % self.plotting_interval == 0
                ):
                    self.plot_results("explore", "all", save_fig=True)

                if self.iteration_idx == self.num_iterations:
                    break

        with torch.no_grad():
            self.test(self.lstm_explore, self.mlp_explore, "explore")

        self.env.close()

    def validate_test_middle_explore(
        self, val_or_test: str
    ) -> tuple[list, list, list, list]:
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
                    observations, action, reward, done, q_values = self.step_explore(
                        state, observations["questions"], greedy=True
                    )
                    if not done:
                        self.process_room_observations(observations["room"])
                    score += reward

                    if idx == self.num_samples_for_results - 1:
                        states_local.append(state)
                        q_values_local.append(q_values)
                        self.q_values["explore"][val_or_test].append(q_values)
                        actions_local.append(action)

                if done:
                    break

            scores_local.append(score)

        return scores_local, states_local, q_values_local, actions_local

    def train(self) -> None:
        """Train the agent.

        First train the memory management agent, then train the exploration agent.

        """
        self.train_mm()
        self.train_explore()

    def plot_results(
        self,
        policy_type: Literal["mm", "explore"],
        to_plot: str = "all",
        save_fig: bool = False,
    ) -> None:
        """Plot things for DQN training.

        Args:
            policy_type: which policy to plot, "mm" or "explore"
            to_plot: what to plot:
                training_td_loss
                epsilons
                training_score
                validation_score
                test_score
                q_values_train
                q_values_val
                q_values_test
            save_fig: whether to save the figure.

        """
        plot_results(
            policy_type,
            self.scores[policy_type],
            self.training_loss[policy_type],
            self.epsilons[policy_type],
            self.q_values[policy_type],
            self.iteration_idx,
            self.action_space[policy_type].n.item(),
            self.num_iterations,
            self.env.unwrapped.total_maximum_episode_rewards,
            self.default_root_dir,
            to_plot,
            save_fig,
        )
