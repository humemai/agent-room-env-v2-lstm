"""Agent that uses a GNN for its DQN, for the RoomEnv2 environment."""

import datetime
import os
import shutil
from copy import deepcopy
from typing import Literal

import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
from humemai_research.memory import LongMemory, MemorySystems, ShortMemory
from humemai_research.utils import is_running_notebook, write_yaml

from ..policy import (answer_question, encode_all_observations, explore,
                      manage_memory)
from .nn import GNN
from .utils import (ReplayBuffer, plot_results, save_final_results,
                    save_states_q_values_actions, save_validation,
                    select_action, target_hard_update, update_epsilon,
                    update_model)


class DQNAgent:
    r"""DQN Agent interacting with environment.

    This is an upgrade from https://github.com/humemai/agent-room-env-v2-lstm. The two
    policies, i.e., memory management and exploration, are learned by a GNN at once!
    The question-answering function is still hand-crafted. Potentially, this can be
    learned by a contextual bandit algorithm.

    Based on https://github.com/Curt-Park/rainbow-is-all-you-need/
    """

    def __init__(
        self,
        env_str: str = "room_env:RoomEnv-v2",
        num_iterations: int = 10000,
        replay_buffer_size: int = 1000,
        warm_start: int = 32,
        batch_size: int = 32,
        target_update_interval: int = 10,
        epsilon_decay_until: float = 10000,
        max_epsilon: float = 1.0,
        min_epsilon: float = 0.01,
        gamma: dict = {"mm": 0.9, "explore": 0.9},
        learning_rate: int = 0.001,
        capacity: dict = {
            "long": 12,
            "short": 15,
        },
        pretrain_semantic: Literal[False, "include_walls", "exclude_walls"] = False,
        semantic_decay_factor: float = 1.0,
        dqn_params: dict = {
            "gcn_layer_params": {
                "type": "StarE",
                "embedding_dim": 8,
                "num_layers": 2,
                "gcn_drop": 0.1,
                "triple_qual_weight": 0.8,
            },
            "relu_between_gcn_layers": True,
            "dropout_between_gcn_layers": True,
            "mlp_params": {"num_hidden_layers": 2, "dueling_dqn": True},
        },
        num_samples_for_results: dict = {"val": 5, "test": 10},
        validation_interval: int = 5,
        plotting_interval: int = 20,
        train_seed: int = 5,
        test_seed: int = 0,
        device: Literal["cpu", "cuda"] = "cpu",
        env_config: dict = {
            "question_prob": 1.0,
            "terminates_at": 99,
            "randomize_observations": "all",
            "room_size": "l",
            "rewards": {"correct": 1, "wrong": 0, "partial": 0},
            "make_everything_static": False,
            "num_total_questions": 1000,
            "question_interval": 1,
            "include_walls_in_observations": True,
        },
        intrinsic_explore_reward: float = 1.0,
        ddqn: bool = True,
        default_root_dir: str = "./training-results/",
        explore_policy: Literal["random", "avoid_walls", "RL"] = "RL",
        qa_function: str = "latest_strongest",
        mm_policy: Literal[
            "random", "episodic", "semantic", "forget", "RL", "handcrafted"
        ] = "RL",
        scale_reward: bool = False,
    ) -> None:
        r"""Initialization.

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
            learning_rate: learning rate for the optimizer
            capacity: The capacity of each human-like memory systems
            pretrain_semantic: whether to pretrain the semantic memory system.
            semantic_decay_factor: decay factor for the semantic memory system
            dqn_params: parameters for the DQN
            num_samples_for_results: The number of samples to validate / test the agent.
            validation_interval: interval to validate
            plotting_interval: interval to plot results
            train_seed: seed for training
            test_seed: seed for testing
            device: This is either "cpu" or "cuda".
            env_config: The configuration of the environment.
                question_prob: The probability of a question being asked at every
                    observation.
                terminates_at: The maximum number of steps to take in an episode.
                seed: seed for env
                room_size: The room configuration to use. Choose one of "dev", "xxs",
                    "xs", "s", "m", or "l".
            intrinsic_explore_reward: intrinsic reward for exploration
            ddqn: whether to use double DQN
            default_root_dir: default root directory to save results
            explore_policy: exploration policy. Choose one of "random", "avoid_walls",
                "RL".
            qa_function: question answering policy Choose one of "episodic_semantic",
                "random", or "neural". qa_function shouldn't be trained with RL. There is
                no sequence of states / actions to learn from.
            mm_policy: memory management policy. Choose one of "random", "episodic",
                "semantic", "forget", "RL", or "handcrafted".
            scale_reward: whether to scale the reward

        """
        params_to_save = deepcopy(locals())
        del params_to_save["self"]
        self.default_root_dir = os.path.join(
            default_root_dir, str(datetime.datetime.now())
        )
        self._create_directory(params_to_save)

        self.train_seed = train_seed
        self.test_seed = test_seed
        env_config["seed"] = self.train_seed

        self.env_str = env_str
        self.env_config = env_config
        self.qa_function = qa_function
        assert self.qa_function in ["random", "latest_strongest", "strongest_latest"]
        self.num_samples_for_results = num_samples_for_results
        self.validation_interval = validation_interval
        self.capacity = capacity
        self.pretrain_semantic = pretrain_semantic
        self.semantic_decay_factor = semantic_decay_factor
        self.env = gym.make(self.env_str, **self.env_config)

        self.device = torch.device(device)
        print(f"Running on {self.device}")

        self.intrinsic_explore_reward = intrinsic_explore_reward
        self.ddqn = ddqn
        self.val_file_names = []
        self.is_notebook = is_running_notebook()
        self.num_iterations = num_iterations
        self.plotting_interval = plotting_interval

        self.replay_buffer_size = replay_buffer_size
        self.batch_size = batch_size
        self.epsilon = max_epsilon
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay_until = epsilon_decay_until
        self.target_update_interval = target_update_interval
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.warm_start = warm_start
        assert self.batch_size <= self.warm_start <= self.replay_buffer_size

        self.explore_policy = explore_policy
        self.mm_policy = mm_policy
        self.scale_reward = scale_reward

        self.action_mm2str = {0: "episodic", 1: "semantic", 2: "forget"}
        self.action_mm2int = {v: k for k, v in self.action_mm2str.items()}
        self.action_explore2str = {
            0: "north",
            1: "east",
            2: "south",
            3: "west",
            4: "stay",
        }
        self.action_explore2int = {v: k for k, v in self.action_explore2str.items()}

        self.init_memory_systems()

        self.dqn_params = dqn_params
        self.dqn_params["device"] = self.device
        self.dqn_params["entities"] = [
            e for entities in self.env.unwrapped.entities.values() for e in entities
        ]
        # We are gonna treat the real numbers 0, 1, ..., 100 as entities. This is
        # very stupid, but it is what it is.
        self.dqn_params["entities"] += [
            str(i) for i in range(self.env.unwrapped.terminates_at + 2)
        ]
        # Main triple relations have "inv", while qualifier relations don't have "inv".
        self.dqn_params["relations"] = (
            self.env.unwrapped.relations
            + [rel + "_inv" for rel in self.env.unwrapped.relations]
            + self.memory_systems.qualifier_relations
        )
        self.dqn = GNN(**self.dqn_params)
        self.dqn_target = GNN(**self.dqn_params)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()

        # optimizer
        self.optimizer = optim.Adam(list(self.dqn.parameters()), lr=self.learning_rate)

        self.q_values = {
            "train": {"mm": [], "explore": []},
            "val": {"mm": [], "explore": []},
            "test": {"mm": [], "explore": []},
        }
        self._save_number_of_parameters()

    def _create_directory(self, params_to_save: dict) -> None:
        r"""Create the directory to store the results.

        Args:
            params_to_save: parameters to save

        """
        os.makedirs(self.default_root_dir, exist_ok=True)
        write_yaml(params_to_save, os.path.join(self.default_root_dir, "train.yaml"))

    def _save_number_of_parameters(self) -> None:
        r"""Save the number of parameters in the model."""
        write_yaml(
            {
                "total": sum(p.numel() for p in self.dqn.parameters()),
                "gcn_layers": sum(p.numel() for p in self.dqn.gcn_layers.parameters()),
                "mlp_mm": sum(p.numel() for p in self.dqn.mlp_mm.parameters()),
                "mlp_explore": sum(
                    p.numel() for p in self.dqn.mlp_explore.parameters()
                ),
                "entity_embeddings": self.dqn.entity_embeddings.numel(),
                "relation_embeddings": self.dqn.relation_embeddings.numel(),
            },
            os.path.join(self.default_root_dir, "num_params.yaml"),
        )

    def remove_results_from_disk(self) -> None:
        r"""Remove the results from the disk."""
        shutil.rmtree(self.default_root_dir)

    def init_memory_systems(self) -> None:
        r"""Initialize the agent's memory systems. This has nothing to do with the
        replay buffer."""
        self.memory_systems = MemorySystems(
            short=ShortMemory(capacity=self.capacity["short"]),
            long=LongMemory(
                capacity=self.capacity["long"],
                semantic_decay_factor=self.semantic_decay_factor,
                min_strength=1,
            ),
        )

        assert self.pretrain_semantic in [False, "exclude_walls", "include_walls"]
        if self.pretrain_semantic in ["exclude_walls", "include_walls"]:

            if self.pretrain_semantic == "exclude_walls":
                exclude_walls = True
            else:
                exclude_walls = False
            room_layout = self.env.unwrapped.return_room_layout(exclude_walls)

            self.memory_systems.long.pretrain_semantic(semantic_knowledge=room_layout)

            if self.pretrain_semantic == "include_walls":
                assert self.memory_systems.long.size > 0
            elif "xxs" in self.env_config["room_size"]:
                assert self.memory_systems.long.size == 0
            else:
                assert self.memory_systems.long.size > 0

        self.num_semantic_decayed = 0

    def reset(self) -> None:
        """Reset the env and the memory systems. observations are encoded."""

        self.init_memory_systems()
        self.observations, info = self.env.reset()
        # 0. encode observations
        encode_all_observations(self.memory_systems, self.observations["room"])

    def step(self, greedy: bool) -> tuple[
        dict,
        list[int],
        list[list[float]],
        list[int],
        list[list[float]],
        float,
        float,
        list[str],
        bool,
    ]:
        r"""Step of the algorithm. This is the only step that interacts with the
        environment.

        Args:
            greedy: whether to use greedy policy

        Returns:
            a_explore, q_explore, a_mm, q_mm, reward, intrinsic_explore_reward, answers,
            done

        """
        assert not self.memory_systems.short.is_empty, "encode all observations first"
        # 1. explore
        if self.explore_policy.lower() == "rl":
            a_explore, q_explore = select_action(
                state=self.memory_systems.get_working_memory().to_list(),
                greedy=greedy,
                dqn=self.dqn,
                epsilon=self.epsilon,
                policy_type="explore",
            )
            if self.intrinsic_explore_reward > 0:
                intrinsic_explore_reward = self.get_intrinsic_explore_reward(
                    self.action_explore2str[a_explore.item()]
                )
            else:
                intrinsic_explore_reward = 0

        else:
            a_explore = explore(self.memory_systems, self.explore_policy)
            a_explore = np.array(self.action_explore2int[a_explore])
            # Create dummy Q-values
            q_explore = np.zeros((1, len(self.action_explore2str)))
            intrinsic_explore_reward = 0

        # 2. question answering
        answers = [
            answer_question(
                self.memory_systems,
                self.qa_function,
                question,
            )
            for question in self.observations["questions"]
        ]

        # 3. manage memory
        if self.mm_policy.lower() == "rl":
            a_mm, q_mm = select_action(
                state=self.memory_systems.get_working_memory().to_list(),
                greedy=greedy,
                dqn=self.dqn,
                epsilon=self.epsilon,
                policy_type="mm",
            )
        else:
            a_mm = []
            if self.mm_policy == "handcrafted":
                for mem_short in self.memory_systems.short:
                    if mem_short[0] == "agent":
                        a_mm.append(0)
                    elif "ind" in mem_short[0] or "dep" in mem_short[0]:
                        a_mm.append(0)
                    elif "sta" in mem_short[0]:
                        a_mm.append(1)
                    elif "room" in mem_short[0] and "room" in mem_short[2]:
                        a_mm.append(1)
                    elif "wall" in mem_short[2]:
                        a_mm.append(2)
                    else:
                        raise ValueError("something is wrong")
            else:
                raise NotImplementedError(f"{self.mm_policy} is not implemented.")

            a_mm = np.array(a_mm)
            # Create dummy Q-values
            q_mm = np.zeros((len(self.memory_systems.short), len(self.action_mm2str)))

        assert len(a_mm) == self.memory_systems.short.size

        for a_mm_, mem_short in zip(a_mm, self.memory_systems.short):
            manage_memory(self.memory_systems, self.action_mm2str[a_mm_], mem_short)

        (
            self.observations,
            reward,
            done,
            truncated,
            info,
        ) = self.env.step((answers, self.action_explore2str[a_explore.item()]))
        self.memory_systems.long.decay()
        self.num_semantic_decayed += 1
        done = done or truncated

        # 4. encode observations
        encode_all_observations(self.memory_systems, self.observations["room"])

        return (
            a_explore,
            q_explore,
            a_mm,
            q_mm,
            reward,
            intrinsic_explore_reward,
            answers,
            done,
        )

    def get_intrinsic_explore_reward(self, a_explore: str) -> float:
        r"""Get intrinsic actions.

        Args:
            a_explore: action in string

        Returns:
            intrinsic explore reward

        """
        assert isinstance(a_explore, str)
        assert not self.memory_systems.short.is_empty
        intrinsic_explore_actions = []
        for mem in self.memory_systems.get_working_memory():
            if (
                "room" in mem[0]
                and mem[1] in ["north", "east", "south", "west"]
                and "wall" not in mem[2]
                and "current_time" in mem[3]
            ):
                intrinsic_explore_actions.append(mem[1])

        assert len(intrinsic_explore_actions) > 0, "No intrinsic actions found."

        if a_explore in intrinsic_explore_actions:
            return self.intrinsic_explore_reward
        else:
            return 0

    def fill_replay_buffer(self) -> None:
        r"""Make the replay buffer full in the beginning with the uniformly-sampled
        actions. The filling continues until it reaches the warm start size.

        """
        self.replay_buffer = ReplayBuffer(self.replay_buffer_size, self.batch_size)
        done = True

        while len(self.replay_buffer) < self.warm_start:
            if done:
                self.reset()
                done = False
            else:
                state = deepcopy(self.memory_systems.get_working_memory().to_list())
                (
                    a_explore,
                    q_explore,
                    a_mm,
                    q_mm,
                    reward,
                    intrinsic_explore_reward,
                    answers,
                    done,
                ) = self.step(greedy=False)
                next_state = deepcopy(
                    self.memory_systems.get_working_memory().to_list()
                )

                if self.scale_reward:
                    reward /= self.env.unwrapped.num_questions_step
                    assert reward <= 1

                self.replay_buffer.store(
                    *[
                        state,
                        a_explore,
                        a_mm,
                        reward + intrinsic_explore_reward,
                        reward,
                        next_state,
                        done,
                    ]
                )

    def train(self) -> None:
        r"""Train the agent."""
        self.fill_replay_buffer()  # fill up the buffer till warm start size

        self.epsilons = []
        self.training_loss = {"total": [], "mm": [], "explore": []}
        self.scores = {"train": [], "val": [], "test": None}

        self.dqn.train()

        done = True
        score = 0
        self.iteration_idx = 0

        while True:
            if done:
                self.reset()
                done = False
            else:
                state = deepcopy(self.memory_systems.get_working_memory().to_list())
                (
                    a_explore,
                    q_explore,
                    a_mm,
                    q_mm,
                    reward,
                    intrinsic_explore_reward,
                    answers,
                    done,
                ) = self.step(greedy=False)
                score += reward
                next_state = deepcopy(
                    self.memory_systems.get_working_memory().to_list()
                )

                if self.scale_reward:
                    reward /= self.env.unwrapped.num_questions_step
                    assert reward <= 1

                self.replay_buffer.store(
                    *[
                        state,
                        a_explore,
                        a_mm,
                        reward + intrinsic_explore_reward,
                        reward,
                        next_state,
                        done,
                    ]
                )

                self.q_values["train"]["explore"].append(q_explore)
                self.q_values["train"]["mm"].append(q_mm)
                self.iteration_idx += 1

            if done:
                assert self.num_semantic_decayed == self.env_config["terminates_at"] + 1
                self.scores["train"].append(score)
                score = 0

                if (
                    self.iteration_idx
                    % (
                        self.validation_interval
                        * (self.env_config["terminates_at"] + 1)
                    )
                    == 0
                ):
                    with torch.no_grad():
                        self.validate()

            else:
                loss_mm, loss_explore, loss = update_model(
                    replay_buffer=self.replay_buffer,
                    optimizer=self.optimizer,
                    device=self.device,
                    dqn=self.dqn,
                    dqn_target=self.dqn_target,
                    ddqn=self.ddqn,
                    gamma=self.gamma,
                )

                self.training_loss["total"].append(loss)
                self.training_loss["mm"].append(loss_mm)
                self.training_loss["explore"].append(loss_explore)

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
                    target_hard_update(dqn=self.dqn, dqn_target=self.dqn_target)

                # plotting & show training results
                if (
                    self.iteration_idx == self.num_iterations
                    or self.iteration_idx % self.plotting_interval == 0
                ):
                    self.plot_results("all", save_fig=True)

                if self.iteration_idx >= self.num_iterations:
                    break

        with torch.no_grad():
            self.test()

        self.env.close()

    def validate_test_middle(self, val_or_test: str) -> tuple[list, list, list, list]:
        r"""A function shared by explore validation and test in the middle.

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

        for idx in range(self.num_samples_for_results[val_or_test]):
            done = True
            score = 0
            while True:
                if done:
                    self.reset()
                    done = False

                else:
                    state = deepcopy(self.memory_systems.get_working_memory().to_list())
                    (
                        a_explore,
                        q_explore,
                        a_mm,
                        q_mm,
                        reward,
                        intrinsic_explore_reward,
                        answers,
                        done,
                    ) = self.step(greedy=True)
                    score += reward

                    if idx == self.num_samples_for_results[val_or_test] - 1:
                        states_local.append(state)
                        q_values_local.append({"explore": q_explore, "mm": q_mm})
                        actions_local.append({"explore": a_explore, "mm": a_mm})
                        self.q_values[val_or_test]["explore"].append(q_explore)
                        self.q_values[val_or_test]["mm"].append(q_mm)

                if done:
                    assert (
                        self.num_semantic_decayed
                        == self.env_config["terminates_at"] + 1
                    )
                    break

            scores_local.append(score)

        return scores_local, states_local, q_values_local, actions_local

    def validate(self) -> None:
        r"""Validate the agent."""
        self.dqn.eval()
        scores_temp, states, q_values, actions = self.validate_test_middle("val")

        num_episodes = self.iteration_idx // (self.env_config["terminates_at"] + 1) - 1

        save_validation(
            scores_temp=scores_temp,
            scores=self.scores,
            default_root_dir=self.default_root_dir,
            num_episodes=num_episodes,
            validation_interval=self.validation_interval,
            val_file_names=self.val_file_names,
            dqn=self.dqn,
        )
        save_states_q_values_actions(
            states, q_values, actions, self.default_root_dir, "val", num_episodes
        )
        self.env.close()
        self.dqn.train()

    def test(self, checkpoint: str | None = None) -> None:
        r"""Test the agent.

        Args:
            checkpoint: The checkpoint to override.

        """
        self.dqn.eval()

        self.env_config["seed"] = self.test_seed
        self.env = gym.make(self.env_str, **self.env_config)

        assert len(self.val_file_names) == 1, f"{len(self.val_file_names)} should be 1"

        self.dqn.load_state_dict(torch.load(self.val_file_names[0]))

        if checkpoint is not None:
            self.dqn.load_state_dict(torch.load(checkpoint))

        scores, states, q_values, actions = self.validate_test_middle("test")
        self.scores["test"] = scores

        save_final_results(
            self.scores,
            self.training_loss,
            self.default_root_dir,
            self.q_values,
            self,
        )
        save_states_q_values_actions(
            states, q_values, actions, self.default_root_dir, "test"
        )

        self.plot_results("all", save_fig=True)
        self.env.close()
        self.dqn.train()

    def plot_results(self, to_plot: str = "all", save_fig: bool = False) -> None:
        r"""Plot things for DQN training.

        Args:
            to_plot: what to plot:
                training_td_loss
                epsilons
                training_score
                validation_score
                test_score
                q_values_train
                q_values_val
                q_values_test

        """
        plot_results(
            self.scores,
            self.training_loss,
            self.epsilons,
            self.q_values,
            self.iteration_idx,
            self.num_iterations,
            self.env.unwrapped.total_maximum_episode_rewards,
            self.default_root_dir,
            self.action_mm2str,
            self.action_explore2str,
            to_plot,
            save_fig,
        )
