"""Learn memory management, exploration, and qa policies."""

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
from humemai_research.utils import is_running_notebook, read_yaml, write_yaml

from ..policy import (
    answer_question,
    encode_all_observations,
    explore,
    manage_memory,
    manage_short,
)
from ..utils import is_dict_subset
from .nn import GNN
from .utils import (
    ReplayBuffer,
    plot_results,
    save_final_results,
    save_validation,
    select_action,
    update_epsilon,
    update_model,
)


class DQNAgent:
    r"""DQN Agent interacting with environment.

    This is an upgrade from https://github.com/humemai/agent-room-env-v2-gnn.
    It aims to learn the question-answering function using a contextual bandit
    algorithm.

    Based on https://github.com/Curt-Park/rainbow-is-all-you-need/
    """

    def __init__(
        self,
        env_str: str = "room_env:RoomEnv-v2",
        num_iterations: int = 20000,
        replay_buffer_size: int = 20000,
        warm_start: int = 32,
        batch_size: int = 32,
        epsilon_decay_until: float = 20000,
        max_epsilon: float = 1.0,
        min_epsilon: float = 0.1,
        learning_rate: int = 0.001,
        capacity: dict = {
            "long": 96,
            "short": 15,
        },
        semantic_decay_factor: float = 0.8,
        dqn_params: dict = {
            "gcn_layer_params": {
                "type": "StarE",
                "embedding_dim": 64,
                "num_layers": 2,
                "gcn_drop": 0.1,
                "triple_qual_weight": 0.8,
            },
            "relu_between_gcn_layers": True,
            "dropout_between_gcn_layers": False,
            "mlp_params": {"num_hidden_layers": 2, "dueling_dqn": True},
            "use_raw_embeddings_for_qa": False,
        },
        num_samples_for_results: dict = {"val": 5, "test": 10},
        plotting_interval: int = 50,
        train_seed: int = 5,
        test_seed: int = 0,
        device: Literal["cpu", "cuda"] = "cpu",
        env_config: dict = {
            "question_prob": 1.0,
            "terminates_at": 99,
            "randomize_observations": "all",
            "room_size": "xl-different-prob",
            "rewards": {"correct": 1, "wrong": 0, "partial": 0},
            "make_everything_static": False,
            "num_total_questions": 1000,
            "question_interval": 1,
            "include_walls_in_observations": True,
        },
        ddqn: bool = True,
        default_root_dir: str = "./training-results/",
        qa_function: Literal["latest_strongest", "bandit", "llm"] = "bandit",
        qa_entities: Literal["all", "room"] = "room",
        pretrained_path: str | None = None,
        llm_params: dict | None = None,
    ) -> None:
        r"""Initialization.

        Args:
            env_str: environment string. This has to be "room_env:RoomEnv-v2"
            num_iterations: number of iterations to train
            replay_buffer_size: size of replay buffer
            warm_start: number of steps to fill the replay buffer, before training
            batch_size: This is the amount of samples sampled from the replay buffer.
            epsilon_decay_until: until which iteration to decay epsilon
            max_epsilon: maximum epsilon
            min_epsilon: minimum epsilon
            learning_rate: learning rate for the optimizer
            capacity: The capacity of each Human-Like Memory
            semantic_decay_factor: decay factor for the semantic memory system
            dqn_params: parameters for the DQN
            num_samples_for_results: The number of samples to validate / test the agent.
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
            ddqn: whether to use double DQN
            default_root_dir: default root directory to save results
            qa_function: question answering policy Choose one of "latest_strongest",
                "bandit", "llm".
            qa_entities: entities to consider for QA. Choose one of "all", "room".
            pretrained_path: path to pretrained model
            llm_params: parameters for the LLM

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
        self.num_samples_for_results = num_samples_for_results
        self.capacity = capacity
        self.semantic_decay_factor = semantic_decay_factor
        self.env = gym.make(self.env_str, **self.env_config)

        self.device = torch.device(device)
        print(f"Running on {self.device}")

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
        self.learning_rate = learning_rate
        self.warm_start = warm_start
        assert self.batch_size <= self.warm_start <= self.replay_buffer_size

        self.qa_function = qa_function
        self.qa_entities = qa_entities
        self.pretrained_path = pretrained_path
        self.llm_params = llm_params

        if self.qa_function == "llm":
            self._setup_llm(**self.llm_params)

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

        if self.pretrained_path is not None:
            pretrained_params = read_yaml(
                os.path.join(self.pretrained_path, "train.yaml")
            )["dqn_params"]

            assert is_dict_subset(
                pretrained_params, dqn_params
            ), "Pretrained params mismatch."

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
        self.dqn_params["pretrained_path"] = self.pretrained_path

        if self.qa_entities == "all":

            self.dqn_params["qa_entities"] = [
                e for entities in self.env.unwrapped.entities.values() for e in entities
            ]

        else:
            assert self.qa_entities == "room"

            self.dqn_params["qa_entities"] = [
                e
                for entities in self.env.unwrapped.entities.values()
                for e in entities
                if "room" in e
            ]

        self.dqn = GNN(**self.dqn_params)

        # optimizer
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.dqn.parameters()),
            lr=self.learning_rate,
        )

        self._save_number_of_parameters()

    def _setup_llm(self, **kwargs) -> None:
        r"""Setup the LLM model."""
        from ..llm import Llm

        self.llm = Llm(**kwargs)

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
                "mlp_qa": sum(p.numel() for p in self.dqn.mlp_qa.parameters()),
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

        self.num_semantic_decayed = 0

    def reset(self) -> None:
        """Reset the env and the memory systems. observations are encoded."""

        self.init_memory_systems()
        self.observations, info = self.env.reset()
        # 0. encode observations
        encode_all_observations(self.memory_systems, self.observations["room"])

    def step(self, greedy: bool) -> tuple:
        r"""Step of the algorithm. This is the only step that interacts with the
        environment.

        Args:
            greedy: whether to use greedy policy

        Returns:
            q_qa, reward, done, question

        """
        assert not self.memory_systems.short.is_empty, "encode all observations first"
        # 1. explore
        a_explore = explore(
            self.memory_systems,
            "neural",
            self.action_explore2int,
            nn=self.dqn,
        )

        # 2. question answering
        que = deepcopy(self.observations["questions"])
        if self.qa_function == "bandit":
            answers, a_qa, q_qa = select_action(
                state=self.memory_systems.get_working_memory().to_list(),
                greedy=greedy,
                dqn=self.dqn,
                epsilon=self.epsilon,
                questions=que,
            )

        else:
            answers = [
                answer_question(
                    self.memory_systems,
                    self.qa_function,
                    q,
                    self.llm if self.qa_function == "llm" else None,
                )
                for q in que
            ]
            # Create dummy Q-values
            q_qa = [np.zeros(1) for answer in answers]
            a_qa = None

        # 3. manage memory
        a_mm = manage_short(
            self.memory_systems,
            "neural",
            self.action_mm2int,
            nn=self.dqn,
        )

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
            answers,
            a_qa,
            q_qa,
            reward,
            done,
            que,
        )

    def fill_replay_buffer(self) -> None:
        r"""Make the replay buffer full in the beginning with the uniformly-sampled
        actions. The filling continues until it reaches the warm start size.

        """
        self.replay_buffer = ReplayBuffer(
            self.replay_buffer_size,
            self.batch_size,
            self.env.unwrapped.num_questions_step,
        )

        done = True

        while len(self.replay_buffer) < self.warm_start:
            if done:
                self.reset()
                done = False
            else:
                state = deepcopy(self.memory_systems.get_working_memory().to_list())
                (
                    answers,
                    action,
                    q_value,
                    reward,
                    done,
                    question,
                ) = self.step(greedy=False)

                self.replay_buffer.store(*[state, action, reward, question])

    def train(self) -> None:
        r"""Train the agent."""
        self.fill_replay_buffer()  # fill up the buffer till warm start size

        self.epsilons = []
        self.training_loss = []
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
                (answers, action, q_value, reward, done, question) = self.step(
                    greedy=False
                )
                score += sum(reward)

                self.replay_buffer.store(*[state, action, reward, question])

                self.iteration_idx += 1

            if done:
                assert self.num_semantic_decayed == self.env_config["terminates_at"] + 1
                self.scores["train"].append(score)
                score = 0

                with torch.no_grad():
                    self.validate()

            else:
                if self.qa_function != "llm":
                    loss = update_model(
                        replay_buffer=self.replay_buffer,
                        optimizer=self.optimizer,
                        device=self.device,
                        dqn=self.dqn,
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

                    # plotting & show training results
                    if (
                        self.iteration_idx == self.num_iterations
                        or self.iteration_idx % self.plotting_interval == 0
                    ):

                        self.plot_results()

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

        """
        scores_local = []

        for idx in range(self.num_samples_for_results[val_or_test]):
            done = True
            score = 0
            while True:
                if done:
                    self.reset()
                    done = False

                else:
                    state = deepcopy(self.memory_systems.get_working_memory().to_list())
                    (answers, action, q_value, reward, done, question) = self.step(
                        greedy=True
                    )
                    score += sum(reward)

                if done:
                    assert (
                        self.num_semantic_decayed
                        == self.env_config["terminates_at"] + 1
                    )
                    break

            scores_local.append(score)

        return scores_local

    def validate(self) -> None:
        r"""Validate the agent."""
        self.dqn.eval()
        scores_temp = self.validate_test_middle("val")
        self.scores["val"].append(scores_temp)

        num_episodes = self.iteration_idx // (self.env_config["terminates_at"] + 1) - 1

        save_validation(
            scores_temp=scores_temp,
            default_root_dir=self.default_root_dir,
            num_episodes=num_episodes,
            val_file_names=self.val_file_names,
            dqn=self.dqn,
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

        scores = self.validate_test_middle("test")
        self.scores["test"] = scores

        save_final_results(
            self.scores,
            self.training_loss,
            self.default_root_dir,
        )

        if self.qa_function != "llm":
            self.plot_results()
        self.env.close()
        self.dqn.train()

    def plot_results(self) -> None:
        r"""Plot things for DQN training."""
        plot_results(
            self.scores,
            self.training_loss,
            self.epsilons,
            self.iteration_idx,
            self.num_iterations,
            self.env.unwrapped.total_maximum_episode_rewards,
            self.default_root_dir,
        )
