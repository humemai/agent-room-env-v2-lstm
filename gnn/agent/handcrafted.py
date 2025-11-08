"""Agent that uses a GNN for its DQN, for the RoomEnv2 environment."""

import datetime
import os
import shutil
from copy import deepcopy
from typing import Literal

import gymnasium as gym
import numpy as np
from humemai_research.memory import LongMemory, MemorySystems, ShortMemory
from humemai_research.utils import write_yaml

from .policy import (answer_question, encode_all_observations, explore,
                     manage_memory)


class HandcraftedAgent:
    """Handcrafted agent interacting with environment.

    This agent explores the roooms, i.e., KGs. The exploration can be uniform-random,
    or just avoiding walls.

    """

    def __init__(
        self,
        env_str: str = "room_env:RoomEnv-v2",
        env_config: dict = {
            "question_prob": 1.0,
            "seed": 42,
            "terminates_at": 99,
            "randomize_observations": "objects",
            "make_everything_static": False,
            "rewards": {"correct": 1, "wrong": 0, "partial": 0},
            "num_total_questions": 100,
            "question_interval": 1,
            "room_size": "xxs",
        },
        mm_policy: Literal["random", "episodic", "semantic", "handcrafted"] = "random",
        qa_function: Literal[
            "latest_strongest", "latest", "strongest", "random"
        ] = "latest_strongest",
        explore_policy: Literal["random", "avoid_walls"] = "avoid_walls",
        num_samples_for_results: int = 10,
        capacity: dict = {
            "long": 12,
            "short": 15,
        },
        pretrain_semantic: Literal[False, "include_walls", "exclude_walls"] = False,
        semantic_decay_factor: float = 1.0,
        default_root_dir: str = "./training-results/",
    ) -> None:
        """Initialize the agent.

        Args:
            env_str: This has to be "room_env:RoomEnv-v2"
            env_config: The configuration of the environment.
            mm_policy: memory management policy. Choose one of "random", "episodic",
                "semantic", or "handcrafted"
            qa_function: The question answering policy. Choose one of
                "episodic_semantic", "episodic", "semantic", or "random"
            explore_policy: The room exploration policy. Choose one of "random",
                or "avoid_walls"
            num_samples_for_results: The number of samples to validate / test the agent.
            capacity: The capacity of each human-like memory systems.
            pretrain_semantic: Whether or not to pretrain the semantic memory system.
            semantic_decay_factor: The decay factor for the semantic memory system.
            default_root_dir: default root directory to store the results.

        """
        params_to_save = deepcopy(locals())
        del params_to_save["self"]

        self.env_str = env_str
        self.env_config = env_config
        self.mm_policy = mm_policy
        assert self.mm_policy in [
            "random",
            "episodic",
            "semantic",
            "handcrafted",
        ]
        self.qa_function = qa_function
        assert self.qa_function in [
            "latest_strongest",
            "latest",
            "strongest",
            "random",
        ]
        self.explore_policy = explore_policy
        assert self.explore_policy in [
            "random",
            "avoid_walls",
        ]
        self.num_samples_for_results = num_samples_for_results
        self.capacity = capacity
        self.pretrain_semantic = pretrain_semantic
        self.semantic_decay_factor = semantic_decay_factor
        self.env = gym.make(self.env_str, **self.env_config)
        self.default_root_dir = os.path.join(
            default_root_dir, str(datetime.datetime.now())
        )
        self._create_directory(params_to_save)

    def _create_directory(self, params_to_save: dict) -> None:
        """Create the directory to store the results."""
        os.makedirs(self.default_root_dir, exist_ok=True)
        write_yaml(params_to_save, os.path.join(self.default_root_dir, "train.yaml"))

    def remove_results_from_disk(self) -> None:
        """Remove the results from the disk."""
        shutil.rmtree(self.default_root_dir)

    def init_memory_systems(self, reset_semantic_decay: bool = True) -> None:
        """Initialize the agent's memory systems. This has nothing to do with the
        replay buffer.

        Args:
            reset_semantic_decay: whether to reset the semantic memory system's decay

        """
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

        if reset_semantic_decay:
            self.num_semantic_decayed = 0

    def test(self):
        """Test the agent. There is no training for this agent, since it is
        handcrafted."""
        self.scores = []

        for _ in range(self.num_samples_for_results):
            score = 0
            env_started = False
            action_pair = ([], None)
            done = False
            self.init_memory_systems(reset_semantic_decay=True)

            while not done:
                if env_started:
                    (
                        observations,
                        reward,
                        done,
                        truncated,
                        info,
                    ) = self.env.step(action_pair)
                    self.memory_systems.long.decay()
                    self.num_semantic_decayed += 1

                    score += reward

                    if done:
                        assert (
                            self.num_semantic_decayed
                            == self.env_config["terminates_at"] + 1
                        )
                        break

                else:
                    observations, info = self.env.reset()
                    env_started = True

                # 0. Encode the observations as short-term memory
                encode_all_observations(self.memory_systems, observations["room"])

                # 1. explore the room
                action_explore = explore(self.memory_systems, self.explore_policy)

                # 2. Answer the questions
                answers = [
                    str(
                        answer_question(
                            self.memory_systems,
                            self.qa_function,
                            question,
                        )
                    )
                    for question in observations["questions"]
                ]

                # 3. Manage the memory
                for mem_short in self.memory_systems.short:
                    manage_memory(self.memory_systems, self.mm_policy, mem_short)

                action_pair = (answers, action_explore)
            self.scores.append(score)

        self.scores = {
            "test_score": {
                "mean": round(np.mean(self.scores).item(), 2),
                "std": round(np.std(self.scores).item(), 2),
            }
        }
        write_yaml(self.scores, os.path.join(self.default_root_dir, "results.yaml"))
        write_yaml(
            self.memory_systems.get_working_memory().to_list(),
            os.path.join(self.default_root_dir, "last_memory_state.yaml"),
        )
