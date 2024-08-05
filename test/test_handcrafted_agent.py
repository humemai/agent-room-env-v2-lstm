import logging

logger = logging.getLogger()
logger.disabled = True

import random
import unittest

from tqdm.auto import tqdm

from agent import HandcraftedAgent


class HandcraftedAgentTest(unittest.TestCase):
    def test_all_agents(self) -> None:
        terminates_at = 9
        for idx in range(10):
            rng = random.Random(idx)
            capacity = {
                "episodic": random.randint(1, 4),
                "semantic": random.randint(1, 4),
                "episodic_agent": random.randint(0, 4),
                "semantic_map": random.randint(0, 4),
                "short": 1,
            }

            config = {
                "question_prob": 1.0,
                "seed": 42,
                "terminates_at": terminates_at,
                "room_size": rng.choice(["xxs", "xs", "s", "m", "l"]),
                "randomize_observations": "all",
                "make_everything_static": False,
                "rewards": {"correct": 1, "wrong": -1, "partial": 0},
                "num_total_questions": 100,
                "question_interval": 5,
            }

            results = {}
            for mm_policy in ["random", "generalize", "handcrafted"]:
                for qa_function in ["random", "episodic_semantic"]:
                    for explore_policy in ["random", "avoid_walls"]:
                        for pretrain_semantic in [False, "include_walls"]:
                            for semantic_decay_factor in [0.5, 1.0]:
                                key = (mm_policy, qa_function, explore_policy)
                                if key not in results:
                                    results[key] = []

                                for seed in tqdm([42]):
                                    config["seed"] = seed

                                    agent = HandcraftedAgent(
                                        env_str="room_env:RoomEnv-v2",
                                        env_config=config,
                                        mm_policy=mm_policy,
                                        qa_function=qa_function,
                                        explore_policy=explore_policy,
                                        num_samples_for_results=3,
                                        capacity=capacity,
                                        pretrain_semantic=pretrain_semantic,
                                        semantic_decay_factor=semantic_decay_factor,
                                    )
                                    agent.test()
                                    if (
                                        hasattr(agent.memory_systems, "semantic")
                                        and agent.memory_systems.semantic.capacity > 0
                                    ):
                                        self.assertEqual(
                                            agent.num_semantic_decayed,
                                            terminates_at + 1,
                                        )
                                    else:
                                        self.assertEqual(agent.num_semantic_decayed, 0)

                                    agent.remove_results_from_disk()
                                    print(agent.scores)
                                    results[key].append(agent.scores)
