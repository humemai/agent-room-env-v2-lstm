import logging

logger = logging.getLogger()
logger.disabled = True

import matplotlib

matplotlib.use("Agg")

import unittest

from tqdm.auto import tqdm

from agent import DQNAgent


class DQNAgentTest(unittest.TestCase):
    def test_agent(self) -> None:
        params_all = []
        room_size = "s"
        for capacity_max in [2, 4]:
            for semantic_decay_factor in [0.5, 1.0]:
                num_iterations = 10 * 2
                batch_size = 2
                question_interval = 2
                terminates_at = 9

                capacity_h = {
                    "episodic": capacity_max // 2,
                    "semantic": capacity_max // 2,
                    "short": 1,
                }
                capacity_e = {
                    "episodic": capacity_max,
                    "semantic": 0,
                    "short": 1,
                }
                capacity_s = {
                    "episodic": 0,
                    "semantic": capacity_max,
                    "short": 1,
                }

                for capacity in [capacity_h, capacity_e, capacity_s]:
                    params = {
                        "env_str": "room_env:RoomEnv-v2",
                        "num_iterations": num_iterations,
                        "replay_buffer_size": num_iterations,
                        "validation_starts_at": num_iterations // 2,
                        "warm_start": num_iterations // 10,
                        "batch_size": batch_size,
                        "target_update_interval": 10,
                        "epsilon_decay_until": num_iterations,
                        "max_epsilon": 1.0,
                        "min_epsilon": 0.1,
                        "gamma": {"mm": 0.99, "explore": 0.9},
                        "capacity": capacity,
                        "pretrain_semantic": False,
                        "semantic_decay_factor": semantic_decay_factor,
                        "lstm_params": {
                            "hidden_size": 2,
                            "num_layers": 1,
                            "embedding_dim": 2,
                            "bidirectional": False,
                            "max_timesteps": 10,
                            "max_strength": 10,
                        },
                        "mlp_params": {
                            "hidden_size": 2,
                            "num_hidden_layers": 1,
                            "dueling_dqn": True,
                        },
                        "num_samples_for_results": {"val": 10, "test": 10},
                        "plotting_interval": 20,
                        "train_seed": 5,
                        "test_seed": 0,
                        "device": "cpu",
                        "qa_function": "episodic_semantic",
                        "explore_policy_heuristic": "avoid_walls",
                        "env_config": {
                            "question_prob": 1.0,
                            "terminates_at": terminates_at,
                            "randomize_observations": "objects",
                            "room_size": room_size,
                            "rewards": {"correct": 1, "wrong": 0, "partial": 0},
                            "make_everything_static": False,
                            "num_total_questions": 100,
                            "question_interval": question_interval,
                            "include_walls_in_observations": True,
                        },
                        "ddqn": True,
                        "default_root_dir": "./training-results/",
                    }
                    params_all.append(params)

        for params in tqdm(params_all):
            agent = DQNAgent(**params)
            agent.train()
            if (
                hasattr(agent.memory_systems, "semantic")
                and agent.memory_systems.semantic.capacity > 0
            ):
                self.assertEqual(agent.num_semantic_decayed, terminates_at + 1)
            else:
                self.assertEqual(agent.num_semantic_decayed, 0)
            agent.remove_results_from_disk()
