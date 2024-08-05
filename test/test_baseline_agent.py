import logging

logger = logging.getLogger()
logger.disabled = True

import matplotlib

matplotlib.use("Agg")

import unittest

from agent import DQNLSTMMLPBaselineAgent


class DQNLSTMMLPBaselineAgentTest(unittest.TestCase):
    def test_agent(self) -> None:
        hparams = {
            "env_str": "room_env:RoomEnv-v2",
            "num_iterations": 10,
            "replay_buffer_size": 10,
            "validation_starts_at": 5,
            "warm_start": 5,
            "batch_size": 2,
            "target_update_interval": 10,
            "epsilon_decay_until": 10,
            "max_epsilon": 1.0,
            "min_epsilon": 0.1,
            "gamma": 0.9,
            "history_block_size": 2,
            "lstm_params": {
                "hidden_size": 2,
                "num_layers": 1,
                "embedding_dim": 2,
                "bidirectional": False,
            },
            "mlp_params": {
                "hidden_size": 2,
                "num_hidden_layers": 1,
                "dueling_dqn": True,
            },
            "num_samples_for_results": {"val": 10, "test": 10},
            "plotting_interval": 20,
            "train_seed": 6,
            "test_seed": 1,
            "device": "cpu",
            "env_config": {
                "question_prob": 1.0,
                "terminates_at": 4,
                "randomize_observations": "objects",
                "room_size": "s",
                "rewards": {"correct": 1, "wrong": 0, "partial": 0},
                "make_everything_static": False,
                "num_total_questions": 10,
                "question_interval": 5,
                "include_walls_in_observations": True,
            },
            "default_root_dir": "./training-results/",
            "run_handcrafted_baselines": True,
        }

        agent = DQNLSTMMLPBaselineAgent(**hparams)
        agent.train()
        agent.remove_results_from_disk()
