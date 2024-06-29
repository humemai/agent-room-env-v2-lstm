import unittest

import torch
from torch import nn

from agent.dqn.nn import MLP


class TestMLP(unittest.TestCase):
    def setUp(self):
        self.device = "cpu"
        self.mlp_config = {
            "n_actions": 5,
            "hidden_size": 64,
            "device": self.device,
            "num_hidden_layers": 2,
            "dueling_dqn": True,
        }

    def test_mlp_initialization(self):
        mlp = MLP(**self.mlp_config)
        self.assertEqual(mlp.n_actions, self.mlp_config["n_actions"])
        self.assertEqual(mlp.hidden_size, self.mlp_config["hidden_size"])
        self.assertEqual(mlp.num_hidden_layers, self.mlp_config["num_hidden_layers"])
        self.assertEqual(mlp.dueling_dqn, self.mlp_config["dueling_dqn"])
        self.assertIsInstance(mlp.advantage_layer, nn.Sequential)
        if mlp.dueling_dqn:
            self.assertIsInstance(mlp.value_layer, nn.Sequential)

    def test_mlp_forward(self):
        batch_size = 2
        input_size = 64
        n_actions = 5

        mlp = MLP(n_actions=n_actions, hidden_size=input_size, device=self.device)
        mlp.to(self.device)

        input_tensor = torch.randn(batch_size, input_size).to(self.device)
        output = mlp(input_tensor)

        self.assertEqual(output.shape, torch.Size([batch_size, n_actions]))
        self.assertTrue(torch.all(torch.isfinite(output)).item())

    def test_mlp_layer_configuration(self):
        mlp = MLP(**self.mlp_config)

        expected_advantage_layers = [
            nn.Linear(self.mlp_config["hidden_size"], self.mlp_config["hidden_size"]),
            nn.ReLU(),
            nn.Linear(self.mlp_config["hidden_size"], self.mlp_config["hidden_size"]),
            nn.ReLU(),
            nn.Linear(self.mlp_config["hidden_size"], self.mlp_config["n_actions"]),
        ]
        expected_value_layers = [
            nn.Linear(self.mlp_config["hidden_size"], self.mlp_config["hidden_size"]),
            nn.ReLU(),
            nn.Linear(self.mlp_config["hidden_size"], self.mlp_config["hidden_size"]),
            nn.ReLU(),
            nn.Linear(self.mlp_config["hidden_size"], 1),
        ]

        actual_advantage_layers = list(mlp.advantage_layer.children())
        self.assertEqual(len(actual_advantage_layers), len(expected_advantage_layers))
        for actual, expected in zip(actual_advantage_layers, expected_advantage_layers):
            self.assertIsInstance(actual, type(expected))
            if isinstance(actual, nn.Linear):
                self.assertEqual(actual.in_features, expected.in_features)
                self.assertEqual(actual.out_features, expected.out_features)

        if mlp.dueling_dqn:
            actual_value_layers = list(mlp.value_layer.children())
            self.assertEqual(len(actual_value_layers), len(expected_value_layers))
            for actual, expected in zip(actual_value_layers, expected_value_layers):
                self.assertIsInstance(actual, type(expected))
                if isinstance(actual, nn.Linear):
                    self.assertEqual(actual.in_features, expected.in_features)
                    self.assertEqual(actual.out_features, expected.out_features)
