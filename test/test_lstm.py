import unittest

import numpy as np
import torch

from agent.dqn.nn import LSTM


class TestLSTM(unittest.TestCase):

    def setUp(self):
        # Initialize parameters for testing
        self.capacity = {"episodic": 3, "semantic": 2, "short": 1}
        self.entities_dict = {
            "category1": ["entity1", "entity2"],
            "category2": ["entity3"],
        }
        self.entities_list = ["entity1", "entity2", "entity3"]
        self.relations = ["relation1", "relation2"]
        self.hidden_size = 64
        self.num_layers = 2
        self.embedding_dim = 32
        self.device = "cpu"
        self.max_timesteps = 10
        self.max_strength = 5

        # Initialize the LSTM model for testing
        self.lstm_dict_entities = LSTM(
            capacity=self.capacity,
            entities=self.entities_dict,
            relations=self.relations,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            embedding_dim=self.embedding_dim,
            device=self.device,
            max_timesteps=self.max_timesteps,
            max_strength=self.max_strength,
        )

        self.lstm_list_entities = LSTM(
            capacity=self.capacity,
            entities=self.entities_list,
            relations=self.relations,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            embedding_dim=self.embedding_dim,
            device=self.device,
            max_timesteps=self.max_timesteps,
            max_strength=self.max_strength,
        )

    def test_create_embeddings_dict_entities(self):
        # Test create_embeddings method with dictionary entities
        expected_word_count = (
            1
            + len(self.entities_dict["category1"])
            + len(self.entities_dict["category2"])
            + len(self.relations)
            + 3
        )
        self.assertEqual(len(self.lstm_dict_entities.word2idx), expected_word_count)

    def test_create_embeddings_list_entities(self):
        # Test create_embeddings method with list entities
        expected_word_count = 1 + len(self.entities_list) + len(self.relations) + 3
        self.assertEqual(len(self.lstm_list_entities.word2idx), expected_word_count)

    def test_make_embedding_short(self):
        # Test make_embedding method for short memory type
        mem = ["entity1", "relation1", "entity2", 5]
        embedding = self.lstm_dict_entities.make_embedding(mem, "short")
        self.assertIsInstance(embedding, torch.Tensor)
        self.assertEqual(embedding.shape, torch.Size([self.embedding_dim]))

    def test_make_embedding_episodic(self):
        # Test make_embedding method for episodic memory type
        mem = ["entity3", "relation2", "entity1", 2]
        embedding = self.lstm_dict_entities.make_embedding(mem, "episodic")
        self.assertIsInstance(embedding, torch.Tensor)
        self.assertEqual(embedding.shape, torch.Size([self.embedding_dim]))

    def test_make_embedding_semantic(self):
        # Test make_embedding method for semantic memory type
        mem = ["entity2", "relation1", "entity3", 7]
        embedding = self.lstm_dict_entities.make_embedding(mem, "semantic")
        self.assertIsInstance(embedding, torch.Tensor)
        self.assertEqual(embedding.shape, torch.Size([self.embedding_dim]))

    def test_create_batch(self):
        # Test create_batch method
        batch_size = 2
        x = [
            [
                ["entity1", "relation1", "entity2", 5],
                ["entity2", "relation2", "entity3", 3],
            ],
            [["entity3", "relation1", "entity1", 2]],
        ]
        batch_embeddings = self.lstm_dict_entities.create_batch(x, "episodic")
        self.assertIsInstance(batch_embeddings, torch.Tensor)
        self.assertEqual(
            batch_embeddings.shape,
            torch.Size([batch_size, self.capacity["episodic"], self.embedding_dim]),
        )

    def test_forward(self):
        # Test forward method
        batch_size = 2
        memory_types = ["short", "episodic", "semantic"]

        # Example structured memories
        x = np.empty(batch_size, dtype=object)

        x[0] = {
            "short": [["entity1", "relation1", "entity2", 3]],
            "episodic": [
                ["entity2", "relation1", "entity3", 5],
                ["entity2", "relation1", "entity2", 4],
                ["entity2", "relation1", "entity3", 6],
            ],
            "semantic": [
                ["entity2", "relation1", "entity2", 1],
            ],
        }
        x[1] = {
            "short": [["entity3", "relation2", "entity1", 2]],
            "episodic": [
                ["entity2", "relation1", "entity3", 6],
            ],
            "semantic": [
                ["entity2", "relation1", "entity2", 1],
                ["entity1", "relation2", "entity1", 2],
            ],
        }

        output = self.lstm_dict_entities.forward(x, memory_types)[0]
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, torch.Size([batch_size, self.hidden_size]))
