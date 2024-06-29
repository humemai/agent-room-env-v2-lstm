import unittest

import numpy as np

from agent.dqn.utils import ReplayBuffer, update_epsilon


class TestReplayBuffer(unittest.TestCase):

    def setUp(self):
        self.observation_type = "dict"
        self.size = 100
        self.batch_size = 32
        self.buffer = ReplayBuffer(
            self.observation_type, size=self.size, batch_size=self.batch_size
        )

    def test_store_and_sample_batch(self):
        obs_dim = 4
        act_dim = 1

        # Store some data
        for _ in range(self.size):
            obs = {"state": [np.random.rand(obs_dim).tolist()]}
            act = np.random.rand(act_dim)
            rew = np.random.rand()
            next_obs = {"state": [np.random.rand(obs_dim).tolist()]}
            done = False
            self.buffer.store(obs, act, rew, next_obs, done)

        # Sample a batch
        batch = self.buffer.sample_batch()

        # Check the batch size
        self.assertEqual(len(batch["obs"]), self.batch_size)
        self.assertEqual(len(batch["next_obs"]), self.batch_size)
        self.assertEqual(len(batch["acts"]), self.batch_size)
        self.assertEqual(len(batch["rews"]), self.batch_size)
        self.assertEqual(len(batch["done"]), self.batch_size)

        # Check the content types and shapes
        self.assertTrue(isinstance(batch["obs"][0], dict))
        self.assertTrue(isinstance(batch["next_obs"][0], dict))
        self.assertEqual(
            batch["acts"][0].shape, ()
        )  # Check if it's a scalar (numpy.float32)
        self.assertIsInstance(batch["rews"][0], np.float32)
        self.assertIsInstance(batch["done"][0], np.float32)

    def test_length(self):
        # Test length of the buffer
        self.assertEqual(len(self.buffer), 0)
        for i in range(self.size):
            obs = {"state": [np.random.rand(4).tolist()]}
            act = np.random.rand(1)
            rew = np.random.rand()
            next_obs = {"state": [np.random.rand(4).tolist()]}
            done = False
            self.buffer.store(obs, act, rew, next_obs, done)
            self.assertEqual(len(self.buffer), i + 1)
        self.assertEqual(len(self.buffer), self.size)


class TestUpdateEpsilon(unittest.TestCase):

    def test_update_epsilon(self):
        # Define test parameters
        epsilon = 0.5
        max_epsilon = 1.0
        min_epsilon = 0.1
        epsilon_decay_until = 100

        # Calculate expected epsilon after update
        expected_epsilon = max(
            min_epsilon, epsilon - (max_epsilon - min_epsilon) / epsilon_decay_until
        )

        # Call the function under test
        updated_epsilon = update_epsilon(
            epsilon, max_epsilon, min_epsilon, epsilon_decay_until
        )

        # Assert that the updated epsilon matches the expected epsilon
        self.assertAlmostEqual(
            updated_epsilon, expected_epsilon, places=7
        )  # Adjust places as needed

    # Add more test cases as needed to cover edge cases and boundary conditions
