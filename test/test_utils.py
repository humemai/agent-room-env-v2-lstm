import os
import tempfile
import unittest

from agent.utils import (read_json, read_pickle, read_yaml, write_json,
                         write_pickle, write_yaml)


class TestUtils(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.test_json_file = os.path.join(self.test_dir.name, "test.json")
        self.test_yaml_file = os.path.join(self.test_dir.name, "test.yaml")
        self.test_pickle_file = os.path.join(self.test_dir.name, "test.pkl")
        self.seed = 42

    def tearDown(self):
        self.test_dir.cleanup()

    def test_read_write_json(self):
        data = {"key1": "value1", "key2": [1, 2, 3]}
        write_json(data, self.test_json_file)
        loaded_data = read_json(self.test_json_file)
        self.assertEqual(data, loaded_data)

    def test_read_write_yaml(self):
        data = {"key1": "value1", "key2": [1, 2, 3]}
        write_yaml(data, self.test_yaml_file)
        loaded_data = read_yaml(self.test_yaml_file)
        self.assertEqual(data, loaded_data)

    def test_read_write_pickle(self):
        data = {"key1": "value1", "key2": [1, 2, 3]}
        write_pickle(data, self.test_pickle_file)
        loaded_data = read_pickle(self.test_pickle_file)
        self.assertEqual(data, loaded_data)
