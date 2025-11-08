import unittest
from collections import Counter
from copy import deepcopy

from humemai_research.memory import LongMemory, MemorySystems, ShortMemory

from agent.policy import (answer_question, encode_all_observations,
                          encode_observation, explore, find_agent_location,
                          manage_memory)


class TestFindAgentLocation(unittest.TestCase):

    def setUp(self):
        """Set up the test case with predefined memory systems."""
        self.short_memory = ShortMemory(capacity=10)
        self.long_memory = LongMemory(capacity=10)
        self.memory_systems = MemorySystems(
            short=self.short_memory, long=self.long_memory
        )

        # Add memories to long-term memory
        self.long_memory.add(
            ["agent", "atlocation", "room3", {"timestamp": [2, 3], "strength": 3}]
        )
        self.long_memory.add(
            ["agent", "atlocation", "room4", {"timestamp": [1], "strength": 4}]
        )
        self.long_memory.add(["agent", "atlocation", "room2", {"strength": 5}])
        self.long_memory.add(
            ["agent", "atlocation", "room5", {"timestamp": [3, 4], "strength": 2}]
        )

        # Add memories to short-term memory
        self.short_memory.add(["agent", "atlocation", "room1", {"current_time": 5}])
        self.short_memory.add(["room1", "south", "wall", {"current_time": 5}])
        self.short_memory.add(["laptop", "atlocation", "room1", {"current_time": 5}])

    def test_find_agent_location_latest(self):
        """Test finding the agent's location based on the latest timestamp."""
        self.assertEqual(find_agent_location(self.memory_systems), "room1")

    def test_find_agent_location_not_in_short(self):
        """Test finding the agent's location when it is not in short-term memory."""
        self.short_memory.forget_all()

        with self.assertRaises(ValueError) as context:
            find_agent_location(self.memory_systems)
        self.assertEqual(
            str(context.exception),
            "Make sure that the agent's short-term memory has the current location.",
        )


class TestExplore(unittest.TestCase):

    def setUp(self):
        """Set up the test case with predefined memory systems."""
        self.short_memory = ShortMemory(capacity=10)
        self.long_memory = LongMemory(capacity=10)
        self.memory_systems = MemorySystems(
            short=self.short_memory, long=self.long_memory
        )

        # Add diverse memories to long-term memory
        self.long_memory.add(
            ["agent", "atlocation", "room3", {"timestamp": [3], "strength": 3}]
        )
        self.long_memory.add(
            ["agent", "atlocation", "room4", {"timestamp": [2, 4], "strength": 4}]
        )
        self.long_memory.add(["agent", "atlocation", "room2", {"strength": 5}])
        self.long_memory.add(["agent", "atlocation", "room5", {"timestamp": [1]}])
        self.long_memory.add(
            ["room1", "north", "room2", {"timestamp": [2], "strength": 3}]
        )
        self.long_memory.add(["room1", "east", "wall", {"strength": 3}])
        self.long_memory.add(
            ["room1", "south", "room3", {"timestamp": [2], "strength": 3}]
        )
        self.long_memory.add(
            ["room4", "north", "wall", {"timestamp": [1, 2, 3], "strength": 1}]
        )
        self.long_memory.add(["room4", "east", "wall", {"strength": 1}])
        self.long_memory.add(["room4", "south", "wall", {"timestamp": [5]}])

        # Add memories to short-term memory. This should contain the map of the rooms
        self.short_memory.add(["agent", "atlocation", "room1", {"current_time": 6}])
        self.short_memory.add(["room1", "south", "room3", {"current_time": 6}])
        self.short_memory.add(["laptop", "atlocation", "room1", {"current_time": 6}])

    def test_explore_random(self):
        """Test the 'random' exploration policy."""
        actions = set()
        for _ in range(100):
            action = explore(self.memory_systems, "random")
            actions.add(action)
        expected_actions = {"north", "east", "south", "west", "stay"}
        self.assertTrue(actions.issubset(expected_actions))
        self.assertGreaterEqual(
            len(actions), 3
        )  # Expect at least 3 different actions due to randomness

    def test_explore_avoid_walls(self):
        """Test the 'avoid_walls' exploration policy."""
        actions = set()
        for _ in range(100):
            action = explore(self.memory_systems, "avoid_walls")
            actions.add(action)
        expected_actions = {"north", "south", "west"}
        unexpected_actions = {"east", "stay"}
        self.assertTrue(actions.issubset(expected_actions))
        self.assertFalse(any(action in actions for action in unexpected_actions))

    def test_explore_avoid_walls_no_location(self):
        """Test 'avoid_walls' when agent's location is unknown."""
        self.long_memory.forget_all()
        actions = set()
        for _ in range(100):
            action = explore(self.memory_systems, "avoid_walls")
            actions.add(action)
        expected_actions = {"north", "east", "south", "west", "stay"}
        self.assertTrue(actions.issubset(expected_actions))

    def test_explore_avoid_walls_no_directions(self):
        """Test 'avoid_walls' when there are no directional memories."""
        self.long_memory.forget_all()
        self.long_memory.add(
            ["agent", "atlocation", "room1", {"timestamp": [2], "strength": 3}]
        )
        self.long_memory.add(
            ["agent", "atlocation", "room2", {"timestamp": [3], "strength": 3}]
        )
        self.long_memory.add(
            ["agent", "atlocation", "room3", {"timestamp": [4], "strength": 3}]
        )
        actions = set()
        for _ in range(100):
            action = explore(self.memory_systems, "avoid_walls")
            actions.add(action)
        expected_actions = {"north", "east", "south", "west", "stay"}
        self.assertTrue(actions.issubset(expected_actions))

    def test_explore_avoid_walls_only_strongest_memory(self):
        """Test 'avoid_walls' when the agent should choose based on the strongest
        memory."""
        self.short_memory.forget_all()
        self.long_memory.forget_all()
        self.short_memory.add(["agent", "atlocation", "room1", {"current_time": 6}])
        self.short_memory.add(["room1", "north", "room6", {"current_time": 6}])
        self.long_memory.add(["agent", "atlocation", "room6", {"strength": 6}])
        self.long_memory.add(["room6", "north", "room7", {"strength": 6}])
        actions = set()
        for _ in range(100):
            action = explore(self.memory_systems, "avoid_walls")
            actions.add(action)
        expected_actions = {"north", "east", "south", "west"}
        unexpected_actions = {"stay"}
        self.assertTrue(actions.issubset(expected_actions))
        self.assertFalse(any(action in actions for action in unexpected_actions))

    def test_explore_no_short_memory(self):
        """Test exploration when the agent's location is not in short-term memory."""
        self.short_memory.forget_all()
        with self.assertRaises(ValueError) as context:
            explore(self.memory_systems, "avoid_walls")
        self.assertEqual(
            str(context.exception),
            "Make sure that the agent's short-term memory has the current location.",
        )


class TestEncodeObservation(unittest.TestCase):

    def setUp(self):
        """Set up the test case with predefined memory systems."""
        self.short_memory = ShortMemory(capacity=2)  # Set a small capacity for testing
        self.long_memory = LongMemory(capacity=10)
        self.memory_systems = MemorySystems(
            short=self.short_memory, long=self.long_memory
        )

    def test_encode_observation_success(self):
        """Test successful encoding of an observation into short-term memory."""
        obs = ["agent", "atlocation", "room1", 1]
        encode_observation(self.memory_systems, obs)
        expected_memory = ["agent", "atlocation", "room1", {"current_time": 1}]
        self.assertIn(expected_memory, self.short_memory.entries)

    def test_encode_observation_full_memory(self):
        """Test encoding of an observation into a full short-term memory."""
        self.short_memory.add(["agent", "atlocation", "room2", {"current_time": 2}])
        self.short_memory.add(["agent", "atlocation", "room3", {"current_time": 3}])
        obs = ["agent", "atlocation", "room4", 4]
        with self.assertRaises(ValueError) as context:
            encode_observation(self.memory_systems, obs)
        self.assertEqual(str(context.exception), "The memory system is full!")

    def test_encode_observation_already_present(self):
        """Test encoding an observation already present in short-term memory."""
        obs = ["agent", "atlocation", "room1", 1]
        encode_observation(self.memory_systems, obs)
        encode_observation(
            self.memory_systems, obs
        )  # Encode the same observation again
        self.assertEqual(len(self.short_memory.entries), 1)

    def test_encode_observation_mixed_memory(self):
        """Test encoding multiple observations to short-term memory."""
        obs1 = ["agent", "atlocation", "room1", 1]
        obs2 = ["agent", "atlocation", "room2", 2]
        encode_observation(self.memory_systems, obs1)
        encode_observation(self.memory_systems, obs2)
        expected_memory1 = ["agent", "atlocation", "room1", {"current_time": 1}]
        expected_memory2 = ["agent", "atlocation", "room2", {"current_time": 2}]
        self.assertIn(expected_memory1, self.short_memory.entries)
        self.assertIn(expected_memory2, self.short_memory.entries)


class TestEncodeAllObservations(unittest.TestCase):

    def setUp(self):
        """Set up the test case with predefined memory systems."""
        self.short_memory = ShortMemory(capacity=3)  # Set a small capacity for testing
        self.long_memory = LongMemory(capacity=10)
        self.memory_systems = MemorySystems(
            short=self.short_memory, long=self.long_memory
        )

    def test_encode_all_observations_success(self):
        """Test successful encoding of multiple observations into short-term memory."""
        obs_list = [
            ["agent", "atlocation", "room1", 1],
            ["agent", "atlocation", "room2", 2],
            ["agent", "atlocation", "room3", 3],
        ]
        encode_all_observations(self.memory_systems, obs_list)
        expected_memory1 = ["agent", "atlocation", "room1", {"current_time": 1}]
        expected_memory2 = ["agent", "atlocation", "room2", {"current_time": 2}]
        expected_memory3 = ["agent", "atlocation", "room3", {"current_time": 3}]
        self.assertIn(expected_memory1, self.short_memory.entries)
        self.assertIn(expected_memory2, self.short_memory.entries)
        self.assertIn(expected_memory3, self.short_memory.entries)

    def test_encode_all_observations_partial(self):
        """Test partial encoding when memory is full."""
        obs_list = [
            ["agent", "atlocation", "room1", 1],
            ["agent", "atlocation", "room2", 2],
            ["agent", "atlocation", "room3", 3],
            ["agent", "atlocation", "room3", 4],
        ]
        with self.assertRaises(ValueError) as context:
            encode_all_observations(self.memory_systems, obs_list)
        self.assertEqual(str(context.exception), "The memory system is full!")
        # Check that only part of the observations were added
        self.assertEqual(len(self.short_memory.entries), 3)

    def test_encode_all_observations_no_duplicates(self):
        """Test encoding of multiple observations without duplication."""
        obs_list = [
            ["agent", "atlocation", "room1", 1],
            ["agent", "atlocation", "room1", 1],
        ]
        encode_all_observations(self.memory_systems, obs_list)
        expected_memory = ["agent", "atlocation", "room1", {"current_time": 1}]
        self.assertEqual(self.short_memory.entries.count(expected_memory), 1)


class TestManageMemory(unittest.TestCase):

    def setUp(self):
        """Set up the test case with predefined memory systems."""
        self.short_memory = ShortMemory(capacity=3)
        self.long_memory = LongMemory(capacity=3)
        self.memory_systems = MemorySystems(
            short=self.short_memory, long=self.long_memory
        )

        # Add a short-term memory
        self.short_memory.add(["agent", "atlocation", "room1", {"current_time": 1}])
        self.mem_short = ["agent", "atlocation", "room1", {"current_time": 1}]

    def test_manage_memory_episodic(self):
        """Test moving short-term memory to long-term episodic memory."""
        manage_memory(self.memory_systems, "episodic", self.mem_short)
        expected_memory = ["agent", "atlocation", "room1", {"timestamp": [1]}]
        self.assertIn(expected_memory, self.long_memory.entries)
        self.assertNotIn(self.mem_short, self.short_memory.entries)

    def test_manage_memory_semantic(self):
        """Test moving short-term memory to long-term semantic memory."""
        manage_memory(self.memory_systems, "semantic", self.mem_short)
        expected_memory = ["agent", "atlocation", "room1", {"strength": 1}]
        self.assertIn(expected_memory, self.long_memory.entries)
        self.assertNotIn(self.mem_short, self.short_memory.entries)

    def test_manage_memory_random(self):
        """Test moving short-term memory to long-term memory with random policy."""
        episodic_count = 0
        semantic_count = 0
        for _ in range(100):
            self.short_memory.add(self.mem_short)  # Add memory back to short-term
            manage_memory(self.memory_systems, "random", self.mem_short)
            if [
                "agent",
                "atlocation",
                "room1",
                {"timestamp": [1]},
            ] in self.long_memory.entries:
                episodic_count += 1
                self.long_memory.forget(
                    ["agent", "atlocation", "room1", {"timestamp": [1]}]
                )  # Remove it for next iteration
            if [
                "agent",
                "atlocation",
                "room1",
                {"strength": 1},
            ] in self.long_memory.entries:
                semantic_count += 1
                self.long_memory.forget(
                    ["agent", "atlocation", "room1", {"strength": 1}]
                )  # Remove it for next iteration
        self.assertGreater(episodic_count, 0)
        self.assertGreater(semantic_count, 0)

    def test_manage_memory_forget(self):
        """Test policy where short-term memory is simply forgotten."""
        manage_memory(self.memory_systems, "forget", self.mem_short)
        self.assertNotIn(self.mem_short, self.short_memory.entries)
        self.assertEqual(len(self.long_memory.entries), 0)

    def test_manage_memory_full_memory(self):
        """Test managing memory when long-term memory is full."""
        self.long_memory.add(["agent", "atlocation", "room2", {"timestamp": [2]}])
        self.long_memory.add(["agent", "atlocation", "room3", {"timestamp": [3]}])
        self.long_memory.add(["agent", "atlocation", "room4", {"timestamp": [4]}])
        manage_memory(self.memory_systems, "episodic", self.mem_short)
        self.assertIn(
            ["agent", "atlocation", "room1", {"timestamp": [1]}],
            self.long_memory.entries,
        )
        self.assertEqual(
            len(self.long_memory.entries), 3
        )  # Ensure capacity is maintained

    def test_manage_memory_balancing(self):
        """Test managing memory to maintain balance between episodic and semantic
        memories."""
        self.long_memory.add(["agent", "atlocation", "room2", {"timestamp": [2]}])
        self.long_memory.add(["agent", "atlocation", "room3", {"timestamp": [3]}])
        manage_memory(self.memory_systems, "semantic", self.mem_short)
        self.assertIn(
            ["agent", "atlocation", "room1", {"strength": 1}], self.long_memory.entries
        )
        self.assertNotIn(self.mem_short, self.short_memory.entries)
        self.assertEqual(
            len(self.long_memory.entries), 3
        )  # Ensure capacity is maintained

    def test_manage_memory_complex_memories(self):
        """Test managing memories with both timestamp and strength, and various
        timestamp lists."""
        self.short_memory.add(["agent", "atlocation", "room5", {"current_time": 5}])
        mem_short_complex = ["agent", "atlocation", "room5", {"current_time": 5}]

        # Add complex long-term memories
        self.long_memory.add(
            ["agent", "atlocation", "room6", {"timestamp": [2, 4], "strength": 2}]
        )
        self.long_memory.add(
            ["agent", "atlocation", "room7", {"timestamp": [3], "strength": 3}]
        )
        self.long_memory.add(["agent", "atlocation", "room8", {"strength": 4}])

        manage_memory(self.memory_systems, "episodic", mem_short_complex)
        expected_memory = ["agent", "atlocation", "room5", {"timestamp": [5]}]
        self.assertIn(expected_memory, self.long_memory.entries)
        self.assertNotIn(mem_short_complex, self.short_memory.entries)

        self.short_memory.add(["agent", "atlocation", "room5", {"current_time": 5}])
        manage_memory(self.memory_systems, "semantic", mem_short_complex)
        expected_memory = [
            "agent",
            "atlocation",
            "room5",
            {"timestamp": [5], "strength": 1},
        ]
        self.assertIn(expected_memory, self.long_memory.entries)
        self.assertNotIn(mem_short_complex, self.short_memory.entries)

    def test_manage_memory_balancing_oldest(self):
        """Test managing memory to remove the oldest memory when balancing."""
        self.long_memory.add(["agent", "atlocation", "room2", {"timestamp": [2]}])
        self.long_memory.add(["agent", "atlocation", "room3", {"timestamp": [3]}])
        self.long_memory.add(["agent", "atlocation", "room4", {"timestamp": [4]}])
        manage_memory(self.memory_systems, "episodic", self.mem_short)
        self.assertIn(
            ["agent", "atlocation", "room1", {"timestamp": [1]}],
            self.long_memory.entries,
        )
        self.assertNotIn(
            ["agent", "atlocation", "room2", {"timestamp": [2]}],
            self.long_memory.entries,
        )
        self.assertEqual(
            len(self.long_memory.entries), 3
        )  # Ensure capacity is maintained

    def test_manage_memory_balancing_weakest(self):
        """Test managing memory to remove the weakest memory when balancing."""
        self.long_memory.add(["agent", "atlocation", "room2", {"strength": 1}])
        self.long_memory.add(["agent", "atlocation", "room3", {"strength": 2}])
        self.long_memory.add(["agent", "atlocation", "room4", {"strength": 3}])
        manage_memory(self.memory_systems, "semantic", self.mem_short)
        self.assertIn(
            ["agent", "atlocation", "room1", {"strength": 1}], self.long_memory.entries
        )
        self.assertNotIn(
            ["agent", "atlocation", "room2", {"strength": 1}], self.long_memory.entries
        )
        self.assertEqual(
            len(self.long_memory.entries), 3
        )  # Ensure capacity is maintained

    def test_manage_memory_balancing_complex(self):
        """Test managing memory to balance between episodic and semantic memories."""

        memory_state = []
        for _ in range(100):
            short_memory = ShortMemory(capacity=2)
            long_memory = LongMemory(capacity=2)
            memory_systems = MemorySystems(short=short_memory, long=long_memory)

            # Add a short-term memory
            mem_short = ["agent", "atlocation", "room1", {"current_time": 1}]
            short_memory.add(mem_short)

            long_memory.add(["agent", "atlocation", "room2", {"timestamp": [2]}])
            long_memory.add(["agent", "atlocation", "room3", {"strength": 2}])

            manage_memory(memory_systems, "random", mem_short)
            memory_state.append(deepcopy(memory_systems.get_working_memory().to_list()))

        possible_rooms = [mem[2] for mems in memory_state for mem in mems]
        self.assertGreaterEqual(len(set(possible_rooms)), 3)


class TestAnswerQuestion(unittest.TestCase):

    def setUp(self):
        """Set up the test case with predefined memory systems."""
        self.short_memory = ShortMemory(capacity=3)
        self.long_memory = LongMemory(capacity=6)
        self.memory_systems = MemorySystems(
            short=self.short_memory, long=self.long_memory
        )

        # Add memories to long-term memory
        self.long_memory.add(
            ["agent", "atlocation", "room3", {"timestamp": [3], "strength": 3}]
        )
        self.long_memory.add(
            ["agent", "atlocation", "room4", {"timestamp": [2, 4], "strength": 4}]
        )
        self.long_memory.add(["agent", "atlocation", "room2", {"strength": 5}])
        self.long_memory.add(["agent", "atlocation", "room5", {"timestamp": [1]}])
        self.long_memory.add(["laptop", "atlocation", "room2", {"timestamp": [4]}])
        self.long_memory.add(["laptop", "atlocation", "room2", {"strength": 2}])

        # Add memories to short-term memory
        self.short_memory.add(["agent", "atlocation", "room1", {"current_time": 5}])
        self.short_memory.add(["laptop", "atlocation", "room1", {"current_time": 5}])

    def test_answer_question_latest(self):
        """Test answering a question using the latest memory."""
        question = ["agent", "atlocation", "?", 5]
        answer = answer_question(self.memory_systems, "latest", question)
        self.assertEqual(answer, "room1")

    def test_answer_question_strongest(self):
        """Test answering a question using the strongest memory."""
        question = ["agent", "atlocation", "?", 5]
        answer = answer_question(self.memory_systems, "strongest", question)
        self.assertEqual(answer, "room2")

    def test_answer_question_random(self):
        """Test answering a question using a random memory."""
        question = ["agent", "atlocation", "?", 5]
        answers = set()
        for _ in range(100):
            answer = answer_question(self.memory_systems, "random", question)
            answers.add(answer)
        self.assertTrue(answers.issubset({"room1", "room2", "room3", "room4", "room5"}))
        self.assertGreaterEqual(
            len(answers), 2
        )  # Expect at least 2 different answers due to randomness

    def test_answer_question_latest_strongest(self):
        """Test answering a question using the latest and strongest memory."""
        question = ["agent", "atlocation", "?", 5]
        answer = answer_question(self.memory_systems, "latest_strongest", question)
        self.assertEqual(answer, "room1")

    def test_answer_question_strongest_latest(self):
        """Test answering a question using the strongest and latest memory."""
        question = ["agent", "atlocation", "?", 5]
        answer = answer_question(self.memory_systems, "strongest_latest", question)
        self.assertEqual(answer, "room2")

    def test_answer_question_no_relevant_memory(self):
        """Test answering a question when no relevant memory is found."""
        question = ["car", "atlocation", "?", 5]
        answer = answer_question(self.memory_systems, "latest", question)
        self.assertIsNone(answer)
        answer = answer_question(self.memory_systems, "strongest", question)
        self.assertIsNone(answer)
        answer = answer_question(self.memory_systems, "latest_strongest", question)
        self.assertIsNone(answer)
        answer = answer_question(self.memory_systems, "strongest_latest", question)
        self.assertIsNone(answer)
