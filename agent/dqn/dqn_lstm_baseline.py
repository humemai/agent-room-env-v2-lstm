"""DQN LSTM MLP Baseline Agent for the RoomEnv2 environment."""

import datetime
import os
import random
import shutil
from copy import deepcopy

import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
from humemai.utils import is_running_notebook, write_yaml
from room_env.envs.room2 import RoomEnv2
from torch import nn

from .utils import (
    ReplayBuffer,
    plot_results,
    save_final_results,
    save_states_q_values_actions,
    save_validation,
    select_action,
    target_hard_update,
    update_epsilon,
    update_model,
)


class LSTM(torch.nn.Module):
    """LSTMMLP for the DQN baseline."""

    def __init__(
        self,
        entities: list,
        relations: list,
        num_layers: int = 2,
        embedding_dim: int = 64,
        hidden_size: int = 64,
        bidirectional: bool = False,
        device: str = "cpu",
    ) -> None:
        """Initialize the LSTM.

        Args:
            entities: List of entities.
            relations: List of relations.
            num_layers: Number of layers in the LSTM.
            embedding_dim: Dimension of the embeddings.
            hidden_size: Hidden size of the LSTM.
            bidirectional: Whether the LSTM is bidirectional.
            device: Device to use.

        """
        super().__init__()
        self.entities = entities
        self.relations = relations
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.bidirectional = bidirectional
        self.device = device

        self.create_embeddings()
        self.lstm = torch.nn.LSTM(
            self.embedding_dim,
            self.hidden_size,
            self.num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            device=self.device,
        )

    def create_embeddings(self) -> None:
        """Create learnable embeddings."""
        if isinstance(self.entities, dict):
            self.word2idx = (
                ["<PAD>"]
                + [name for names in self.entities.values() for name in names]
                + self.relations
            )
        elif isinstance(self.entities, list):
            self.word2idx = ["<PAD>"] + self.entities + self.relations
        else:
            raise ValueError(
                "entities should be either a list or a dictionary, but "
                f"{type(self.entities)} was given!"
            )
        self.word2idx = {word: idx for idx, word in enumerate(self.word2idx)}

        self.embeddings = nn.Embedding(
            len(self.word2idx),
            self.embedding_dim,
            device=self.device,
            padding_idx=0,
        )

    def make_embedding(self, obs: list[str]) -> torch.Tensor:
        """Create one embedding vector with summation.

        Args:
            obs: Observation.

        Returns:
            final_embedding: Final embedding.

        """
        if obs == ["<PAD>", "<PAD>", "<PAD>", "<PAD>"]:
            return self.embeddings(
                torch.tensor(self.word2idx["<PAD>"], device=self.device)
            )

        head_embedding = self.embeddings(
            torch.tensor(self.word2idx[obs[0]], device=self.device)
        )
        relation_embedding = self.embeddings(
            torch.tensor(self.word2idx[obs[1]], device=self.device)
        )
        tail_embedding = self.embeddings(
            torch.tensor(self.word2idx[obs[2]], device=self.device)
        )
        final_embedding = head_embedding + relation_embedding + tail_embedding

        return final_embedding

    def forward(
        self, x: np.ndarray, memory_types: None = None
    ) -> tuple[torch.Tensor, None]:
        """Forward pass.

        Args:
            x: Input.

        Returns:
            lstm_out: LSTM last hidden state.
            None: This is for compatibility with the DQN agent.

        """
        obs_pad = ["<PAD>", "<PAD>", "<PAD>", "<PAD>"]
        batch = [sample["data"] for sample in x]
        max_len = 0
        for sample in batch:
            if len(sample) > max_len:
                max_len = len(sample)

        for observations in batch:
            while len(observations) < max_len:
                observations.append(obs_pad)

        batch_embeddings = []
        for observations in batch:
            embeddings = []
            for obs in observations:
                obs_emb = self.make_embedding(obs)
                embeddings.append(obs_emb)
            embeddings = torch.stack(embeddings)
            batch_embeddings.append(embeddings)

        batch_embeddings = torch.stack(batch_embeddings)

        lstm_out = self.lstm(batch_embeddings)[0]
        lstm_out = lstm_out[:, -1, :]

        return lstm_out, None


class MLP(torch.nn.Module):
    """Multi-layer perceptron with ReLU activation functions."""

    def __init__(
        self,
        n_actions: int,
        hidden_size: int,
        device: str,
        num_hidden_layers: int = 1,
        dueling_dqn: bool = True,
    ) -> None:
        """Initialize the MLP.

        Args:
            n_actions: Number of actions.
            hidden_size: Hidden size of the linear layer.
            device: "cpu" or "cuda".
            num_hidden_layers: int, number of hidden layers in the MLP.
            dueling_dqn: Whether to use dueling DQN.

        """
        super(MLP, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.n_actions = n_actions
        self.dueling_dqn = dueling_dqn

        # Define the layers for the advantage stream
        advantage_layers = []
        for _ in range(self.num_hidden_layers):
            advantage_layers.append(
                torch.nn.Linear(self.hidden_size, self.hidden_size, device=self.device)
            )
            advantage_layers.append(torch.nn.ReLU())
        advantage_layers.append(
            torch.nn.Linear(self.hidden_size, self.n_actions, device=self.device)
        )
        self.advantage_layer = torch.nn.Sequential(*advantage_layers)

        if self.dueling_dqn:
            # Define the layers for the value stream
            value_layers = []
            for _ in range(self.num_hidden_layers):
                value_layers.append(
                    torch.nn.Linear(
                        self.hidden_size, self.hidden_size, device=self.device
                    )
                )
                value_layers.append(torch.nn.ReLU())
            value_layers.append(
                torch.nn.Linear(self.hidden_size, 1, device=self.device)
            )
            self.value_layer = torch.nn.Sequential(*value_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the neural network.

        Args:
            x: Input tensor. The shape is (batch_size, lstm_hidden_size).
        Returns:
            torch.Tensor: Output tensor. The shape is (batch_size, n_actions).

        """

        if self.dueling_dqn:
            value = self.value_layer(x)
            advantage = self.advantage_layer(x)
            q = value + advantage - advantage.mean(dim=-1, keepdim=True)
        else:
            q = self.advantage_layer(x)

        return q


class History:
    def __init__(
        self,
        block_size: int = 4,
        action2str: dict = {0: "north", 1: "east", 2: "south", 3: "west", 4: "stay"},
    ) -> None:
        """Initialize the history.

        Args:
            block_size: Block size.
            action2str: Action number to action string.

        """
        self.block_size = block_size
        self.blocks = [[]] * self.block_size

        self.action2str = action2str
        self.actions_str = list(self.action2str.values())
        self.actions_int = list(self.action2str.keys())

    def to_list(self) -> list:
        """Convert the history to a list."""
        return [element for block in self.blocks for element in block]

    def add_block(self, block: list) -> None:
        """Add a block to the history. The oldest block is removed."""
        self.blocks = self.blocks[1:] + [block]

    def __repr__(self) -> str:
        return str(self.blocks)

    def answer_question(self, question: list) -> str:
        """Answer a question, by going through the history in backwards. This means that
        the most recent and relevant observation is used to answer the question.
        """
        assert len(question) == 4 and question[2] == "?"
        for obs in self.to_list()[::-1]:
            if obs[0] == question[0] and obs[1] == question[1]:
                return obs[2]

    def answer_questions(self, questions: list[list[str]]) -> list[str]:
        """Answer a list of questions.

        Args:
            questions: List of questions.

        Returns:
            answers: List of answers.
        """
        return [self.answer_question(question) for question in questions]

    def find_agent_current_location(self) -> str | None:
        """Find the current location of the agent, by going through the history in
        backwards."""
        for obs in self.to_list()[::-1]:
            if obs[0] == "agent" and obs[1] == "atlocation":
                return obs[2]

        return None

    def explore(self, explore_policy: str) -> str:
        """Explore the environment. The agent can either explore randomly or avoid
        walls. This is not reinforcement learning, but a simple heuristic.

        Args:
            explore_policy: Explore policy.

        Returns:
            action_explore: Action to explore.

        """
        if explore_policy not in ["random", "avoid_walls"]:
            raise ValueError(f"Unknown explore policy: {explore_policy}")

        if explore_policy == "random":
            action_explore = random.choice(self.actions_str)
        elif explore_policy == "avoid_walls":
            current_agent_location = self.find_agent_current_location()
            if current_agent_location is None:
                action_explore = random.choice(self.actions_str)
            else:
                relevant_observations = []
                relevant_observations += [
                    obs
                    for obs in self.to_list()[::-1]
                    if obs[0] == current_agent_location
                    and obs[1] in ["north", "east", "south", "west"]
                ]
                # we know the agent's current location but there is no memory about map
                if len(relevant_observations) == 0:
                    action_explore = random.choice(self.actions_str)
                else:
                    # we know the agent's current location and there is at least one
                    # memory about the map and we want to avoid the walls

                    to_take = []
                    to_avoid = []

                    for obs in relevant_observations:
                        if isinstance(obs, dict):
                            continue
                        if obs[2].split("_")[0] == "room":
                            to_take.append(obs[1])
                        elif obs[2] == "wall":
                            if obs[1] not in to_avoid:
                                to_avoid.append(obs[1])

                    if len(to_take) > 0:
                        action_explore = random.choice(to_take)
                    else:
                        options = deepcopy(self.actions_str)
                        for e in to_avoid:
                            options.remove(e)

                        action_explore = random.choice(options)

        return action_explore


class DQNLSTMMLPBaselineAgent:
    """DQN LSTM Baseline Agent interacting with environment.

    Based on https://github.com/Curt-Park/rainbow-is-all-you-need/
    """

    def __init__(
        self,
        env_str: str = "room_env:RoomEnv-v2",
        num_iterations: int = 10000,
        replay_buffer_size: int = 10000,
        validation_starts_at: int = 5000,
        warm_start: int = 1000,
        batch_size: int = 32,
        target_update_interval: int = 10,
        epsilon_decay_until: float = 10000,
        max_epsilon: float = 1.0,
        min_epsilon: float = 0.1,
        gamma: float = 0.9,
        history_block_size: int = 4,
        lstm_params: dict = {
            "num_layers": 2,
            "embedding_dim": 64,
            "hidden_size": 64,
            "bidirectional": False,
        },
        mlp_params: dict = {
            "hidden_size": 64,
            "num_hidden_layers": 1,
            "dueling_dqn": True,
        },
        num_samples_for_results: dict = {"val": 10, "test": 10},
        validation_interval: int = 1,
        plotting_interval: int = 20,
        train_seed: int = 5,
        test_seed: int = 0,
        device: str = "cpu",
        env_config: dict = {
            "question_prob": 1.0,
            "terminates_at": 99,
            "randomize_observations": "objects",
            "room_size": "l",
            "rewards": {"correct": 1, "wrong": 0, "partial": 0},
            "make_everything_static": False,
            "num_total_questions": 1000,
            "question_interval": 5,
            "include_walls_in_observations": True,
        },
        default_root_dir: str = "./training-results/",
        run_handcrafted_baselines: bool = True,
    ) -> None:
        """Initialize the DQN LSTM Baseline Agent.

        Args:
            env_str: Environment string.
            num_iterations: Number of iterations.
            replay_buffer_size: Size of the replay buffer.
            validation_starts_at: validation_starts_at
            warm_start: Warm start size.
            batch_size: Batch size.
            target_update_interval: Target update interval.
            epsilon_decay_until: Epsilon decay until.
            max_epsilon: Maximum epsilon.
            min_epsilon: Minimum epsilon.
            gamma: discount factor.
            history_block_size: History block size.
            lstm_params: LSTM parameters.
            mlp_params: MLP parameters.
            num_samples_for_results: Number of samples to use for results.
            validation_interval: Validation interval.
            plotting_interval: Plotting interval.
            train_seed: Train seed.
            test_seed: Test seed.
            device: "cpu" or "cuda".
            env_config: Environment configuration.
            default_root_dir: Default root directory to save training results
            run_handcrafted_baselines: Whether to run handcrafted baselines.

        """
        params_to_save = deepcopy(locals())
        del params_to_save["self"]
        self.default_root_dir = os.path.join(
            default_root_dir, str(datetime.datetime.now())
        )
        self._create_directory(params_to_save)

        self.history_block_size = history_block_size

        self.train_seed = train_seed
        self.test_seed = test_seed
        self.env_config = env_config
        self.env_config["seed"] = self.train_seed
        self.env_str = env_str
        self.num_samples_for_results = num_samples_for_results
        self.validation_interval = validation_interval
        self.env = gym.make(self.env_str, **self.env_config)

        self.run_handcrafted_baselines = run_handcrafted_baselines

        self.device = torch.device(device)
        print(f"Running on {self.device}")

        self.action2str = {0: "north", 1: "east", 2: "south", 3: "west", 4: "stay"}
        self.str2action = {v: k for k, v in self.action2str.items()}
        self.action_space = gym.spaces.Discrete(len(self.action2str))

        self.val_dir_names = []
        self.is_notebook = is_running_notebook()
        self.num_iterations = num_iterations
        self.plotting_interval = plotting_interval

        self.replay_buffer_size = replay_buffer_size
        self.validation_starts_at = validation_starts_at
        self.batch_size = batch_size
        self.epsilon = max_epsilon
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay_until = epsilon_decay_until
        self.target_update_interval = target_update_interval
        self.gamma = gamma
        self.warm_start = warm_start
        assert self.batch_size <= self.warm_start <= self.replay_buffer_size

        self.lstm_params = lstm_params
        self.lstm_params["device"] = self.device
        self.lstm_params["entities"] = self.env.unwrapped.entities
        self.lstm_params["relations"] = self.env.unwrapped.relations

        self.lstm = LSTM(**self.lstm_params)
        self.lstm_target = LSTM(**self.lstm_params)
        self.lstm_target.load_state_dict(self.lstm.state_dict())
        self.lstm_target.eval()

        self.mlp_params = mlp_params
        self.mlp_params["device"] = self.device

        self.mlp = MLP(n_actions=len(self.action2str), **self.mlp_params)
        self.mlp_target = MLP(n_actions=len(self.action2str), **self.mlp_params)
        self.mlp_target.load_state_dict(self.mlp.state_dict())
        self.mlp_target.eval()

        self._save_number_of_parameters()

        self.replay_buffer = ReplayBuffer(
            observation_type="dict", size=replay_buffer_size, batch_size=batch_size
        )

        # optimizer
        self.optimizer = optim.Adam(
            list(self.lstm.parameters()) + list(self.mlp.parameters())
        )

        self.q_values = {"train": [], "val": [], "test": []}

        if self.run_handcrafted_baselines:
            self._run_explore_baselines()

    def _run_explore_baselines(self) -> None:
        """Run the explore baselines."""
        env = RoomEnv2(**self.env_config)
        observations, info = env.reset()
        env.render("image", save_fig_dir=self.default_root_dir)

        del env

        env_config = deepcopy(self.env_config)

        results = {}
        for explore_policy in ["random", "avoid_walls"]:
            results[str((explore_policy))] = []
            scores = []
            for test_seed in [0, 1, 2, 3, 4]:
                score = 0
                env_config["seed"] = test_seed
                env = gym.make(self.env_str, **env_config)
                observations, info = env.reset()
                self.history = History(self.history_block_size, self.action2str)
                self.history.add_block(observations["room"])

                while True:
                    action_explore = self.history.explore(explore_policy)
                    actions_qa = self.history.answer_questions(
                        observations["questions"]
                    )

                    action_pair = (actions_qa, action_explore)
                    (
                        observations,
                        reward,
                        done,
                        truncated,
                        info,
                    ) = env.step(action_pair)
                    score += reward
                    done = done or truncated

                    self.history.add_block(observations["room"])

                    if done:
                        break

                scores.append(score)

            results[str((explore_policy))] = {
                "mean": np.mean(scores).item(),
                "std": np.std(scores).item(),
            }

        write_yaml(results, os.path.join(self.default_root_dir, "handcrafted.yaml"))

    def _create_directory(self, params_to_save: dict) -> None:
        """Create the directory to store the results.

        Args:
            params_to_save: (hyper) parameters to save.

        """
        os.makedirs(self.default_root_dir, exist_ok=True)
        write_yaml(params_to_save, os.path.join(self.default_root_dir, "train.yaml"))

    def _save_number_of_parameters(self) -> None:
        """Save the number of parameters in the model."""
        write_yaml(
            {
                "lstm": sum(p.numel() for p in self.lstm.parameters()),
                "mlp": sum(p.numel() for p in self.mlp.parameters()),
            },
            os.path.join(self.default_root_dir, "num_params.yaml"),
        )

    def remove_results_from_disk(self) -> None:
        """Remove the results from the disk."""
        shutil.rmtree(self.default_root_dir)

    def step(
        self, state: dict, questions: list, greedy: bool
    ) -> tuple[dict, int, float, bool, list]:
        """Step through the environment.

        This is the only place where env.step() is called. Make sure to call this
        function to interact with the environment.

        Args:
            state: state of the history
            questions: questions to answer
            greedy: whether to act greedily

        Returns:
            observations: observations from the environment
            action_explore: exploration action taken
            reward: reward received
            done: whether the episode is done
            q_values: q values (list of floats)

        """
        # select explore action
        action_explore, q_values = select_action(
            memory_types=None,
            state=state,
            greedy=greedy,
            lstm=self.lstm,
            mlp=self.mlp,
            epsilon=self.epsilon,
            action_space=self.action_space,
        )
        actions_qa = self.history.answer_questions(questions)

        action_pair = (actions_qa, self.action2str[action_explore])
        (
            observations,
            reward,
            done,
            truncated,
            info,
        ) = self.env.step(action_pair)
        done = done or truncated

        return observations, action_explore, reward, done, q_values

    def fill_replay_buffer(self) -> None:
        """Make the replay buffer full in the beginning with the uniformly-sampled
        actions. The filling continues until it reaches the warm start size.

        """
        new_episode_starts = True
        while len(self.replay_buffer) < self.warm_start:

            if new_episode_starts:
                self.history = History(self.history_block_size)
                observations, info = self.env.reset()
                done = False
                self.history.add_block(observations["room"])
                new_episode_starts = False

            else:
                state = deepcopy({"data": self.history.to_list()})
                observations, action, reward, done, q_values = self.step(
                    state, observations["questions"], greedy=False
                )
                self.history.add_block(observations["room"])
                next_state = deepcopy({"data": self.history.to_list()})
                scaled_reward = reward / self.env.unwrapped.num_questions_step
                assert scaled_reward <= 1
                self.replay_buffer.store(
                    *[state, action, scaled_reward, next_state, done]
                )

            if done:
                new_episode_starts = True

    def train(self) -> None:
        """Train the explore agent."""
        self.fill_replay_buffer()  # fill up the buffer till warm start size

        self.epsilons = []
        self.training_loss = []
        self.scores = {"train": [], "val": [], "test": None}

        if self.validation_starts_at > 0:
            for _ in range(
                self.validation_starts_at // (self.env_config["terminates_at"] + 1)
                - self.validation_interval
            ):
                self.scores["val"].append([0] * self.num_samples_for_results["val"])

        self.lstm.train()
        self.mlp.train()

        new_episode_starts = True
        score = 0

        self.iteration_idx = 0

        while True:
            if new_episode_starts:
                self.history = History(self.history_block_size)
                observations, info = self.env.reset()
                done = False
                self.history.add_block(observations["room"])
                new_episode_starts = False

            else:
                state = deepcopy({"data": self.history.to_list()})
                observations, action, reward, done, q_values = self.step(
                    state, observations["questions"], greedy=False
                )
                self.history.add_block(observations["room"])
                next_state = deepcopy({"data": self.history.to_list()})
                scaled_reward = reward / self.env.unwrapped.num_questions_step
                assert scaled_reward <= 1
                self.replay_buffer.store(
                    *[state, action, scaled_reward, next_state, done]
                )
                self.q_values["train"].append(q_values)
                score += reward
                self.iteration_idx += 1

            if done:
                new_episode_starts = True

                self.scores["train"].append(score)
                score = 0

                if (
                    self.iteration_idx >= self.validation_starts_at
                ) and self.iteration_idx % (
                    self.validation_interval * (self.env_config["terminates_at"] + 1)
                ) == 0:
                    with torch.no_grad():
                        self.validate()

            if not new_episode_starts:
                loss = update_model(
                    memory_types=None,
                    replay_buffer=self.replay_buffer,
                    optimizer=self.optimizer,
                    device=self.device,
                    lstm=self.lstm,
                    lstm_target=self.lstm_target,
                    mlp=self.mlp,
                    mlp_target=self.mlp_target,
                    ddqn=True,
                    gamma=self.gamma,
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

                # if hard update is needed
                if self.iteration_idx % self.target_update_interval == 0:
                    target_hard_update(self.lstm, self.lstm_target)
                    target_hard_update(self.mlp, self.mlp_target)

                # plotting & show training results
                if (
                    self.iteration_idx == self.num_iterations
                    or self.iteration_idx % self.plotting_interval == 0
                ):
                    self.plot_results("all", save_fig=True)

                if self.iteration_idx == self.num_iterations:
                    break

        with torch.no_grad():
            self.test()

        self.env.close()

    def validate_test_middle(self, val_or_test: str) -> tuple[list[float], dict]:
        """A function shared by explore validation and test in the middle.

        Args:
            val_or_test: "val" or "test"

        Returns:
            scores_local: a list of total episode rewards
            states_local: memory states
            q_values_local: q values
            actions_local: greey actions taken

        """
        scores_local = []
        states_local = []
        q_values_local = []
        actions_local = []

        for idx in range(self.num_samples_for_results[val_or_test]):
            new_episode_starts = True
            score = 0
            while True:
                if new_episode_starts:
                    self.history = History(self.history_block_size)
                    observations, info = self.env.reset()
                    done = False
                    self.history.add_block(observations["room"])
                    new_episode_starts = False

                else:
                    state = deepcopy({"data": self.history.to_list()})
                    observations, action, reward, done, q_values = self.step(
                        state, observations["questions"], greedy=True
                    )
                    if not done:
                        self.history.add_block(observations["room"])
                    score += reward

                    if idx == self.num_samples_for_results[val_or_test] - 1:
                        states_local.append(state)
                        q_values_local.append(q_values)
                        self.q_values[val_or_test].append(q_values)
                        actions_local.append(action)

                if done:
                    break

            scores_local.append(score)

        return scores_local, states_local, q_values_local, actions_local

    def validate(self) -> None:
        """Validate the agent."""
        self.lstm.eval()
        self.mlp.eval()
        scores_temp, states, q_values, actions = self.validate_test_middle("val")

        num_episodes = self.iteration_idx // (self.env_config["terminates_at"] + 1)

        save_validation(
            policy=None,
            scores_temp=scores_temp,
            scores=self.scores,
            default_root_dir=self.default_root_dir,
            num_episodes=num_episodes,
            validation_interval=self.validation_interval,
            val_dir_names=self.val_dir_names,
            lstm=self.lstm,
            mlp=self.mlp,
        )
        save_states_q_values_actions(
            None,
            states,
            q_values,
            actions,
            self.default_root_dir,
            "val",
            num_episodes,
        )
        self.env.close()

        self.lstm.train()
        self.mlp.train()

    def test(
        self, checkpoint_lstm: str | None = None, checkpoint_mlp: str | None = None
    ) -> None:
        """Test the agent.

        Args:
            checkpoint: Checkpoint to load. If None, the highest validation checkpoint
                is loaded.

        """
        self.lstm.eval()
        self.mlp.eval()
        self.env_config["seed"] = self.test_seed
        self.env = gym.make(self.env_str, **self.env_config)

        assert len(self.val_dir_names) == 1, f"{len(self.val_dir_names)} is not 1!"

        self.lstm.load_state_dict(
            torch.load(os.path.join(self.val_dir_names[0], "lstm.pt"))
        )
        self.mlp.load_state_dict(
            torch.load(os.path.join(self.val_dir_names[0], "mlp.pt"))
        )

        if checkpoint_lstm is not None:
            self.lstm.load_state_dict(torch.load(checkpoint_lstm))
        if checkpoint_mlp is not None:
            self.mlp.load_state_dict(torch.load(checkpoint_mlp))

        scores, states, q_values, actions = self.validate_test_middle("test")
        self.scores["test"] = scores

        save_final_results(
            self.scores,
            self.training_loss,
            self.default_root_dir,
            self.q_values,
            self,
            None,
        )
        save_states_q_values_actions(
            None, states, q_values, actions, self.default_root_dir, "test"
        )

        self.plot_results("all", save_fig=True)
        self.env.close()

    def plot_results(self, to_plot: str = "all", save_fig: bool = False) -> None:
        """Plot things for DQN training.

        Args:
            to_plot: what to plot:
                training_td_loss
                epsilons
                training_score
                validation_score
                test_score
                q_values_train
                q_values_val
                q_values_test

        """
        plot_results(
            None,
            self.scores,
            self.training_loss,
            self.epsilons,
            self.q_values,
            self.iteration_idx,
            self.action_space.n.item(),
            self.num_iterations,
            self.env.unwrapped.total_maximum_episode_rewards,
            self.default_root_dir,
            to_plot,
            save_fig,
        )
