"""LSTM to approximate a function."""

from copy import deepcopy
from typing import Literal

import numpy as np
import torch
from torch import nn

from ...utils import positional_encoding


def init_layer_uniform(layer: nn.Linear, init_w: float = 3e-3) -> None:
    """Init uniform parameters on the single layer."""
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_w, init_w)


class LSTM(nn.Module):
    """A simple LSTM network."""

    def __init__(
        self,
        capacity: dict,
        entities: list,
        relations: list,
        n_actions: int,
        memory_of_interest: list,
        hidden_size: int = 64,
        num_layers: int = 2,
        embedding_dim: int = 64,
        make_categorical_embeddings: bool = False,
        batch_first: bool = True,
        device: str = "cpu",
        dueling_dqn: bool = False,
        fuse_information: Literal["concat", "sum"] = "sum",
        include_positional_encoding: bool = True,
        max_timesteps: int | None = None,
        max_strength: int | None = None,
    ) -> None:
        """Initialize the LSTM.

        Args:
            capacity: the capacities of memory systems. e.g., {"episodic": 16,
                "semantic": 16, "short": 1}
            entities: list of entities, e.g., ["Foo", "Bar", "laptop", "phone",
                "desk", "lap"]
            relations : list of relations, e.g., ["atlocation", "north", "south"]
            n_actions: number of actions. This should be 3, at the moment.
            memory_of_interest: e.g., ["episodic", "semantic", "short"]
            hidden_size: hidden size of the LSTM
            num_layers: number of the LSTM layers
            embedding_dim: entity embedding dimension (e.g., 32)
            make_categorical_embeddings: whether to use categorical embeddings or not.
            batch_first: Should the batch dimension be the first or not.
            device: "cpu" or "cuda"
            dueling_dqn: whether to use dueling DQN or not.
            fuse_information: "concat" or "sum"
            include_positional_encoding: whether to include the number 4, i.e.,
                strength or timestamp in the entity list.
            max_timesteps: maximum number of timesteps. This is only used when
                `include_positional_encoding` is True.
            max_strength: maximum strength. This is only used when
                `include_positional_encoding` is True.

        """
        super().__init__()
        self.capacity = capacity
        self.memory_of_interest = memory_of_interest
        self.entities = entities
        self.relations = relations
        self.n_actions = n_actions
        self.embedding_dim = embedding_dim
        self.make_categorical_embeddings = make_categorical_embeddings
        self.device = device
        self.dueling_dqn = dueling_dqn
        self.fuse_information = fuse_information
        self.include_positional_encoding = include_positional_encoding
        self.max_timesteps = max_timesteps
        self.max_strength = max_strength

        if self.fuse_information == "concat":
            self.linear_layer_hidden_size = hidden_size * len(self.memory_of_interest)
        elif self.fuse_information == "sum":
            self.linear_layer_hidden_size = hidden_size
        else:
            raise ValueError(
                f"fuse_information should be one of 'concat' or 'sum', but "
                f"{self.fuse_information} was given!"
            )

        if self.include_positional_encoding:
            assert self.max_timesteps is not None
            assert self.max_strength is not None
            self.positional_encoding = positional_encoding(
                positions=max(self.max_timesteps, self.max_strength) + 1,
                dimensions=self.linear_layer_hidden_size,
                scaling_factor=10000,
                return_tensor=True,
            ).to(self.device)

        self.create_embeddings()
        if "episodic" in self.memory_of_interest:
            self.lstm_e = nn.LSTM(
                self.input_size_e,
                hidden_size,
                num_layers,
                batch_first=batch_first,
                device=self.device,
            )
            if self.fuse_information == "concat":
                self.fc_e0 = nn.Linear(hidden_size, hidden_size, device=self.device)
                self.fc_e1 = nn.Linear(hidden_size, hidden_size, device=self.device)

        if "episodic_agent" in self.memory_of_interest:
            self.lstm_e_agent = nn.LSTM(
                self.input_size_e,
                hidden_size,
                num_layers,
                batch_first=batch_first,
                device=self.device,
            )
            if self.fuse_information == "concat":
                self.fc_e0_agent = nn.Linear(
                    hidden_size, hidden_size, device=self.device
                )
                self.fc_e1_agent = nn.Linear(
                    hidden_size, hidden_size, device=self.device
                )

        if "semantic" in self.memory_of_interest:
            self.lstm_s = nn.LSTM(
                self.input_size_s,
                hidden_size,
                num_layers,
                batch_first=batch_first,
                device=self.device,
            )
            if self.fuse_information == "concat":
                self.fc_s0 = nn.Linear(hidden_size, hidden_size, device=self.device)
                self.fc_s1 = nn.Linear(hidden_size, hidden_size, device=self.device)

        if "semantic_map" in self.memory_of_interest:
            self.lstm_s_map = nn.LSTM(
                self.input_size_s,
                hidden_size,
                num_layers,
                batch_first=batch_first,
                device=self.device,
            )
            if self.fuse_information == "concat":
                self.fc_s0_map = nn.Linear(hidden_size, hidden_size, device=self.device)
                self.fc_s1_map = nn.Linear(hidden_size, hidden_size, device=self.device)

        if "short" in self.memory_of_interest:
            self.lstm_o = nn.LSTM(
                self.input_size_o,
                hidden_size,
                num_layers,
                batch_first=batch_first,
                device=self.device,
            )
            if self.fuse_information == "concat":
                self.fc_o0 = nn.Linear(hidden_size, hidden_size, device=self.device)
                self.fc_o1 = nn.Linear(hidden_size, hidden_size, device=self.device)

        self.advantage_layer = nn.Sequential(
            nn.Linear(
                self.linear_layer_hidden_size,
                self.linear_layer_hidden_size,
                device=self.device,
            ),
            nn.ReLU(),
            nn.Linear(
                self.linear_layer_hidden_size,
                self.n_actions,
                device=self.device,
            ),
        )

        if self.dueling_dqn:
            self.value_layer = nn.Sequential(
                nn.Linear(
                    self.linear_layer_hidden_size,
                    self.linear_layer_hidden_size,
                    device=self.device,
                ),
                nn.ReLU(),
                nn.Linear(
                    self.linear_layer_hidden_size,
                    1,
                    device=self.device,
                ),
            )
        self.relu = nn.ReLU()

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

        if self.make_categorical_embeddings:
            # Assuming self.entities is a dictionary where keys are categories and
            # values are lists of entity names

            # Create a dictionary to keep track of starting index for each category
            category_start_indices = {}
            current_index = 1  # Start from 1 to skip the <PAD> token
            for category, names in self.entities.items():
                category_start_indices[category] = current_index
                current_index += len(names)

            # Re-initialize embeddings by category
            for category, start_idx in category_start_indices.items():
                end_idx = start_idx + len(self.entities[category])
                init_vector = torch.randn(self.embedding_dim, device=self.device)
                self.embeddings.weight.data[start_idx:end_idx] = init_vector.repeat(
                    end_idx - start_idx, 1
                )
            # Note: Relations are not re-initialized by category, assuming they are
            # separate from entities

        if self.fuse_information == "concat":
            self.input_size_s = self.embedding_dim * 3
            self.input_size_e = self.embedding_dim * 3
            self.input_size_o = self.embedding_dim * 3

        elif self.fuse_information == "sum":
            self.input_size_s = self.embedding_dim
            self.input_size_e = self.embedding_dim
            self.input_size_o = self.embedding_dim
        else:
            raise ValueError(
                f"fuse_information should be one of 'concat' or 'sum', but "
                f"{self.fuse_information} was given!"
            )

    def make_embedding(self, mem: list[str], *args) -> torch.Tensor:
        """Create one embedding vector with summation and concatenation.

        Args:
            mem: memory as a quadruple: [head, relation, tail, num]

        Returns:
            one embedding vector made from one memory element.

        """
        if mem == ["<PAD>", "<PAD>", "<PAD>", "<PAD>"]:
            if self.fuse_information == "sum":
                return self.embeddings(
                    torch.tensor(self.word2idx["<PAD>"], device=self.device)
                )
            else:
                final_embedding = torch.concat(
                    [
                        self.embeddings(
                            torch.tensor(self.word2idx["<PAD>"], device=self.device)
                        ),
                        self.embeddings(
                            torch.tensor(self.word2idx["<PAD>"], device=self.device)
                        ),
                        self.embeddings(
                            torch.tensor(self.word2idx["<PAD>"], device=self.device)
                        ),
                    ]
                )
                return final_embedding

        head_embedding = self.embeddings(
            torch.tensor(self.word2idx[mem[0]], device=self.device)
        )
        relation_embedding = self.embeddings(
            torch.tensor(self.word2idx[mem[1]], device=self.device)
        )
        tail_embedding = self.embeddings(
            torch.tensor(self.word2idx[mem[2]], device=self.device)
        )
        if self.fuse_information == "concat":
            final_embedding = torch.concat(
                [head_embedding, relation_embedding, tail_embedding]
            )
        elif self.fuse_information == "sum":
            final_embedding = head_embedding + relation_embedding + tail_embedding
        else:
            raise ValueError(
                f"fuse_information should be one of 'concat' or 'sum', but "
                f"{self.fuse_information} was given!"
            )

        if self.include_positional_encoding:
            final_embedding += self.positional_encoding[mem[3]]

        return final_embedding

    def create_batch(self, x: list[list[list]], memory_type: str) -> torch.Tensor:
        """Create one batch from data.

        Args:
            x: a batch of episodic, semantic, or short memories.
            memory_type: "episodic", "semantic", or "short"

        Returns:
            batch of embeddings.

        """
        mem_pad = ["<PAD>", "<PAD>", "<PAD>", "<PAD>"]

        for mems in x:
            for _ in range(self.capacity[memory_type] - len(mems)):
                # this is a dummy entry for padding.
                mems.append(mem_pad)
        batch_embeddings = []
        for mems in x:
            embeddings = []
            for mem in mems:
                mem_emb = self.make_embedding(mem, memory_type)
                embeddings.append(mem_emb)
            embeddings = torch.stack(embeddings)
            batch_embeddings.append(embeddings)

        batch_embeddings = torch.stack(batch_embeddings)

        return batch_embeddings

    def forward(self, x_: np.ndarray) -> torch.Tensor:
        """Forward-pass.

        Note that before we make a forward pass, argument x_ will be deepcopied. This
        is because we will modify x_ in the forward pass, and we don't want to modify
        the original x_. This slows down the process, but it's necessary.

        Args:
            x: a batch of memories. Each element of the batch is a np.ndarray of dict
            memories. x being a np.ndarray speeds up the process.

        Returns:
            Q-values, (action, distribution), or value.

        """
        x = deepcopy(x_)
        assert isinstance(x, np.ndarray)
        to_concat = []
        if "episodic" in self.memory_of_interest:
            batch_e = [sample["episodic"] for sample in x]  # sample is a dict
            batch_e = self.create_batch(batch_e, memory_type="episodic")
            lstm_out_e, _ = self.lstm_e(batch_e)

            if self.fuse_information == "concat":
                fc_out_e = self.relu(
                    self.fc_e1(self.relu(self.fc_e0(lstm_out_e[:, -1, :])))
                )
                to_concat.append(fc_out_e)
            else:
                to_concat.append(lstm_out_e[:, -1, :])

        if "episodic_agent" in self.memory_of_interest:
            batch_e_agent = [sample["episodic_agent"] for sample in x]
            batch_e_agent = self.create_batch(
                batch_e_agent, memory_type="episodic_agent"
            )
            lstm_out_e_agent, _ = self.lstm_e_agent(batch_e_agent)

            if self.fuse_information == "concat":
                fc_out_e_agent = self.relu(
                    self.fc_e1_agent(
                        self.relu(self.fc_e0_agent(lstm_out_e_agent[:, -1, :]))
                    )
                )
                to_concat.append(fc_out_e_agent)
            else:
                to_concat.append(lstm_out_e_agent[:, -1, :])

        if "semantic" in self.memory_of_interest:
            batch_s = [sample["semantic"] for sample in x]
            batch_s = self.create_batch(batch_s, memory_type="semantic")
            lstm_out_s, _ = self.lstm_s(batch_s)

            if self.fuse_information == "concat":
                fc_out_s = self.relu(
                    self.fc_s1(self.relu(self.fc_s0(lstm_out_s[:, -1, :])))
                )
                to_concat.append(fc_out_s)
            else:
                to_concat.append(lstm_out_s[:, -1, :])

        if "semantic_map" in self.memory_of_interest:
            batch_s_map = [sample["semantic_map"] for sample in x]
            batch_s_map = self.create_batch(batch_s_map, memory_type="semantic_map")
            lstm_out_s_map, _ = self.lstm_s_map(batch_s_map)

            if self.fuse_information == "concat":
                fc_out_s_map = self.relu(
                    self.fc_s1_map(self.relu(self.fc_s0_map(lstm_out_s_map[:, -1, :])))
                )
                to_concat.append(fc_out_s_map)
            else:
                to_concat.append(lstm_out_s_map[:, -1, :])

        if "short" in self.memory_of_interest:
            batch_o = [sample["short"] for sample in x]
            batch_o = self.create_batch(batch_o, memory_type="short")
            lstm_out_o, _ = self.lstm_o(batch_o)

            if self.fuse_information == "concat":
                fc_out_o = self.relu(
                    self.fc_o1(self.relu(self.fc_o0(lstm_out_o[:, -1, :])))
                )
                to_concat.append(fc_out_o)
            else:
                to_concat.append(lstm_out_o[:, -1, :])

        if self.fuse_information == "concat":
            fc_out_all = torch.concat(to_concat, dim=-1)
        elif self.fuse_information == "sum":
            fc_out_all = torch.sum(torch.stack(to_concat), dim=0)
        else:
            raise ValueError(
                f"fuse_information should be one of 'concat' or 'sum', but "
                f"{self.fuse_information} was given!"
            )
        if self.dueling_dqn:
            value = self.value_layer(fc_out_all)
            advantage = self.advantage_layer(fc_out_all)
            q = value + advantage - advantage.mean(dim=-1, keepdim=True)
        else:
            q = self.advantage_layer(fc_out_all)

        return q
