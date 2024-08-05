"""LSTM to approximate a function."""

from copy import deepcopy
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class LSTM(nn.Module):
    """A simple LSTM network.

    Attributes:
        capacity: the capacities of memory systems. e.g., {"episodic": 12,
            "semantic": 12, "short": 1}
        entities: list of entities, e.g., ["Foo", "Bar", "laptop", "phone",
            "desk", "lap"]
        relations: list of relations, e.g., ["atlocation", "north", "south"]
        hidden_size: hidden size of the LSTM
        num_layers: number of the LSTM layers
        embedding_dim: entity embedding dimension (e.g., 32)
        bidirectional: whether the LSTM is bidirectional
        device: "cpu" or "cuda"
        max_timesteps: maximum number of timesteps.
        max_strength: maximum strength.
        word2idx: dictionary that maps words to indices.
        embeddings: learnable embeddings or one-hot vectors.
        short_term_scale: learnable scaling factor for short-term memory.
        episodic_scale: learnable scaling factor for episodic memory.
        semantic_scale: learnable scaling factor for semantic memory.
        short_term_weight: learnable weight for short-term memory.
        episodic_weight: learnable weight for episodic memory.
        semantic_weight: learnable weight for semantic memory.
        use_one_hot: whether to use one-hot encoding instead of embeddings.

    """

    def __init__(
        self,
        capacity: dict,
        entities: list | dict,
        relations: list,
        num_layers: int = 2,
        embedding_dim: int = 64,
        hidden_size: int = 64,
        bidirectional: bool = False,
        device: str = "cpu",
        max_timesteps: int | None = None,
        max_strength: int | None = None,
        relu_for_attention: bool = True,
        use_one_hot: bool = False,
    ) -> None:
        """Initialize the LSTM.

        Args:
            capacity: the capacities of memory systems. e.g., {"episodic": 12,
                "semantic": 12, "short": 1}
            entities: list of entities, e.g., ["Foo", "Bar", "laptop", "phone",
                "desk", "lap"]
            relations : list of relations, e.g., ["atlocation", "north", "south"]
            num_layers: number of the LSTM layers
            embedding_dim: entity embedding dimension (e.g., 32)
            hidden_size: hidden size of the LSTM
            bidirectional: whether the LSTM is bidirectional
            device: "cpu" or "cuda"
            max_timesteps: maximum number of timesteps.
            max_strength: maximum strength.
            relu_for_attention: whether to apply non-linearity to the value
                matrix
            use_one_hot: whether to use one-hot encoding instead of embeddings

        """
        super().__init__()
        self.capacity = capacity
        self.entities = entities
        self.relations = relations
        self.embedding_dim = embedding_dim
        self.device = device
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.max_fourth_val = max(max_timesteps, max_strength)
        self.relu_for_attention = relu_for_attention
        self.use_one_hot = use_one_hot

        self.create_embeddings()
        self.lstm = nn.LSTM(
            self.embedding_dim if not use_one_hot else len(self.word2idx),
            self.hidden_size,
            self.num_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
            device=self.device,
        )

        # Learnable scaling factors
        self.short_term_scale = nn.Parameter(torch.tensor(1.0))
        self.episodic_scale = nn.Parameter(torch.tensor(1.0))
        self.semantic_scale = nn.Parameter(torch.tensor(1.0))

        # Define linear layers for query, key, and value matrices
        input_dim_attention = (
            self.hidden_size if not self.bidirectional else 2 * self.hidden_size
        )
        self.query_net = nn.Linear(input_dim_attention, self.hidden_size)
        self.key_net = nn.Linear(input_dim_attention, self.hidden_size)
        self.value_net = nn.Linear(input_dim_attention, self.hidden_size)

    def create_embeddings(self) -> None:
        """Create learnable embeddings."""

        if not (isinstance(self.entities, dict) or isinstance(self.entities, list)):
            raise ValueError(
                "entities should be either a list or a dictionary, but "
                f"{type(self.entities)} was given!"
            )

        self.word2idx = (
            ["<PAD>"]
            + (
                [name for names in self.entities.values() for name in names]
                if isinstance(self.entities, dict)
                else self.entities
            )
            + self.relations
            + ["current_time", "timestamp", "strength"]
        )

        self.word2idx = {word: idx for idx, word in enumerate(self.word2idx)}

        if not self.use_one_hot:
            self.embeddings = nn.Embedding(
                len(self.word2idx),
                self.embedding_dim,
                device=self.device,
                padding_idx=0,
            )

    def get_one_hot(self, idx: int, length: int) -> torch.Tensor:
        """Get one-hot encoding for a given index.

        Args:
            idx: index of the word
            length: length of the one-hot vector

        Returns:
            one-hot encoding of the given index.
        """
        one_hot = torch.zeros(length, device=self.device)
        one_hot[idx] = 1.0
        return one_hot

    def make_embedding(
        self, mem: list[str], memory_type: Literal["short", "episodic", "semantic"]
    ) -> torch.Tensor:
        """Create one embedding vector with summation.

        Args:
            mem: memory as a quadruple: [head, relation, tail, num]
            memory_type: "episodic", "semantic", or "short"

        Returns:
            one embedding vector made from one memory element.

        """
        if mem == ["<PAD>", "<PAD>", "<PAD>", "<PAD>"]:
            if self.use_one_hot:
                return self.get_one_hot(self.word2idx["<PAD>"], len(self.word2idx))
            return self.embeddings(
                torch.tensor(self.word2idx["<PAD>"], device=self.device)
            )

        head_embedding = (
            self.get_one_hot(self.word2idx[mem[0]], len(self.word2idx))
            if self.use_one_hot
            else self.embeddings(
                torch.tensor(self.word2idx[mem[0]], device=self.device)
            )
        )
        relation_embedding = (
            self.get_one_hot(self.word2idx[mem[1]], len(self.word2idx))
            if self.use_one_hot
            else self.embeddings(
                torch.tensor(self.word2idx[mem[1]], device=self.device)
            )
        )
        tail_embedding = (
            self.get_one_hot(self.word2idx[mem[2]], len(self.word2idx))
            if self.use_one_hot
            else self.embeddings(
                torch.tensor(self.word2idx[mem[2]], device=self.device)
            )
        )
        final_embedding = head_embedding + relation_embedding + tail_embedding

        num_normalized = mem[3] / self.max_fourth_val  # Normalize num to [0, 1]
        if memory_type == "short":
            time_embedding = (
                self.get_one_hot(self.word2idx["current_time"], len(self.word2idx))
                if self.use_one_hot
                else self.embeddings(
                    torch.tensor(self.word2idx["current_time"], device=self.device)
                )
            )
            final_embedding += time_embedding * num_normalized * self.short_term_scale

        elif memory_type == "episodic":
            timestamp_embedding = (
                self.get_one_hot(self.word2idx["timestamp"], len(self.word2idx))
                if self.use_one_hot
                else self.embeddings(
                    torch.tensor(self.word2idx["timestamp"], device=self.device)
                )
            )
            final_embedding += (
                timestamp_embedding * num_normalized * self.episodic_scale
            )

        elif memory_type == "semantic":
            strength_embedding = (
                self.get_one_hot(self.word2idx["strength"], len(self.word2idx))
                if self.use_one_hot
                else self.embeddings(
                    torch.tensor(self.word2idx["strength"], device=self.device)
                )
            )
            final_embedding += strength_embedding * num_normalized * self.semantic_scale

        else:
            raise ValueError(
                f"memory_type should be either 'short', 'episodic', or 'semantic', "
                f"but {memory_type} was given!"
            )

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

    def forward(
        self, x_: np.ndarray, memory_types: list[str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward-pass.

        Note that before we make a forward pass, argument x_ will be deepcopied. This
        is because we will modify x_ in the forward pass, and we don't want to modify
        the original x_. This slows down the process, but it's necessary.

        Args:
            x: a batch of memories. Each element of the batch is a np.ndarray of dict
            memories. x being a np.ndarray speeds up the process.
            memory_types: e.g., ["episodic", "semantic", "short"]

        Returns:
            memory_representation: sum of the last hidden states of the LSTM. This is
                the output of the forward pass. The dimension is (batch_size,
                hidden_size)
            attention_weights: attention weights. The dimension is (batch_size,
                len(memory_types), len(memory_types))

        """
        x = deepcopy(x_)
        assert isinstance(x, np.ndarray)

        hidden_states = []

        for memory_type in memory_types:
            batch = [sample[memory_type] for sample in x]
            batch = self.create_batch(batch, memory_type=memory_type)
            lstm_out, _ = self.lstm(batch)
            lstm_last_hidden_state = lstm_out[:, -1, :]  # (batch_size, hidden_size)
            hidden_states.append(lstm_last_hidden_state)

        # Convert hidden_states into a torch.Tensor
        # (batch_size, num_memory_types, hidden_size)
        hidden_states = torch.stack(hidden_states, dim=1)

        # Apply key, query, and value networks
        # (batch_size, num_memory_types, hidden_size)
        keys = self.key_net(hidden_states)
        queries = self.query_net(hidden_states)
        if self.relu_for_attention:
            values = F.relu(self.value_net(hidden_states))
        else:
            values = self.value_net(hidden_states)

        # Compute attention weights using queries and keys, with scaling
        # (batch_size, num_memory_types, num_memory_types)
        attention_logits = torch.matmul(queries, keys.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(keys.shape[-1], dtype=torch.float32)
        )

        # Softmax over last dimension
        # (batch_size, num_memory_types, num_memory_types)
        attention_weights = F.softmax(attention_logits, dim=-1)
        attention_weights = attention_weights / attention_weights.shape[-2]

        # Weighted sum of values using attention weights
        # (batch_size, num_memory_types, hidden_size)
        weighted_sum = torch.matmul(attention_weights, values)

        # Summing over memory types to get the final memory representation
        # (batch_size, hidden_size)
        memory_representation = torch.sum(weighted_sum, dim=1)

        return memory_representation, attention_weights.detach().cpu()
