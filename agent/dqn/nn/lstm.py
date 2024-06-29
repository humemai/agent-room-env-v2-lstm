"""LSTM to approximate a function."""

from copy import deepcopy
from typing import Literal

import numpy as np
import torch
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
        embeddings: learnable embeddings.
        short_term_scale: learnable scaling factor for short-term memory.
        episodic_scale: learnable scaling factor for episodic memory.
        semantic_scale: learnable scaling factor for semantic memory.
        short_term_weight: learnable weight for short-term memory.
        episodic_weight: learnable weight for episodic memory.
        semantic_weight: learnable weight for semantic memory.

    """

    def __init__(
        self,
        capacity: dict,
        entities: list | dict,
        relations: list,
        hidden_size: int = 64,
        num_layers: int = 2,
        embedding_dim: int = 64,
        bidirectional: bool = False,
        device: str = "cpu",
        max_timesteps: int | None = None,
        max_strength: int | None = None,
    ) -> None:
        """Initialize the LSTM.

        Args:
            capacity: the capacities of memory systems. e.g., {"episodic": 12,
                "semantic": 12, "short": 1}
            entities: list of entities, e.g., ["Foo", "Bar", "laptop", "phone",
                "desk", "lap"]
            relations : list of relations, e.g., ["atlocation", "north", "south"]
            hidden_size: hidden size of the LSTM
            num_layers: number of the LSTM layers
            embedding_dim: entity embedding dimension (e.g., 32)
            bidirectional: whether the LSTM is bidirectional
            device: "cpu" or "cuda"
            max_timesteps: maximum number of timesteps.
            max_strength: maximum strength.

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

        self.create_embeddings()
        self.lstm = nn.LSTM(
            self.embedding_dim,
            self.hidden_size,
            self.num_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
            device=self.device,
        )

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

        self.embeddings = nn.Embedding(
            len(self.word2idx),
            self.embedding_dim,
            device=self.device,
            padding_idx=0,
        )

        # Learnable scaling factors
        self.short_term_scale = nn.Parameter(torch.tensor(1.0))
        self.episodic_scale = nn.Parameter(torch.tensor(1.0))
        self.semantic_scale = nn.Parameter(torch.tensor(1.0))

        # Learnable weights for memory types
        self.short_term_weight = nn.Parameter(torch.tensor(1.0))
        self.episodic_weight = nn.Parameter(torch.tensor(1.0))
        self.semantic_weight = nn.Parameter(torch.tensor(1.0))

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
            return self.embeddings(
                torch.tensor(self.word2idx["<PAD>"], device=self.device)
            )

        head_embedding = self.embeddings(
            torch.tensor(self.word2idx[mem[0]], device=self.device)
        )
        relation_embedding = self.embeddings(
            torch.tensor(self.word2idx[mem[1]], device=self.device)
        )
        tail_embedding = self.embeddings(
            torch.tensor(self.word2idx[mem[2]], device=self.device)
        )
        final_embedding = head_embedding + relation_embedding + tail_embedding

        num_normalized = mem[3] / self.max_fourth_val  # Normalize num to [0, 1]
        if memory_type == "short":
            time_embedding = self.embeddings(
                torch.tensor(self.word2idx["current_time"], device=self.device)
            )
            final_embedding += time_embedding * num_normalized * self.short_term_scale

        elif memory_type == "episodic":
            timestamp_embedding = self.embeddings(
                torch.tensor(self.word2idx["timestamp"], device=self.device)
            )
            final_embedding += (
                timestamp_embedding * num_normalized * self.episodic_scale
            )

        elif memory_type == "semantic":
            strength_embedding = self.embeddings(
                torch.tensor(self.word2idx["strength"], device=self.device)
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

    def forward(self, x_: np.ndarray, memory_types: list[str]) -> torch.Tensor:
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

        """
        x = deepcopy(x_)
        assert isinstance(x, np.ndarray)
        to_sum = []

        weights = {
            "short": self.short_term_weight,
            "episodic": self.episodic_weight,
            "semantic": self.semantic_weight,
        }

        for memory_type in memory_types:
            batch = [sample[memory_type] for sample in x]
            batch = self.create_batch(batch, memory_type=memory_type)
            lstm_out, _ = self.lstm(batch)
            weighted_lstm_out = lstm_out[:, -1, :] * weights[memory_type]
            to_sum.append(weighted_lstm_out)

        memory_representation = torch.sum(torch.stack(to_sum), dim=0)

        return memory_representation
