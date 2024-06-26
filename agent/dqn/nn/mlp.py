import torch


class MLP(torch.nn.Module):
    """Multi-layer perceptron with ReLU activation functions."""

    def __init__(
        self, n_actions: int, hidden_size: int, device: str, dueling_dqn: bool = True
    ) -> None:
        """Initialize the MLP.

        Args:
            n_actions: Number of actions.
            hidden_size: Hidden size of the linear layer.
            device: "cpu" or "cuda".
            dueling_dqn: Whether to use dueling DQN.

        """
        super(MLP, self).__init__()
        self.device = device

        self.hidden_size = hidden_size
        self.n_actions = n_actions
        self.dueling_dqn = dueling_dqn

        self.advantage_layer = torch.nn.Sequential(
            torch.nn.Linear(
                self.hidden_size,
                self.hidden_size,
                device=self.device,
            ),
            torch.nn.ReLU(),
            torch.nn.Linear(
                self.hidden_size,
                self.n_actions,
                device=self.device,
            ),
        )

        if self.dueling_dqn:
            self.value_layer = torch.nn.Sequential(
                torch.nn.Linear(
                    self.hidden_size,
                    self.hidden_size,
                    device=self.device,
                ),
                torch.nn.ReLU(),
                torch.nn.Linear(
                    self.hidden_size,
                    1,
                    device=self.device,
                ),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the neural network.

        Args:
            x: Input tensor. Shape (batch_size, lstm_hidden_size).
        Returns:
            torch.Tensor: Output tensor.

        """

        if self.dueling_dqn:
            value = self.value_layer(x)
            advantage = self.advantage_layer(x)
            q = value + advantage - advantage.mean(dim=-1, keepdim=True)
        else:
            q = self.advantage_layer(x)

        return q
