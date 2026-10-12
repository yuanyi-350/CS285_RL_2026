import itertools
from torch import nn
from torch.nn import functional as F

import numpy as np
import torch
from torch import distributions

from infrastructure import pytorch_util as ptu


class ValueCritic(nn.Module):
    """Value network, which takes an observation and outputs a value for that observation."""

    def __init__(
        self,
        ob_dim: int,
        n_layers: int,
        layer_size: int,
        learning_rate: float,
        optimizer="adam",
        optimizer_kwargs=None,
    ):
        super().__init__()

        self.network = ptu.build_mlp(
            input_size=ob_dim,
            output_size=1,
            n_layers=n_layers,
            size=layer_size,
        ).to(ptu.device)

        optimizer_classes = {"gd": torch.optim.SGD, "adam": torch.optim.Adam}
        self.optimizer = optimizer_classes[optimizer](
            self.parameters(), lr=learning_rate, **(optimizer_kwargs or {})
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # The network returns (batch_size, 1); each observation needs one
        # scalar value estimate, so remove only the final singleton dimension.
        return self.network(obs).squeeze(-1)

    def update(self, obs: np.ndarray, q_values: np.ndarray) -> dict:
        obs = ptu.from_numpy(obs)
        q_values = ptu.from_numpy(q_values)

        values = self(obs)
        loss = F.mse_loss(values, q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "Baseline Loss": loss.item(),
        }
