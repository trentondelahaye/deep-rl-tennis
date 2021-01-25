from typing import Callable, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

LayerSize = int


def hidden_init(layer) -> Tuple[float, float]:
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return -lim, lim


class NeuralNetwork(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        fc_layers: Iterable[LayerSize],
        final_activation: Optional[Callable],
    ):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(input_size)
        self.final_activation = final_activation
        layer_sizes = [input_size] + list(fc_layers) + [output_size]
        self.fc_layers = []

        for i, (layer_input_size, layer_output_size) in enumerate(
            zip(layer_sizes, layer_sizes[1:])
        ):
            layer = nn.Linear(layer_input_size, layer_output_size)
            setattr(self, f"fc{i}", layer)
            self.fc_layers.append(layer)
        self.reset_parameters()

    @property
    def last_layer(self) -> nn.Linear:
        return self.fc_layers[-1]

    def reset_parameters(self) -> None:
        for layer in self.fc_layers[:-1]:
            layer.weight.data.uniform_(*hidden_init(layer))
        self.last_layer.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.batch_norm(x)
        for layer in self.fc_layers[:-1]:
            x = F.leaky_relu(layer(x))
        if self.final_activation:
            return self.final_activation(self.last_layer(x))
        else:
            return self.last_layer(x)


class Actor(NeuralNetwork):
    def __init__(
        self,
        state_size: int,
        action_size: int,
        fc_layers: Iterable[LayerSize] = (64, 64),
    ):
        super().__init__(state_size, action_size, fc_layers, F.tanh)


class Critic(nn.Module):
    def __init__(
        self,
        state_size: int,
        action_size: int,
        fc_layers: Iterable[LayerSize] = (64, 64),
    ):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(state_size)
        layer_sizes = [state_size] + list(fc_layers) + [1]
        self.fc_layers = []

        for i, (layer_input_size, layer_output_size) in enumerate(
            zip(layer_sizes, layer_sizes[1:])
        ):
            if i == 1:
                layer_input_size += action_size
            layer = nn.Linear(layer_input_size, layer_output_size)
            setattr(self, f"fc{i}", layer)
            self.fc_layers.append(layer)
        self.reset_parameters()

    @property
    def last_layer(self) -> nn.Linear:
        return self.fc_layers[-1]

    def reset_parameters(self) -> None:
        for layer in self.fc_layers[:-1]:
            layer.weight.data.uniform_(*hidden_init(layer))
        self.last_layer.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        x = self.batch_norm(states)
        x = F.leaky_relu(self.fc_layers[0](x))
        x = torch.cat((x, actions), dim=1)
        for layer in self.fc_layers[1:-1]:
            x = F.leaky_relu(layer(x))
        return self.last_layer(x)
