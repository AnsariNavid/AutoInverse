from typing import List

import torch.nn as nn
from torch.nn import Module

from .. import settings


def create_forward_model(
            layers_size: List[int],
            activation_function: Module
        ) -> Module:

    layers = []

    for i in range(0, len(layers_size) - 1):
        layers.append(nn.Linear(in_features=layers_size[i], out_features=layers_size[i + 1], bias=True))
        if i < len(layers_size) - 2:
            layers.append(activation_function)

    return nn.Sequential(*layers).to(settings.DEVICE)


def create_backward_model(
            layers_size: List[int],
            activation_function: Module
        ) -> Module:

    layers = []

    for i in range(0, len(layers_size) - 1):
        layers.append(nn.Linear(in_features=layers_size[i], out_features=layers_size[i + 1], bias=True))
        if i < len(layers_size) - 2:
            layers.append(activation_function)

    return nn.Sequential(*layers).to(settings.DEVICE)
