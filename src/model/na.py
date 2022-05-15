import os
from typing import List

import torch.nn as nn
from torch.nn import Module

from .. import settings, utils


def create_forward_model(
            layers_size: List[int]
        ) -> Module:

    layers = []

    for i in range(0, len(layers_size) - 1):
        layers.append(nn.Linear(in_features=layers_size[i], out_features=layers_size[i + 1], bias=True))
        if i < len(layers_size) - 2:
            layers.append(nn.ReLU())

    return nn.Sequential(*layers).to(settings.DEVICE)


def load_forward_model(
            config: dict
        ) -> Module:

    forward_model = create_forward_model(
        layers_size=[config['design_size']] + config['na']['forward_model']['hidden_layers_size'] +
                    [config['performance_size']]
    )
    forward_model_root_path = utils.get_na_forward_model_directory(
        results_base_path=config['results_base_path'],
        experiment_name=config['experiment_name'],
        config_name=config['config_name']
    )
    forward_model = utils.load_model(
        model=forward_model,
        path=os.path.join(forward_model_root_path, 'model.pt')
    )
    forward_model.eval()

    return forward_model
