import os
from typing import List

import torch.nn as nn
from torch.nn import Module

from .. import settings, utils


def create_backward_model(
            layers_size: List[int]
        ) -> Module:

    layers = []

    for i in range(0, len(layers_size) - 1):
        layers.append(nn.Linear(in_features=layers_size[i], out_features=layers_size[i + 1], bias=True))
        if i < len(layers_size) - 2:
            layers.append(nn.ReLU())

    return nn.Sequential(*layers).to(settings.DEVICE)


def load_backward_model(
            config: dict,
            experiment_repetition_id: int
        ) -> Module:

    backward_model = create_backward_model(
        layers_size=[config['performance_size']] + config['tandem']['backward_model']['hidden_layers_size'] +
                    [config['design_size']]
    )
    backward_model_root_path = utils.get_tandem_backward_model_directory(
        results_base_path=config['results_base_path'],
        experiment_name=config['experiment_name'],
        config_name=config['config_name'],
        experiment_repetition_id=experiment_repetition_id
    )
    backward_model = utils.load_model(
        model=backward_model,
        path=os.path.join(backward_model_root_path, 'model.pt')
    )
    backward_model.eval()

    return backward_model
