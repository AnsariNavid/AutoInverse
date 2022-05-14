import os
from collections import OrderedDict

import numpy
import torch
import torch.nn as nn
from torch import Tensor

from .. import utils


def _get_oracle_performances(
            designs: numpy.ndarray
        ) -> Tensor:

    forward_models = []
    for model_id in range(20):
        forward_model = nn.Sequential(
            nn.Linear(in_features=8, out_features=100, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=250, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=250, out_features=250, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=250, out_features=100, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=31, bias=True),
        )
        custom_state_dict = torch.load(
            os.path.join('spectral_printing_oracle_models', f'model_{model_id}.ckpt'),
            map_location=torch.device('cpu')
        )
        default_state_dict = OrderedDict()
        default_state_dict['0.weight'] = custom_state_dict['layers.0.weight']
        default_state_dict['0.bias'] = custom_state_dict['layers.0.bias']
        default_state_dict['2.weight'] = custom_state_dict['layers.1.weight']
        default_state_dict['2.bias'] = custom_state_dict['layers.1.bias']
        default_state_dict['4.weight'] = custom_state_dict['layers.2.weight']
        default_state_dict['4.bias'] = custom_state_dict['layers.2.bias']
        default_state_dict['6.weight'] = custom_state_dict['layers.3.weight']
        default_state_dict['6.bias'] = custom_state_dict['layers.3.bias']
        default_state_dict['8.weight'] = custom_state_dict['layers.4.weight']
        default_state_dict['8.bias'] = custom_state_dict['layers.4.bias']
        forward_model.load_state_dict(default_state_dict)
        forward_model.eval()
        forward_models.append(forward_model)

    batch_size = designs.shape[0]
    designs = torch.from_numpy(designs).float()
    oracle_performances = torch.empty(batch_size, 31)

    for element_id in range(batch_size):

        ensembled_outputs = torch.empty(20, 31)
        for model_id in range(20):
            ensembled_outputs[model_id, :] = forward_models[model_id](designs[element_id])

        oracle_performances[element_id, :] = (1 / 20) * torch.sum(ensembled_outputs, 0)

    return oracle_performances.detach().clone()


def _mse(
            oracle_performances: numpy.ndarray,
            targets: numpy.ndarray
        ) -> float:

    return (numpy.square(targets - oracle_performances)).mean()


def _ink_density(
            designs: numpy.ndarray
        ) -> dict:

    filtered_designs = designs >= 0.4
    sum = numpy.sum(filtered_designs, axis=0)
    return {
        "Y": int(sum[0]),
        "LK": int(sum[1]),
        "LM": int(sum[2]),
        "LC": int(sum[3]),
        "LLK": int(sum[4]),
        "M": int(sum[5]),
        "C": int(sum[6]),
        "K": int(sum[7])
    }


def validate(
            config: dict,
            root_path: str
        ) -> None:

    designs = utils.load_mat_file(path=os.path.join(root_path, 'design.mat'))['data']
    designs = numpy.clip(designs, a_min=0.0, a_max=1.0)
    targets = utils.load_mat_file(path=config['target_data_path'])['data']
    oracle_performances = _get_oracle_performances(designs=designs)

    ink_density = _ink_density(designs=designs)
    mse = _mse(oracle_performances=oracle_performances.detach().to('cpu').numpy(), targets=targets)

    data = {
        "ink_density": ink_density,
        "mse": mse
    }

    utils.save_json_file(data=data, path=os.path.join(root_path, 'validation_results.json'))
    utils.save_tensor_to_mat_file(tensor=oracle_performances, path=os.path.join(root_path, 'oracle_performance.mat'))
