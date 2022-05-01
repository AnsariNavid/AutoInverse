import json
import os

import scipy.io
import torch
from torch import Tensor
from torch.nn import Module


def save_model(
            model: Module,
            path: str
        ) -> None:

    torch.save(model.state_dict(), path)


def load_model(
            model: Module,
            path: str
        ) -> Module:

    model.load_state_dict(torch.load(path))
    return model


def save_tensor_to_mat_file(
            tensor: Tensor,
            path: str,
            key: str = 'data'
        ) -> None:

    scipy.io.savemat(path, {key: tensor.detach().to('cpu').numpy()})


def save_data_to_json_file(
            data: dict,
            path: str
        ) -> None:

    with open(path, 'w') as fp:
        json.dump(data, fp, sort_keys=True, indent=4)


def load_data_from_json_file(
            path: str
        ) -> dict:

    with open(path, 'r') as fp:
        data = json.load(fp)
    return data


def create_directory(
            directory: str
        ) -> None:

    if not os.path.exists(directory):
        os.makedirs(directory)


def get_mu_model_directory(
            results_base_path: str,
            experiment_name: str,
            config_name: str,
            model_id: int
        ) -> str:

    return os.path.join(results_base_path, experiment_name, config_name, "mu_model", str(model_id))


def get_sigma_model_directory(
            results_base_path: str,
            experiment_name: str,
            config_name: str,
            model_id: int
        ) -> str:

    return os.path.join(results_base_path, experiment_name, config_name, "sigma_model", str(model_id))


def get_inversion_directory(
            results_base_path: str,
            experiment_name: str,
            config_name: str,
            repetition_id: int
        ) -> str:

    return os.path.join(results_base_path, experiment_name, config_name, "inversion", str(repetition_id))
