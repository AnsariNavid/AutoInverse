import json
import os
from typing import Tuple

import scipy.io
import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import TensorDataset, random_split


def save_mat_file(
            data: dict,
            path: str,
        ) -> None:

    scipy.io.savemat(path, data)


def load_mat_file(
            path: str
        ) -> dict:

    return scipy.io.loadmat(path)


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

    save_mat_file(data={key: tensor.detach().to('cpu').numpy()}, path=path)


def load_tensor_from_mat_file(
            path: str,
            key: str = 'data'
        ) -> Tensor:

    return torch.from_numpy(load_mat_file(path=path)[key]).float()


def save_json_file(
            data: dict,
            path: str
        ) -> None:

    with open(path, 'w') as f:
        json.dump(data, f, sort_keys=True, indent=4)


def load_json_file(
            path: str
        ) -> dict:

    with open(path, 'r') as f:
        data = json.load(f)
    return data


def get_train_test_tensor_datasets(
            tensor_dataset: TensorDataset,
            train_split_ratio: float
        ) -> Tuple[TensorDataset, TensorDataset]:

    train_size = int(len(tensor_dataset) * train_split_ratio)
    test_size = len(tensor_dataset) - train_size
    train_tensor_dataset, test_tensor_dataset = random_split(tensor_dataset, [train_size, test_size])
    return train_tensor_dataset, test_tensor_dataset


def create_directory(
            directory: str
        ) -> None:

    if not os.path.exists(directory):
        os.makedirs(directory)


def get_uana_mu_model_directory(
            results_base_path: str,
            experiment_name: str,
            config_name: str,
            model_id: int
        ) -> str:

    return os.path.join(results_base_path, experiment_name, config_name, "uana", "mu_model", str(model_id))


def get_uana_sigma_model_directory(
            results_base_path: str,
            experiment_name: str,
            config_name: str,
            model_id: int
        ) -> str:

    return os.path.join(results_base_path, experiment_name, config_name, "uana", "sigma_model", str(model_id))


def get_uana_inversion_directory(
            results_base_path: str,
            experiment_name: str,
            config_name: str,
            epistemic_weight: float,
            aleatory_weight: float,
            experiment_repetition_id: int,
            inversion_repetition_id: int
        ) -> str:

    return os.path.join(
        results_base_path,
        experiment_name,
        config_name,
        "uana",
        "inversion",
        f"{str(epistemic_weight)}_{str(aleatory_weight)}",
        str(experiment_repetition_id),
        str(inversion_repetition_id)
    )


def get_na_forward_model_directory(
            results_base_path: str,
            experiment_name: str,
            config_name: str
        ) -> str:

    return os.path.join(results_base_path, experiment_name, config_name, "na", "forward_model")


def get_na_inversion_directory(
            results_base_path: str,
            experiment_name: str,
            config_name: str,
            experiment_repetition_id: int,
            inversion_repetition_id: int
        ) -> str:

    return os.path.join(
        results_base_path,
        experiment_name,
        config_name,
        "na",
        "inversion",
        str(experiment_repetition_id),
        str(inversion_repetition_id)
    )


def get_nae_inversion_directory(
            results_base_path: str,
            experiment_name: str,
            config_name: str,
            experiment_repetition_id: int,
            inversion_repetition_id: int
        ) -> str:

    return os.path.join(
        results_base_path,
        experiment_name,
        config_name,
        "nae",
        "inversion",
        str(experiment_repetition_id),
        str(inversion_repetition_id)
    )


def get_tandem_backward_model_directory(
            results_base_path: str,
            experiment_name: str,
            config_name: str,
            experiment_repetition_id: int
        ) -> str:

    return os.path.join(
        results_base_path,
        experiment_name,
        config_name,
        "tandem",
        "backward_model",
        str(experiment_repetition_id)
    )


def get_tandem_inversion_directory(
            results_base_path: str,
            experiment_name: str,
            config_name: str,
            experiment_repetition_id: int
        ) -> str:

    return os.path.join(
        results_base_path,
        experiment_name,
        config_name,
        "tandem",
        "inversion",
        str(experiment_repetition_id)
    )


def get_ua_tandem_backward_model_directory(
            results_base_path: str,
            experiment_name: str,
            config_name: str,
            epistemic_weight: float,
            aleatory_weight: float,
            experiment_repetition_id: int
        ) -> str:

    return os.path.join(
        results_base_path,
        experiment_name,
        config_name,
        "ua_tandem",
        "backward_model",
        f"{str(epistemic_weight)}_{str(aleatory_weight)}",
        str(experiment_repetition_id)
    )


def get_ua_tandem_inversion_directory(
            results_base_path: str,
            experiment_name: str,
            config_name: str,
            epistemic_weight: float,
            aleatory_weight: float,
            experiment_repetition_id: int
        ) -> str:

    return os.path.join(
        results_base_path,
        experiment_name,
        config_name,
        "ua_tandem",
        "inversion",
        f"{str(epistemic_weight)}_{str(aleatory_weight)}",
        str(experiment_repetition_id)
    )
