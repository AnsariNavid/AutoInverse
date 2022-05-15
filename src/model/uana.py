import os
from typing import List

import torch.nn as nn
from torch.nn import Module

from .. import settings, utils


def create_mu_model(
            layers_size: List[int],
            activation_function: Module
        ) -> Module:

    layers = []

    for i in range(0, len(layers_size) - 1):
        layers.append(nn.Linear(in_features=layers_size[i], out_features=layers_size[i + 1], bias=True))
        if i < len(layers_size) - 2:
            layers.append(activation_function)

    return nn.Sequential(*layers).to(settings.DEVICE)


def load_mu_model(
            config: dict,
            model_id: int,
            fine_tuned: bool
        ) -> Module:

    mu_model = create_mu_model(
        layers_size=[config['design_size']] + config['uana']['mu_model']['hidden_layers_size'] +
        [config['performance_size']],
        activation_function=getattr(nn, config['uana']['mu_model']['activation_functions'][model_id])()
    )
    mu_model_root_path = utils.get_uana_mu_model_directory(
        results_base_path=config['results_base_path'],
        experiment_name=config['experiment_name'],
        config_name=config['config_name'],
        model_id=model_id
    )
    path = None
    if fine_tuned is True:
        path = os.path.join(mu_model_root_path, f'model_{model_id}_fine_tuned.pt')
    else:
        path = os.path.join(mu_model_root_path, f'model_{model_id}.pt')

    mu_model = utils.load_model(model=mu_model, path=path)
    mu_model.eval()

    return mu_model


def load_mu_models(
            config: dict,
            fine_tuned: bool
        ) -> List[Module]:

    mu_models = []
    for model_id in range(len(config['uana']['mu_model']['activation_functions'])):
        mu_models.append(load_mu_model(config=config, model_id=model_id, fine_tuned=fine_tuned))

    return mu_models


def save_mu_model(
            mu_model: Module,
            config: dict,
            model_id: int,
            fine_tuned: bool
        ) -> None:

    mu_model_root_path = utils.get_uana_mu_model_directory(
        results_base_path=config['results_base_path'],
        experiment_name=config['experiment_name'],
        config_name=config['config_name'],
        model_id=model_id
    )
    utils.create_directory(directory=mu_model_root_path)

    path = None
    if fine_tuned is True:
        path = os.path.join(mu_model_root_path, f'model_{model_id}_fine_tuned.pt')
    else:
        path = os.path.join(mu_model_root_path, f'model_{model_id}.pt')

    utils.save_model(model=mu_model, path=path)


def save_mu_models(
            mu_models: List[Module],
            config: dict,
            fine_tuned: bool
        ) -> None:

    for model_id, mu_model in enumerate(mu_models):
        save_mu_model(mu_model=mu_model, config=config, model_id=model_id, fine_tuned=fine_tuned)


def create_sigma_model(
            layers_size: List[int]
        ) -> Module:

    layers = []

    for i in range(0, len(layers_size) - 1):
        layers.append(nn.Linear(in_features=layers_size[i], out_features=layers_size[i + 1], bias=True))
        if i < len(layers_size) - 2:
            layers.append(nn.ReLU())
        else:
            layers.append(nn.Softplus())

    return nn.Sequential(*layers).to(settings.DEVICE)


def load_sigma_model(
            config: dict,
            model_id: int
        ) -> Module:

    sigma_model = create_sigma_model(
        layers_size=[config['design_size']] + config['uana']['sigma_model']['hidden_layers_size'] +
                    [config['performance_size']],
    )
    sigma_model_root_path = utils.get_uana_sigma_model_directory(
        results_base_path=config['results_base_path'],
        experiment_name=config['experiment_name'],
        config_name=config['config_name'],
        model_id=model_id
    )
    sigma_model = utils.load_model(
        model=sigma_model,
        path=os.path.join(sigma_model_root_path, f'model_{model_id}.pt')
    )
    sigma_model.eval()

    return sigma_model


def load_sigma_models(
            config: dict
        ) -> List[Module]:

    sigma_models = []
    for model_id in range(len(config['uana']['mu_model']['activation_functions'])):
        sigma_models.append(load_sigma_model(config=config, model_id=model_id))

    return sigma_models


def save_sigma_model(
            sigma_model: Module,
            config: dict,
            model_id: int
        ) -> None:

    sigma_model_root_path = utils.get_uana_sigma_model_directory(
        results_base_path=config['results_base_path'],
        experiment_name=config['experiment_name'],
        config_name=config['config_name'],
        model_id=model_id
    )
    utils.create_directory(directory=sigma_model_root_path)

    utils.save_model(model=sigma_model, path=os.path.join(sigma_model_root_path, f'model_{model_id}.pt'))


def save_sigma_models(
            sigma_models: List[Module],
            config: dict
        ) -> None:

    for model_id, sigma_model in enumerate(sigma_models):
        save_sigma_model(sigma_model=sigma_model, config=config, model_id=model_id)
