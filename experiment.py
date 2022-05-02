import argparse
import os
import time
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim

import src


def train_mu_model(
            config: dict,
            model_id: int
        ) -> None:

    mu_model = src.model.create_mu_model(
        layers_size=[config['input_size']] + config['mu_model']['hidden_layers_size'] + [config['output_size']],
        activation_function=getattr(nn, config['mu_model']['activation_functions'][model_id])()
    )

    train_tensor_dataset, test_tensor_dataset = src.dataset.get_train_test_tensor_datasets(
        tensor_dataset=src.dataset.create_tensor_dataset(
            src.dataset.load_tensor_from_mat_file(path=config['input_data_path']),
            src.dataset.load_tensor_from_mat_file(path=config['output_data_path'])
        ),
        train_split_ratio=config['mu_model']['train_split_ratio']
    )
    train_data_loader = src.dataset.create_data_loader(
        tensor_dataset=train_tensor_dataset,
        batch_size=config['mu_model']['batch_size'],
        shuffle=True,
        pin_memory=True
    )
    test_data_loader = src.dataset.create_data_loader(
        tensor_dataset=test_tensor_dataset,
        batch_size=config['mu_model']['batch_size'],
        shuffle=True,
        pin_memory=True
    )

    optimizer = optim.Adam(mu_model.parameters(), lr=config['mu_model']['learning_rate'])
    criterion = nn.MSELoss()
    learning_rate_scheduler = optim.lr_scheduler.ExponentialLR(
        optimizer=optimizer,
        gamma=config['mu_model']['learning_rate_decay_rate']
    )

    start_time = time.time()
    loss = src.train.train_mu_model(
        mu_model=mu_model,
        criterion=criterion,
        optimizer=optimizer,
        learning_rate_scheduler=learning_rate_scheduler,
        train_data_loader=train_data_loader,
        test_data_loader=test_data_loader,
        number_of_epochs=config['mu_model']['number_of_epochs']
    )
    training_time = time.time() - start_time
    number_of_trainable_parameters = int(sum(p.numel() for p in mu_model.parameters() if p.requires_grad))

    data = {
        "training_time": training_time,
        "number_of_trainable_parameters": number_of_trainable_parameters,
        "loss": loss
    }

    root_path = src.utils.get_mu_model_directory(
        results_base_path=config['results_base_path'],
        experiment_name=config['experiment_name'],
        config_name=config['config_name'],
        model_id=model_id
    )
    src.utils.create_directory(directory=root_path)

    src.utils.save_data_to_json_file(data=data, path=os.path.join(root_path, 'info.json'))
    src.utils.save_model(model=mu_model, path=os.path.join(root_path, f'model_{model_id}.pt'))


def train_sigma_model(
            config: dict,
            model_id: int
        ) -> None:

    mu_model = src.model.create_mu_model(
        layers_size=[config['input_size']] + config['mu_model']['hidden_layers_size'] + [config['output_size']],
        activation_function=getattr(nn, config['mu_model']['activation_functions'][model_id])()
    )
    mu_model_root_path = src.utils.get_mu_model_directory(
        results_base_path=config['results_base_path'],
        experiment_name=config['experiment_name'],
        config_name=config['config_name'],
        model_id=model_id
    )
    mu_model = src.utils.load_model(model=mu_model, path=os.path.join(mu_model_root_path, f'model_{model_id}.pt'))

    sigma_model = src.model.create_sigma_model(
        layers_size=[config['input_size']] + config['sigma_model']['hidden_layers_size'] + [config['output_size']],
    )

    train_tensor_dataset, test_tensor_dataset = src.dataset.get_train_test_tensor_datasets(
        tensor_dataset=src.dataset.create_tensor_dataset(
            src.dataset.load_tensor_from_mat_file(path=config['input_data_path']),
            src.dataset.load_tensor_from_mat_file(path=config['output_data_path'])
        ),
        train_split_ratio=config['sigma_model']['train_split_ratio']
    )
    train_data_loader = src.dataset.create_data_loader(
        tensor_dataset=train_tensor_dataset,
        batch_size=config['sigma_model']['batch_size'],
        shuffle=True,
        pin_memory=True
    )
    test_data_loader = src.dataset.create_data_loader(
        tensor_dataset=test_tensor_dataset,
        batch_size=config['sigma_model']['batch_size'],
        shuffle=True,
        pin_memory=True
    )

    sigma_optimizer = optim.Adam(sigma_model.parameters(), lr=config['sigma_model']['learning_rate'])
    mu_optimizer = optim.Adam(mu_model.parameters(), lr=config['sigma_model']['learning_rate'])
    sigma_criterion = src.loss.custom_negative_log_likelihood_loss
    mu_criterion = nn.MSELoss()
    sigma_learning_rate_scheduler = optim.lr_scheduler.ExponentialLR(
        optimizer=sigma_optimizer,
        gamma=config['sigma_model']['learning_rate_decay_rate']
    )
    mu_learning_rate_scheduler = optim.lr_scheduler.ExponentialLR(
        optimizer=mu_optimizer,
        gamma=config['sigma_model']['learning_rate_decay_rate']
    )

    start_time = time.time()
    loss = src.train.train_sigma_model(
        sigma_model=sigma_model,
        mu_model=mu_model,
        sigma_criterion=sigma_criterion,
        mu_criterion=mu_criterion,
        sigma_optimizer=sigma_optimizer,
        mu_optimizer=mu_optimizer,
        sigma_learning_rate_scheduler=sigma_learning_rate_scheduler,
        mu_learning_rate_scheduler=mu_learning_rate_scheduler,
        train_data_loader=train_data_loader,
        test_data_loader=test_data_loader,
        number_of_epochs=config['sigma_model']['number_of_epochs']
    )
    training_time = time.time() - start_time
    number_of_trainable_parameters = int(sum(p.numel() for p in sigma_model.parameters() if p.requires_grad))

    data = {
        "training_time": training_time,
        "number_of_trainable_parameters": number_of_trainable_parameters,
        "loss": loss
    }

    root_path = src.utils.get_sigma_model_directory(
        results_base_path=config['results_base_path'],
        experiment_name=config['experiment_name'],
        config_name=config['config_name'],
        model_id=model_id
    )
    src.utils.create_directory(directory=root_path)

    src.utils.save_data_to_json_file(data=data, path=os.path.join(root_path, 'info.json'))
    src.utils.save_model(model=sigma_model, path=os.path.join(root_path, f'model_{model_id}.pt'))
    src.utils.save_model(model=mu_model, path=os.path.join(mu_model_root_path, f'model_{model_id}_fine_tuned.pt'))


def inversion(
            config: dict,
            repetition_id: int
        ) -> None:

    mu_models = []
    sigma_models = []
    for model_id in range(len(config['mu_model']['activation_functions'])):
        mu_model = src.model.create_mu_model(
            layers_size=[config['input_size']] + config['mu_model']['hidden_layers_size'] + [config['output_size']],
            activation_function=getattr(nn, config['mu_model']['activation_functions'][model_id])()
        )
        mu_model_root_path = src.utils.get_mu_model_directory(
            results_base_path=config['results_base_path'],
            experiment_name=config['experiment_name'],
            config_name=config['config_name'],
            model_id=model_id
        )
        mu_model = src.utils.load_model(
            model=mu_model,
            path=os.path.join(mu_model_root_path, f'model_{model_id}_fine_tuned.pt')
        )
        mu_model.eval()
        mu_models.append(mu_model)

        sigma_model = src.model.create_sigma_model(
            layers_size=[config['input_size']] + config['sigma_model']['hidden_layers_size'] + [config['output_size']],
        )
        sigma_model_root_path = src.utils.get_sigma_model_directory(
            results_base_path=config['results_base_path'],
            experiment_name=config['experiment_name'],
            config_name=config['config_name'],
            model_id=model_id
        )
        sigma_model = src.utils.load_model(
            model=sigma_model,
            path=os.path.join(sigma_model_root_path, f'model_{model_id}.pt')
        )
        sigma_model.eval()
        sigma_models.append(sigma_model)

    target = src.dataset.load_tensor_from_mat_file(path=config['target_data_path']).to(src.settings.DEVICE)
    design = torch.rand(
        target.shape[0], config['input_size']
    ).type(torch.float).to(src.settings.DEVICE).clone().detach().requires_grad_(True)

    optimizer = optim.Adam([design], lr=config['inversion']['learning_rate'])

    start_time = time.time()
    loss, design, mu_mean, sigma_epistemic, sigma_aleatory, no_reduction_loss = src.inversion.converge_design(
        sigma_models=sigma_models,
        mu_models=mu_models,
        epistemic_weight=config['inversion']['epistemic_weight'],
        aleatory_weight=config['inversion']['aleatory_weight'],
        criterion=getattr(src.loss, config['inversion']['criterion']),
        no_reduction_criterion=getattr(src.loss, config['inversion']['no_reduction_criterion']),
        optimizer=optimizer,
        design=design,
        target=target,
        number_of_iterations=config['inversion']['number_of_iterations'],
        minimum_convergence_length=config['inversion']['minimum_convergence_length'],
        convergence_threshold=config['inversion']['convergence_threshold']
    )
    training_time = time.time() - start_time

    data = {
        "inversion_time": training_time,
        "loss": loss
    }

    root_path = src.utils.get_inversion_directory(
        results_base_path=config['results_base_path'],
        experiment_name=config['experiment_name'],
        config_name=config['config_name'],
        repetition_id=repetition_id
    )
    src.utils.create_directory(directory=root_path)

    src.utils.save_data_to_json_file(data=data, path=os.path.join(root_path, 'info.json'))
    src.utils.save_tensor_to_mat_file(tensor=design, path=os.path.join(root_path, 'design.mat'))
    src.utils.save_tensor_to_mat_file(tensor=mu_mean, path=os.path.join(root_path, 'performance.mat'))
    src.utils.save_tensor_to_mat_file(tensor=sigma_epistemic, path=os.path.join(root_path, 'sigma_epistemic.mat'))
    src.utils.save_tensor_to_mat_file(tensor=sigma_aleatory, path=os.path.join(root_path, 'sigma_aleatory.mat'))
    src.utils.save_tensor_to_mat_file(tensor=no_reduction_loss, path=os.path.join(root_path, 'no_reduction_loss.mat'))


def pars_arguments() -> Any:
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('-m', '--method', type=str, required=True,
                        choices=['train_mu_model', 'train_sigma_model', 'inversion'])
    parser.add_argument('-n', '--model_id', type=int, required=False)
    parser.add_argument('-r', '--repetition_id', type=int, required=False)
    args = parser.parse_args()
    if (args.method in ['train_mu_model', 'train_sigma_model']) and (args.model_id is None):
        parser.error("'train_mu_model' and 'train_sigma_model' require --model_id.")
    if (args.method in ['inversion']) and (args.repetition_id is None):
        parser.error("'inversion' requires --repetition_id.")
    return args


if __name__ == '__main__':
    args = pars_arguments()
    config = src.utils.load_data_from_json_file(path=args.config)

    if args.method == "train_mu_model":
        train_mu_model(config=config, model_id=args.model_id)
    elif args.method == "train_sigma_model":
        train_sigma_model(config=config, model_id=args.model_id)
    elif args.method == "inversion":
        inversion(config=config, repetition_id=args.repetition_id)
