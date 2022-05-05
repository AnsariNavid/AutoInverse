import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .. import inversion, loss, model, settings, train, utils


def train_mu_model(
            config: dict,
            model_id: int
        ) -> None:

    mu_model = model.uana.create_mu_model(
        layers_size=[config['input_size']] + config['uana']['mu_model']['hidden_layers_size'] + [config['output_size']],
        activation_function=getattr(nn, config['uana']['mu_model']['activation_functions'][model_id])()
    )

    train_tensor_dataset, test_tensor_dataset = utils.get_train_test_tensor_datasets(
        tensor_dataset=TensorDataset(
            utils.load_tensor_from_mat_file(path=config['input_data_path']),
            utils.load_tensor_from_mat_file(path=config['output_data_path'])
        ),
        train_split_ratio=config['uana']['mu_model']['train_split_ratio']
    )
    train_data_loader = DataLoader(
        dataset=train_tensor_dataset,
        batch_size=config['uana']['mu_model']['batch_size'],
        shuffle=True,
        pin_memory=True
    )
    test_data_loader = DataLoader(
        dataset=test_tensor_dataset,
        batch_size=config['uana']['mu_model']['batch_size'],
        shuffle=True,
        pin_memory=True
    )

    optimizer = optim.Adam(mu_model.parameters(), lr=config['uana']['mu_model']['learning_rate'])
    criterion = nn.MSELoss()
    learning_rate_scheduler = optim.lr_scheduler.ExponentialLR(
        optimizer=optimizer,
        gamma=config['uana']['mu_model']['learning_rate_decay_rate']
    )

    start_time = time.time()
    loss_value = train.uana.train_mu_model(
        mu_model=mu_model,
        criterion=criterion,
        optimizer=optimizer,
        learning_rate_scheduler=learning_rate_scheduler,
        train_data_loader=train_data_loader,
        test_data_loader=test_data_loader,
        number_of_epochs=config['uana']['mu_model']['number_of_epochs']
    )
    training_time = time.time() - start_time
    number_of_trainable_parameters = int(sum(p.numel() for p in mu_model.parameters() if p.requires_grad))

    data = {
        "training_time": training_time,
        "number_of_trainable_parameters": number_of_trainable_parameters,
        "loss": loss_value
    }

    root_path = utils.get_uana_mu_model_directory(
        results_base_path=config['results_base_path'],
        experiment_name=config['experiment_name'],
        config_name=config['config_name'],
        model_id=model_id
    )
    utils.create_directory(directory=root_path)

    utils.save_json_file(data=data, path=os.path.join(root_path, 'info.json'))
    utils.save_model(model=mu_model, path=os.path.join(root_path, f'model_{model_id}.pt'))


def train_sigma_model(
            config: dict,
            model_id: int
        ) -> None:

    mu_model = model.uana.create_mu_model(
        layers_size=[config['input_size']] + config['uana']['mu_model']['hidden_layers_size'] + [config['output_size']],
        activation_function=getattr(nn, config['uana']['mu_model']['activation_functions'][model_id])()
    )
    mu_model_root_path = utils.get_uana_mu_model_directory(
        results_base_path=config['results_base_path'],
        experiment_name=config['experiment_name'],
        config_name=config['config_name'],
        model_id=model_id
    )
    mu_model = utils.load_model(model=mu_model, path=os.path.join(mu_model_root_path, f'model_{model_id}.pt'))

    sigma_model = model.uana.create_sigma_model(
        layers_size=[config['input_size']] + config['uana']['sigma_model']['hidden_layers_size'] +
                    [config['output_size']],
    )

    train_tensor_dataset, test_tensor_dataset = utils.get_train_test_tensor_datasets(
        tensor_dataset=TensorDataset(
            utils.load_tensor_from_mat_file(path=config['input_data_path']),
            utils.load_tensor_from_mat_file(path=config['output_data_path'])
        ),
        train_split_ratio=config['uana']['sigma_model']['train_split_ratio']
    )
    train_data_loader = DataLoader(
        dataset=train_tensor_dataset,
        batch_size=config['uana']['sigma_model']['batch_size'],
        shuffle=True,
        pin_memory=True
    )
    test_data_loader = DataLoader(
        dataset=test_tensor_dataset,
        batch_size=config['uana']['sigma_model']['batch_size'],
        shuffle=True,
        pin_memory=True
    )

    sigma_optimizer = optim.Adam(sigma_model.parameters(), lr=config['uana']['sigma_model']['learning_rate'])
    mu_optimizer = optim.Adam(mu_model.parameters(), lr=config['uana']['sigma_model']['learning_rate'])
    sigma_criterion = loss.uana.custom_negative_log_likelihood_loss
    mu_criterion = nn.MSELoss()
    sigma_learning_rate_scheduler = optim.lr_scheduler.ExponentialLR(
        optimizer=sigma_optimizer,
        gamma=config['uana']['sigma_model']['learning_rate_decay_rate']
    )
    mu_learning_rate_scheduler = optim.lr_scheduler.ExponentialLR(
        optimizer=mu_optimizer,
        gamma=config['uana']['sigma_model']['learning_rate_decay_rate']
    )

    start_time = time.time()
    loss_value = train.uana.train_sigma_model(
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
        number_of_epochs=config['uana']['sigma_model']['number_of_epochs']
    )
    training_time = time.time() - start_time
    number_of_trainable_parameters = int(sum(p.numel() for p in sigma_model.parameters() if p.requires_grad))

    data = {
        "training_time": training_time,
        "number_of_trainable_parameters": number_of_trainable_parameters,
        "loss": loss_value
    }

    root_path = utils.get_uana_sigma_model_directory(
        results_base_path=config['results_base_path'],
        experiment_name=config['experiment_name'],
        config_name=config['config_name'],
        model_id=model_id
    )
    utils.create_directory(directory=root_path)

    utils.save_json_file(data=data, path=os.path.join(root_path, 'info.json'))
    utils.save_model(model=sigma_model, path=os.path.join(root_path, f'model_{model_id}.pt'))
    utils.save_model(model=mu_model, path=os.path.join(mu_model_root_path, f'model_{model_id}_fine_tuned.pt'))


def inverse_design(
            config: dict,
            experiment_repetition_id: int,
            inversion_repetition_id: int
        ) -> None:

    mu_models = []
    sigma_models = []
    for model_id in range(len(config['uana']['mu_model']['activation_functions'])):
        mu_model = model.uana.create_mu_model(
            layers_size=[config['input_size']] + config['uana']['mu_model']['hidden_layers_size'] +
            [config['output_size']],
            activation_function=getattr(nn, config['uana']['mu_model']['activation_functions'][model_id])()
        )
        mu_model_root_path = utils.get_uana_mu_model_directory(
            results_base_path=config['results_base_path'],
            experiment_name=config['experiment_name'],
            config_name=config['config_name'],
            model_id=model_id
        )
        mu_model = utils.load_model(
            model=mu_model,
            path=os.path.join(mu_model_root_path, f'model_{model_id}_fine_tuned.pt')
        )
        mu_model.eval()
        mu_models.append(mu_model)

        sigma_model = model.uana.create_sigma_model(
            layers_size=[config['input_size']] + config['uana']['sigma_model']['hidden_layers_size'] +
                        [config['output_size']],
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
        sigma_models.append(sigma_model)

    target = utils.load_tensor_from_mat_file(path=config['target_data_path']).to(settings.DEVICE)
    design = torch.rand(
        target.shape[0], config['input_size']
    ).type(torch.float).to(settings.DEVICE).clone().detach().requires_grad_(True)

    optimizer = optim.Adam([design], lr=config['uana']['inversion']['learning_rate'])

    start_time = time.time()
    loss_value, design, mu_mean, sigma_epistemic, sigma_aleatory, no_reduction_loss = inversion.uana.converge_design(
        sigma_models=sigma_models,
        mu_models=mu_models,
        epistemic_weight=config['uana']['inversion']['epistemic_weight'],
        aleatory_weight=config['uana']['inversion']['aleatory_weight'],
        criterion=getattr(loss, config['uana']['inversion']['criterion']),
        no_reduction_criterion=getattr(loss, config['uana']['inversion']['no_reduction_criterion']),
        optimizer=optimizer,
        design=design,
        target=target,
        number_of_iterations=config['uana']['inversion']['number_of_iterations'],
        minimum_convergence_length=config['uana']['inversion']['minimum_convergence_length'],
        convergence_threshold=config['uana']['inversion']['convergence_threshold']
    )
    training_time = time.time() - start_time

    data = {
        "inversion_time": training_time,
        "loss": loss_value
    }

    root_path = utils.get_uana_inversion_directory(
        results_base_path=config['results_base_path'],
        experiment_name=config['experiment_name'],
        config_name=config['config_name'],
        epistemic_weight=config['uana']['inversion']['epistemic_weight'],
        aleatory_weight=config['uana']['inversion']['aleatory_weight'],
        experiment_repetition_id=experiment_repetition_id,
        inversion_repetition_id=inversion_repetition_id
    )
    utils.create_directory(directory=root_path)

    utils.save_json_file(data=data, path=os.path.join(root_path, 'info.json'))
    utils.save_tensor_to_mat_file(tensor=design, path=os.path.join(root_path, 'design.mat'))
    utils.save_tensor_to_mat_file(tensor=mu_mean, path=os.path.join(root_path, 'performance.mat'))
    utils.save_tensor_to_mat_file(tensor=sigma_epistemic, path=os.path.join(root_path, 'sigma_epistemic.mat'))
    utils.save_tensor_to_mat_file(tensor=sigma_aleatory, path=os.path.join(root_path, 'sigma_aleatory.mat'))
    utils.save_tensor_to_mat_file(tensor=no_reduction_loss, path=os.path.join(root_path, 'no_reduction_loss.mat'))


def aggregate_results() -> None:
    pass


def validate() -> None:
    pass
