import os
import time

import numpy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .. import inversion, loss, model, oracle, settings, train, utils


def train_mu_model(
            config: dict,
            model_id: int
        ) -> None:

    mu_model = model.uana.create_mu_model(
        layers_size=[config['design_size']] + config['uana']['mu_model']['hidden_layers_size'] + [config['performance_size']],
        activation_function=getattr(nn, config['uana']['mu_model']['activation_functions'][model_id])()
    )

    train_tensor_dataset, test_tensor_dataset = utils.get_train_test_tensor_datasets(
        tensor_dataset=TensorDataset(
            utils.load_tensor_from_mat_file(path=config['design_path']),
            utils.load_tensor_from_mat_file(path=config['performance_path'])
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

    mu_model = model.uana.load_mu_model(config=config, model_id=model_id, fine_tuned=False)

    sigma_model = model.uana.create_sigma_model(
        layers_size=[config['design_size']] + config['uana']['sigma_model']['hidden_layers_size'] +
                    [config['performance_size']],
    )

    train_tensor_dataset, test_tensor_dataset = utils.get_train_test_tensor_datasets(
        tensor_dataset=TensorDataset(
            utils.load_tensor_from_mat_file(path=config['design_path']),
            utils.load_tensor_from_mat_file(path=config['performance_path'])
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
    mu_model_root_path = utils.get_uana_mu_model_directory(
        results_base_path=config['results_base_path'],
        experiment_name=config['experiment_name'],
        config_name=config['config_name'],
        model_id=model_id
    )

    utils.save_json_file(data=data, path=os.path.join(root_path, 'info.json'))
    utils.save_model(model=sigma_model, path=os.path.join(root_path, f'model_{model_id}.pt'))
    utils.save_model(model=mu_model, path=os.path.join(mu_model_root_path, f'model_{model_id}_fine_tuned.pt'))


def invert_target(
            config: dict,
            experiment_repetition_id: int,
            inversion_repetition_id: int,
            target_name: str
        ) -> None:

    mu_models = model.uana.load_mu_models(config=config, fine_tuned=True)
    sigma_models = model.uana.load_sigma_models(config=config)

    target = utils.load_tensor_from_mat_file(path=config[f'{target_name}_path']).to(settings.DEVICE)
    design = torch.rand(
        target.shape[0], config['design_size']
    ).type(torch.float).to(settings.DEVICE).clone().detach().requires_grad_(True)

    optimizer = optim.Adam([design], lr=config['uana']['inversion']['learning_rate'])

    start_time = time.time()
    loss_value, design, mu_mean, sigma_epistemic, sigma_aleatory, no_reduction_loss = inversion.uana.converge_design(
        mu_models=mu_models,
        sigma_models=sigma_models,
        epistemic_weight=config['uana']['inversion']['epistemic_weight'],
        aleatory_weight=config['uana']['inversion']['aleatory_weight'],
        criterion=getattr(loss.uana, config['uana']['inversion']['criterion']),
        no_reduction_criterion=getattr(loss.uana, config['uana']['inversion']['no_reduction_criterion']),
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


def aggregate_results(
            config: dict,
            experiment_repetition_id: int
        ) -> None:

    root_path = utils.get_uana_inversion_directory(
        results_base_path=config['results_base_path'],
        experiment_name=config['experiment_name'],
        config_name=config['config_name'],
        epistemic_weight=config['uana']['inversion']['epistemic_weight'],
        aleatory_weight=config['uana']['inversion']['aleatory_weight'],
        experiment_repetition_id=experiment_repetition_id,
        inversion_repetition_id=0
    )
    root_path = os.path.dirname(root_path)

    repetition_ids = sorted(
        [
            int(directory)
            for directory in os.listdir(root_path)
            if directory.isdigit() and os.path.isdir(os.path.join(root_path, directory))
        ]
    )

    total_no_reduction_losses = []
    for i in range(len(repetition_ids)):
        repetition_id = repetition_ids[i]
        total_no_reduction_losses.append(
            utils.load_mat_file(path=os.path.join(root_path, str(repetition_id), 'no_reduction_loss.mat'))['data']
        )
    total_no_reduction_losses = numpy.array(total_no_reduction_losses).T
    batch_size = total_no_reduction_losses.shape[0]

    best_repetition_id_per_element = []
    best_design = []
    best_no_reduction_loss = []
    best_performance = []
    best_sigma_aleatory = []
    best_sigma_epistemic = []
    for element_id in range(batch_size):
        best_repetition_id = numpy.argmin(total_no_reduction_losses[element_id, :])

        best_repetition_id_per_element.append(best_repetition_id)
        best_design.append(
            utils.load_mat_file(
                path=os.path.join(root_path, str(best_repetition_id), 'design.mat')
            )['data'][element_id, :]
        )
        best_no_reduction_loss.append(
            utils.load_mat_file(
                path=os.path.join(root_path, str(best_repetition_id), 'no_reduction_loss.mat')
            )['data'][0, element_id]
        )
        best_performance.append(
            utils.load_mat_file(
                path=os.path.join(root_path, str(best_repetition_id), 'performance.mat')
            )['data'][element_id, :]
        )
        best_sigma_aleatory.append(
            utils.load_mat_file(
                path=os.path.join(root_path, str(best_repetition_id), 'sigma_aleatory.mat')
            )['data'][element_id, :]
        )
        best_sigma_epistemic.append(
            utils.load_mat_file(
                path=os.path.join(root_path, str(best_repetition_id), 'sigma_epistemic.mat')
            )['data'][element_id, :]
        )

    aggregated_results_base_path = os.path.join(root_path, 'aggregated')
    if not os.path.exists(aggregated_results_base_path):
        os.makedirs(aggregated_results_base_path)

    utils.save_mat_file(
        data={'data': numpy.array(best_repetition_id_per_element)},
        path=os.path.join(aggregated_results_base_path, 'repetition_id_per_element.mat'),
    )
    utils.save_mat_file(
        data={'data': numpy.array(best_design)},
        path=os.path.join(aggregated_results_base_path, 'design.mat'),
    )
    utils.save_mat_file(
        data={'data': numpy.array(best_no_reduction_loss)},
        path=os.path.join(aggregated_results_base_path, 'no_reduction_loss.mat'),
    )
    utils.save_mat_file(
        data={'data': numpy.array(best_performance)},
        path=os.path.join(aggregated_results_base_path, 'performance.mat'),
    )
    utils.save_mat_file(
        data={'data': numpy.array(best_sigma_aleatory)},
        path=os.path.join(aggregated_results_base_path, 'sigma_aleatory.mat'),
    )
    utils.save_mat_file(
        data={'data': numpy.array(best_sigma_epistemic)},
        path=os.path.join(aggregated_results_base_path, 'sigma_epistemic.mat'),
    )


def evaluate_results(
            config: dict,
            experiment_repetition_id: int,
            target_name: str
        ) -> None:

    root_path = utils.get_uana_inversion_directory(
        results_base_path=config['results_base_path'],
        experiment_name=config['experiment_name'],
        config_name=config['config_name'],
        epistemic_weight=config['uana']['inversion']['epistemic_weight'],
        aleatory_weight=config['uana']['inversion']['aleatory_weight'],
        experiment_repetition_id=experiment_repetition_id,
        inversion_repetition_id=0
    )
    root_path = os.path.join(os.path.dirname(root_path), 'aggregated')

    getattr(getattr(oracle, config['oracle']), 'evaluate_results')(
        config=config,
        root_path=root_path,
        target_name=target_name
    )
