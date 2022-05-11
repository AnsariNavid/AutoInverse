import os
import time

import numpy
import torch
import torch.nn as nn
import torch.optim as optim

from .. import inversion, loss, model, oracle, settings, utils


def inverse_design(
            config: dict,
            experiment_repetition_id: int,
            inversion_repetition_id: int
        ) -> None:

    forward_models = []
    for model_id in range(len(config['uana']['mu_model']['activation_functions'])):
        forward_model = model.uana.create_mu_model(
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
        forward_model = utils.load_model(
            model=forward_model,
            path=os.path.join(mu_model_root_path, f'model_{model_id}.pt')
        )
        forward_model.eval()
        forward_models.append(forward_model)

    target = utils.load_tensor_from_mat_file(path=config['target_data_path']).to(settings.DEVICE)
    design = torch.rand(
        target.shape[0], config['input_size']
    ).type(torch.float).to(settings.DEVICE).clone().detach().requires_grad_(True)

    optimizer = optim.Adam([design], lr=config['nae']['inversion']['learning_rate'])

    start_time = time.time()
    loss_value, design, forward_mean, no_reduction_loss = inversion.na.converge_design(
        forward_models=forward_models,
        criterion=getattr(loss.na, config['nae']['inversion']['criterion']),
        no_reduction_criterion=getattr(loss.na, config['nae']['inversion']['no_reduction_criterion']),
        optimizer=optimizer,
        design=design,
        target=target,
        number_of_iterations=config['nae']['inversion']['number_of_iterations'],
        minimum_convergence_length=config['nae']['inversion']['minimum_convergence_length'],
        convergence_threshold=config['nae']['inversion']['convergence_threshold']
    )
    training_time = time.time() - start_time

    data = {
        "inversion_time": training_time,
        "loss": loss_value
    }

    root_path = utils.get_nae_inversion_directory(
        results_base_path=config['results_base_path'],
        experiment_name=config['experiment_name'],
        config_name=config['config_name'],
        experiment_repetition_id=experiment_repetition_id,
        inversion_repetition_id=inversion_repetition_id
    )
    utils.create_directory(directory=root_path)

    utils.save_json_file(data=data, path=os.path.join(root_path, 'info.json'))
    utils.save_tensor_to_mat_file(tensor=design, path=os.path.join(root_path, 'design.mat'))
    utils.save_tensor_to_mat_file(tensor=forward_mean, path=os.path.join(root_path, 'performance.mat'))
    utils.save_tensor_to_mat_file(tensor=no_reduction_loss, path=os.path.join(root_path, 'no_reduction_loss.mat'))


def aggregate_results(
            config: dict,
            experiment_repetition_id: int
        ) -> None:

    root_path = utils.get_nae_inversion_directory(
        results_base_path=config['results_base_path'],
        experiment_name=config['experiment_name'],
        config_name=config['config_name'],
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


def validate(
            config: dict,
            experiment_repetition_id: int
        ) -> None:

    root_path = utils.get_nae_inversion_directory(
        results_base_path=config['results_base_path'],
        experiment_name=config['experiment_name'],
        config_name=config['config_name'],
        experiment_repetition_id=experiment_repetition_id,
        inversion_repetition_id=0
    )
    root_path = os.path.join(os.path.dirname(root_path), 'aggregated')

    designs = utils.load_mat_file(path=os.path.join(root_path, 'design.mat'))['data']
    performances = utils.load_mat_file(path=os.path.join(root_path, 'performance.mat'))['data']
    targets = utils.load_mat_file(path=config['target_data_path'])['data']

    re_simulation_error = getattr(
        getattr(oracle, config['oracle']),
        'get_re_simulation_error'
    )(designs=designs, targets=targets)
    surrogate_error = getattr(
        getattr(oracle, config['oracle']),
        'get_surrogate_error'
    )(performances=performances, targets=targets)

    data = {
        "re_simulation_error": re_simulation_error,
        "surrogate_error": surrogate_error
    }
    utils.save_json_file(data=data, path=os.path.join(root_path, 'validation_results.json'))
