import os
import time

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .. import inversion, model, oracle, settings, train, utils, loss


def train_backward_model(
            config: dict,
            experiment_repetition_id: int
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

    backward_model = model.tandem.create_backward_model(
        layers_size=[config['output_size']] + config['ua_tandem']['backward_model']['hidden_layers_size'] +
                    [config['input_size']]
    )

    train_tensor_dataset, test_tensor_dataset = utils.get_train_test_tensor_datasets(
        tensor_dataset=TensorDataset(
            utils.load_tensor_from_mat_file(path=config['output_data_path'])
        ),
        train_split_ratio=config['ua_tandem']['backward_model']['train_split_ratio']
    )
    train_data_loader = DataLoader(
        dataset=train_tensor_dataset,
        batch_size=config['ua_tandem']['backward_model']['batch_size'],
        shuffle=True,
        pin_memory=True
    )
    test_data_loader = DataLoader(
        dataset=test_tensor_dataset,
        batch_size=config['ua_tandem']['backward_model']['batch_size'],
        shuffle=True,
        pin_memory=True
    )

    optimizer = optim.Adam(backward_model.parameters(), lr=config['ua_tandem']['backward_model']['learning_rate'])
    learning_rate_scheduler = optim.lr_scheduler.ExponentialLR(
        optimizer=optimizer,
        gamma=config['ua_tandem']['backward_model']['learning_rate_decay_rate']
    )

    start_time = time.time()
    loss_value = train.ua_tandem.train_backward_model(
        backward_model=backward_model,
        mu_models=mu_models,
        sigma_models=sigma_models,
        epistemic_weight=config['ua_tandem']['backward_model']['epistemic_weight'],
        aleatory_weight=config['ua_tandem']['backward_model']['aleatory_weight'],
        criterion=getattr(loss.uana, config['ua_tandem']['backward_model']['criterion']),
        optimizer=optimizer,
        learning_rate_scheduler=learning_rate_scheduler,
        train_data_loader=train_data_loader,
        test_data_loader=test_data_loader,
        number_of_epochs=config['ua_tandem']['backward_model']['number_of_epochs']
    )
    training_time = time.time() - start_time
    number_of_trainable_parameters = int(sum(p.numel() for p in backward_model.parameters() if p.requires_grad))

    data = {
        "training_time": training_time,
        "number_of_trainable_parameters": number_of_trainable_parameters,
        "loss": loss_value
    }

    root_path = utils.get_ua_tandem_backward_model_directory(
        results_base_path=config['results_base_path'],
        experiment_name=config['experiment_name'],
        config_name=config['config_name'],
        epistemic_weight=config['ua_tandem']['backward_model']['epistemic_weight'],
        aleatory_weight=config['ua_tandem']['backward_model']['aleatory_weight'],
        experiment_repetition_id=experiment_repetition_id
    )
    utils.create_directory(directory=root_path)

    utils.save_json_file(data=data, path=os.path.join(root_path, 'info.json'))
    utils.save_model(model=backward_model, path=os.path.join(root_path, 'model.pt'))


def inverse_design(
            config: dict,
            experiment_repetition_id: int
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

    backward_model = model.tandem.create_backward_model(
        layers_size=[config['output_size']] + config['ua_tandem']['backward_model']['hidden_layers_size'] +
                    [config['input_size']]
    )
    backward_model_root_path = utils.get_ua_tandem_backward_model_directory(
        results_base_path=config['results_base_path'],
        experiment_name=config['experiment_name'],
        config_name=config['config_name'],
        epistemic_weight=config['ua_tandem']['backward_model']['epistemic_weight'],
        aleatory_weight=config['ua_tandem']['backward_model']['aleatory_weight'],
        experiment_repetition_id=experiment_repetition_id
    )
    backward_model = utils.load_model(
        model=backward_model,
        path=os.path.join(backward_model_root_path, 'model.pt')
    )
    backward_model.eval()

    target = utils.load_tensor_from_mat_file(path=config['target_data_path']).to(settings.DEVICE)

    design, performance, sigma_epistemic, sigma_aleatory = inversion.ua_tandem.invert_performance(
        backward_model=backward_model,
        mu_models=mu_models,
        sigma_models=sigma_models,
        target=target
    )

    root_path = utils.get_ua_tandem_inversion_directory(
        results_base_path=config['results_base_path'],
        experiment_name=config['experiment_name'],
        config_name=config['config_name'],
        epistemic_weight=config['ua_tandem']['backward_model']['epistemic_weight'],
        aleatory_weight=config['ua_tandem']['backward_model']['aleatory_weight'],
        experiment_repetition_id=experiment_repetition_id
    )
    utils.create_directory(directory=root_path)

    utils.save_tensor_to_mat_file(tensor=design, path=os.path.join(root_path, 'design.mat'))
    utils.save_tensor_to_mat_file(tensor=performance, path=os.path.join(root_path, 'performance.mat'))
    utils.save_tensor_to_mat_file(tensor=sigma_epistemic, path=os.path.join(root_path, 'sigma_epistemic.mat'))
    utils.save_tensor_to_mat_file(tensor=sigma_aleatory, path=os.path.join(root_path, 'sigma_aleatory.mat'))


def validate(
            config: dict,
            experiment_repetition_id: int
        ) -> None:

    root_path = utils.get_ua_tandem_inversion_directory(
        results_base_path=config['results_base_path'],
        experiment_name=config['experiment_name'],
        config_name=config['config_name'],
        epistemic_weight=config['ua_tandem']['backward_model']['epistemic_weight'],
        aleatory_weight=config['ua_tandem']['backward_model']['aleatory_weight'],
        experiment_repetition_id=experiment_repetition_id
    )

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
