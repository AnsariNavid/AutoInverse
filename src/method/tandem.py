import os
import time

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .. import inversion, model, oracle, settings, train, utils


def train_backward_model(
            config: dict,
            experiment_repetition_id: int
        ) -> None:

    forward_model = model.na.load_forward_model(config=config)

    backward_model = model.tandem.create_backward_model(
        layers_size=[config['performance_size']] + config['tandem']['backward_model']['hidden_layers_size'] +
                    [config['design_size']]
    )

    train_tensor_dataset, test_tensor_dataset = utils.get_train_test_tensor_datasets(
        tensor_dataset=TensorDataset(
            utils.load_tensor_from_mat_file(path=config['performance_path'])
        ),
        train_split_ratio=config['tandem']['backward_model']['train_split_ratio']
    )
    train_data_loader = DataLoader(
        dataset=train_tensor_dataset,
        batch_size=config['tandem']['backward_model']['batch_size'],
        shuffle=True,
        pin_memory=True
    )
    test_data_loader = DataLoader(
        dataset=test_tensor_dataset,
        batch_size=config['tandem']['backward_model']['batch_size'],
        shuffle=True,
        pin_memory=True
    )

    optimizer = optim.Adam(backward_model.parameters(), lr=config['tandem']['backward_model']['learning_rate'])
    criterion = nn.MSELoss()
    learning_rate_scheduler = optim.lr_scheduler.ExponentialLR(
        optimizer=optimizer,
        gamma=config['tandem']['backward_model']['learning_rate_decay_rate']
    )

    start_time = time.time()
    loss_value = train.tandem.train_backward_model(
        backward_model=backward_model,
        forward_model=forward_model,
        criterion=criterion,
        optimizer=optimizer,
        learning_rate_scheduler=learning_rate_scheduler,
        train_data_loader=train_data_loader,
        test_data_loader=test_data_loader,
        number_of_epochs=config['tandem']['backward_model']['number_of_epochs']
    )
    training_time = time.time() - start_time
    number_of_trainable_parameters = int(sum(p.numel() for p in backward_model.parameters() if p.requires_grad))

    data = {
        "training_time": training_time,
        "number_of_trainable_parameters": number_of_trainable_parameters,
        "loss": loss_value
    }

    root_path = utils.get_tandem_backward_model_directory(
        results_base_path=config['results_base_path'],
        experiment_name=config['experiment_name'],
        config_name=config['config_name'],
        experiment_repetition_id=experiment_repetition_id
    )
    utils.create_directory(directory=root_path)

    utils.save_json_file(data=data, path=os.path.join(root_path, 'info.json'))
    utils.save_model(model=backward_model, path=os.path.join(root_path, 'model.pt'))


def invert_target(
            config: dict,
            experiment_repetition_id: int,
            target_name: str
        ) -> None:

    forward_model = model.na.load_forward_model(config=config)

    backward_model = model.tandem.load_backward_model(
        config=config,
        experiment_repetition_id=experiment_repetition_id
    )

    target = utils.load_tensor_from_mat_file(path=config[f'{target_name}_path']).to(settings.DEVICE)

    design, performance = inversion.tandem.invert_target(
        backward_model=backward_model,
        forward_model=forward_model,
        target=target
    )

    root_path = utils.get_tandem_inversion_directory(
        results_base_path=config['results_base_path'],
        experiment_name=config['experiment_name'],
        config_name=config['config_name'],
        experiment_repetition_id=experiment_repetition_id,
        target_name=target_name
    )
    utils.create_directory(directory=root_path)

    utils.save_tensor_to_mat_file(tensor=design, path=os.path.join(root_path, 'design.mat'))
    utils.save_tensor_to_mat_file(tensor=performance, path=os.path.join(root_path, 'performance.mat'))


def evaluate_results(
            config: dict,
            experiment_repetition_id: int,
            target_name: str
        ) -> None:

    root_path = utils.get_tandem_inversion_directory(
        results_base_path=config['results_base_path'],
        experiment_name=config['experiment_name'],
        config_name=config['config_name'],
        experiment_repetition_id=experiment_repetition_id,
        target_name=target_name
    )

    getattr(getattr(oracle, config['oracle']), 'evaluate_results')(
        config=config,
        root_path=root_path,
        target_name=target_name
    )
