import os
import time

import torch.nn as nn
import torch.optim as optim

import src


def train_mu_model(
            config: dict,
            model_id: int
        ) -> None:

    mu_model = src.model.create_mu_model(
        layers_size=config['mu_model']['layers_size'],
        activation_function=getattr(nn, config['mu_model']['activation_functions'][model_id])()
    )

    train_tensor_dataset, test_tensor_dataset = src.dataset.get_train_test_tensor_datasets(
        tensor_dataset=src.dataset.create_tensor_dataset(
            src.dataset.load_tensor_from_mat_file(path=config['mu_model']['input_data_path']),
            src.dataset.load_tensor_from_mat_file(path=config['mu_model']['output_data_path'])
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


if __name__ == '__main__':
    config = {
        "results_base_path": "",
        "experiment_name": "",
        "config_name": "",

        "mu_model": {
            "layers_size": [40, 300, 1500, 1500, 300, 206],
            "activation_functions": ["ReLU"],
            "input_data_path": "multi_joint_input.mat",
            "output_data_path": "multi_joint_output.mat",
            "train_split_ratio": 0.8,
            "batch_size": 16,
            "learning_rate": 0.0005,
            "learning_rate_decay_rate": 0.95,
            "number_of_epochs": 2
        },
    }
    train_mu_model(config=config, model_id=0)
