from typing import Callable

import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

from . import settings, test


def _train_mu_model_one_epoch(
            mu_model: Module,
            criterion: Callable[[Tensor, Tensor], Tensor],
            optimizer: Optimizer,
            train_dataset_loader: DataLoader,
            epoch_number: int,
            number_of_epochs: int
        ) -> None:

    mu_model.train()
    for batch_number, (data, ground_truth) in enumerate(train_dataset_loader, 1):
        data = data.to(settings.DEVICE)
        ground_truth = ground_truth.to(settings.DEVICE)

        mu_model_output = mu_model(data)

        loss = criterion(mu_model_output, ground_truth)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_number % settings.NUMBER_OF_BATCHES_PER_LOG == 0:
            print(
                f'Epoch [{epoch_number}/{number_of_epochs}], Batch [{batch_number}/{len(train_dataset_loader)}]',
                f'Loss: {loss.item():.{settings.NUMBER_OF_DECIMAL_PLACES}f}'
            )
    print()


def train_mu_model(
            mu_model: Module,
            criterion: Callable[[Tensor, Tensor], Tensor],
            optimizer: Optimizer,
            learning_rate_scheduler: _LRScheduler,
            train_dataset_loader: DataLoader,
            test_dataset_loader: DataLoader,
            number_of_epochs: int
        ) -> None:

    print("Training Mu Model ...")
    mu_model.train()
    for epoch_number in range(1, number_of_epochs + 1, 1):
        _train_mu_model_one_epoch(
            mu_model=mu_model,
            criterion=criterion,
            optimizer=optimizer,
            train_dataset_loader=train_dataset_loader,
            epoch_number=epoch_number,
            number_of_epochs=number_of_epochs
        )
        learning_rate_scheduler.step()
        test.test_mu_model(
            mu_model=mu_model,
            criterion=criterion,
            test_dataset_loader=test_dataset_loader
        )
    print("Mu Model Trained.")
    print()
    print()


def _train_sigma_model_one_epoch(
            sigma_model: Module,
            mu_model: Module,
            criterion: Callable[[Tensor, Tensor], Tensor],
            sigma_optimizer: Optimizer,
            mu_optimizer: Optimizer,
            train_dataset_loader: DataLoader,
            epoch_number: int,
            number_of_epochs: int
        ) -> None:

    sigma_model.train()
    mu_model.train()
    for batch_number, (data, ground_truth) in enumerate(train_dataset_loader, 1):
        data = data.to(settings.DEVICE)
        ground_truth = ground_truth.to(settings.DEVICE)

        sigma_model_output = sigma_model(data)
        sigma_model_output = torch.add(sigma_model_output, settings.EPSILON)
        mu_model_output = mu_model(data)
        model_output = torch.cat((mu_model_output, sigma_model_output), 1)

        loss = criterion(model_output, ground_truth)

        sigma_optimizer.zero_grad()
        mu_optimizer.zero_grad()
        loss.backward()
        sigma_optimizer.step()
        mu_optimizer.step()

        if batch_number % settings.NUMBER_OF_BATCHES_PER_LOG == 0:
            print(
                f'Epoch [{epoch_number}/{number_of_epochs}], Batch [{batch_number}/{len(train_dataset_loader)}]',
                f'Loss: {loss.item():.{settings.NUMBER_OF_DECIMAL_PLACES}f}'
            )
    print()


def train_sigma_model(
            sigma_model: Module,
            mu_model: Module,
            sigma_criterion: Callable[[Tensor, Tensor], Tensor],
            mu_criterion: Callable[[Tensor, Tensor], Tensor],
            sigma_optimizer: Optimizer,
            mu_optimizer: Optimizer,
            sigma_learning_rate_scheduler: _LRScheduler,
            mu_learning_rate_scheduler: _LRScheduler,
            train_dataset_loader: DataLoader,
            test_dataset_loader: DataLoader,
            number_of_epochs: int
        ) -> None:

    print("Training Sigma Model ...")
    sigma_model.train()
    mu_model.train()
    for epoch_number in range(1, number_of_epochs + 1, 1):
        _train_sigma_model_one_epoch(
            sigma_model=sigma_model,
            mu_model=mu_model,
            criterion=sigma_criterion,
            sigma_optimizer=sigma_optimizer,
            mu_optimizer=mu_optimizer,
            train_dataset_loader=train_dataset_loader,
            epoch_number=epoch_number,
            number_of_epochs=number_of_epochs
        )
        sigma_learning_rate_scheduler.step()
        mu_learning_rate_scheduler.step()
        test.test_sigma_model(
            sigma_model=sigma_model,
            mu_model=mu_model,
            criterion=sigma_criterion,
            test_dataset_loader=test_dataset_loader
        )
        test.test_mu_model(
            mu_model=mu_model,
            criterion=mu_criterion,
            test_dataset_loader=test_dataset_loader
        )
    print("Sigma Model Trained.")
    print()
    print()
