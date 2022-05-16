from typing import Callable, List, Tuple

import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

from .. import settings, test


def _train_backward_model_one_epoch(
            backward_model: Module,
            mu_models: List[Module],
            sigma_models: List[Module],
            epistemic_weight: float,
            aleatory_weight: float,
            criterion: Callable[[Tensor, Tensor, float, Tensor, float, Tensor], Tuple[Tensor, Tensor]],
            optimizer: Optimizer,
            train_data_loader: DataLoader,
            epoch_number: int,
            number_of_epochs: int
        ) -> None:

    backward_model.train()
    number_of_models = len(mu_models)
    for i in range(number_of_models):
        mu_models[i].eval()
        sigma_models[i].eval()

    for batch_number, (data, ) in enumerate(train_data_loader, 1):
        data = data.to(settings.DEVICE)
        output_size = data.shape[1]
        batch_size = data.shape[0]

        backward_model_output = backward_model(data)
        ensembled_outputs = torch.empty(number_of_models, batch_size, output_size * 2).to(settings.DEVICE)

        for i in range(number_of_models):
            ensembled_outputs[i, :, :] = torch.cat(
                (mu_models[i](backward_model_output), sigma_models[i](backward_model_output)),
                1
            )

        mu_mean = (1 / number_of_models) * torch.sum(ensembled_outputs[:, :, :output_size], 0)
        sigma_aleatory = (1 / number_of_models) * torch.sum(ensembled_outputs[:, :, output_size:], 0)
        sigma_epistemic = (1 / number_of_models) * \
            torch.sum(ensembled_outputs[:, :, :output_size] ** 2 - mu_mean.repeat(number_of_models, 1, 1) ** 2, 0)

        mse_loss, uncertainty_loss = criterion(
            mu_mean,
            data,
            epistemic_weight,
            sigma_epistemic,
            aleatory_weight,
            sigma_aleatory
        )
        loss = mse_loss + uncertainty_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_number % settings.NUMBER_OF_BATCHES_PER_LOG == 0:
            print(
                f'Epoch [{epoch_number}/{number_of_epochs}], Batch [{batch_number}/{len(train_data_loader)}]',
                f'Loss: {loss.item():.{settings.NUMBER_OF_DECIMAL_PLACES}f}'
            )


def train_backward_model(
            backward_model: Module,
            mu_models: List[Module],
            sigma_models: List[Module],
            epistemic_weight: float,
            aleatory_weight: float,
            criterion: Callable[[Tensor, Tensor, float, Tensor, float, Tensor], Tuple[Tensor, Tensor]],
            optimizer: Optimizer,
            learning_rate_scheduler: _LRScheduler,
            train_data_loader: DataLoader,
            test_data_loader: DataLoader,
            number_of_epochs: int
        ) -> float:

    print("Training Backward Model ...")

    backward_model.train()
    number_of_models = len(mu_models)
    for i in range(number_of_models):
        mu_models[i].eval()
        sigma_models[i].eval()
    last_loss = 0.0

    for epoch_number in range(1, number_of_epochs + 1, 1):
        _train_backward_model_one_epoch(
            backward_model=backward_model,
            mu_models=mu_models,
            sigma_models=sigma_models,
            epistemic_weight=epistemic_weight,
            aleatory_weight=aleatory_weight,
            criterion=criterion,
            optimizer=optimizer,
            train_data_loader=train_data_loader,
            epoch_number=epoch_number,
            number_of_epochs=number_of_epochs
        )
        print()

        learning_rate_scheduler.step()

        last_loss = test.ua_tandem.test_backward_model(
            backward_model=backward_model,
            mu_models=mu_models,
            sigma_models=sigma_models,
            epistemic_weight=epistemic_weight,
            aleatory_weight=aleatory_weight,
            criterion=criterion,
            test_data_loader=test_data_loader
        )
        print()

    print("Backward Model Trained.")

    return last_loss
