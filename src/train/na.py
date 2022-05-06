from typing import Callable

from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

from .. import settings, test


def _train_forward_model_one_epoch(
            forward_model: Module,
            criterion: Callable[[Tensor, Tensor], Tensor],
            optimizer: Optimizer,
            train_data_loader: DataLoader,
            epoch_number: int,
            number_of_epochs: int
        ) -> None:

    forward_model.train()
    for batch_number, (data, ground_truth) in enumerate(train_data_loader, 1):
        data = data.to(settings.DEVICE)
        ground_truth = ground_truth.to(settings.DEVICE)

        forward_model_output = forward_model(data)

        loss = criterion(forward_model_output, ground_truth)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_number % settings.NUMBER_OF_BATCHES_PER_LOG == 0:
            print(
                f'Epoch [{epoch_number}/{number_of_epochs}], Batch [{batch_number}/{len(train_data_loader)}]',
                f'Loss: {loss.item():.{settings.NUMBER_OF_DECIMAL_PLACES}f}'
            )


def train_forward_model(
            forward_model: Module,
            criterion: Callable[[Tensor, Tensor], Tensor],
            optimizer: Optimizer,
            learning_rate_scheduler: _LRScheduler,
            train_data_loader: DataLoader,
            test_data_loader: DataLoader,
            number_of_epochs: int
        ) -> float:

    print("Training Forward Model ...")
    forward_model.train()
    last_loss = 0.0
    for epoch_number in range(1, number_of_epochs + 1, 1):
        _train_forward_model_one_epoch(
            forward_model=forward_model,
            criterion=criterion,
            optimizer=optimizer,
            train_data_loader=train_data_loader,
            epoch_number=epoch_number,
            number_of_epochs=number_of_epochs
        )
        print()
        learning_rate_scheduler.step()
        last_loss = test.na.test_forward_model(
            forward_model=forward_model,
            criterion=criterion,
            test_data_loader=test_data_loader
        )
        print()
    print("Forward Model Trained.")
    return last_loss
