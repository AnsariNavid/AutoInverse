from typing import Callable

import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader

from . import settings


def test_mu_model(
            mu_model: Module,
            criterion: Callable[[Tensor, Tensor], Tensor],
            test_dataset_loader: DataLoader
        ) -> None:

    print("Testing Mu Model ...")
    mu_model.eval()
    with torch.no_grad():
        total_loss = 0.0
        for data, ground_truth in test_dataset_loader:
            data = data.to(settings.DEVICE)
            ground_truth = ground_truth.to(settings.DEVICE)

            mu_model_output = mu_model(data)

            loss = criterion(mu_model_output, ground_truth)

            total_loss += loss.item()

        print(f'Loss: {total_loss / len(test_dataset_loader):.{settings.NUMBER_OF_DECIMAL_PLACES}f}')
    mu_model.train()
    print("Mu Model Tested.")
    print()
    print()


def test_sigma_model(
            sigma_model: Module,
            mu_model: Module,
            criterion: Callable[[Tensor, Tensor], Tensor],
            test_dataset_loader: DataLoader
        ) -> None:

    print("Testing Sigma Model ...")
    sigma_model.eval()
    mu_model.eval()
    with torch.no_grad():
        total_loss = 0.0
        for data, ground_truth in test_dataset_loader:
            data = data.to(settings.DEVICE)
            ground_truth = ground_truth.to(settings.DEVICE)

            sigma_model_output = sigma_model(data)
            sigma_model_output = torch.add(sigma_model_output, settings.EPSILON)
            mu_model_output = mu_model(data)
            model_output = torch.cat((mu_model_output, sigma_model_output), 1)

            loss = criterion(model_output, ground_truth)

            total_loss += loss.item()

        print(f'Loss: {total_loss / len(test_dataset_loader):.{settings.NUMBER_OF_DECIMAL_PLACES}f}')
    sigma_model.train()
    mu_model.train()
    print("Sigma Model Tested.")
    print()
    print()
