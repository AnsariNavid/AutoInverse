from typing import Callable

import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader

from .. import settings


def test_mu_model(
            mu_model: Module,
            criterion: Callable[[Tensor, Tensor], Tensor],
            test_data_loader: DataLoader
        ) -> float:

    print("Testing Mu Model ...")

    mu_model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for data, ground_truth in test_data_loader:
            data = data.to(settings.DEVICE)
            ground_truth = ground_truth.to(settings.DEVICE)

            mu_model_output = mu_model(data)

            loss = criterion(mu_model_output, ground_truth)

            total_loss += loss.item()

        print(f'Loss: {total_loss / len(test_data_loader):.{settings.NUMBER_OF_DECIMAL_PLACES}f}')

    print("Mu Model Tested.")

    return total_loss / len(test_data_loader)


def test_sigma_model(
            sigma_model: Module,
            mu_model: Module,
            criterion: Callable[[Tensor, Tensor], Tensor],
            test_data_loader: DataLoader
        ) -> float:

    print("Testing Sigma Model ...")

    sigma_model.eval()
    mu_model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for data, ground_truth in test_data_loader:
            data = data.to(settings.DEVICE)
            ground_truth = ground_truth.to(settings.DEVICE)

            sigma_model_output = sigma_model(data)
            sigma_model_output = torch.add(sigma_model_output, settings.EPSILON)
            mu_model_output = mu_model(data)
            model_output = torch.cat((mu_model_output, sigma_model_output), 1)

            loss = criterion(model_output, ground_truth)

            total_loss += loss.item()

        print(f'Loss: {total_loss / len(test_data_loader):.{settings.NUMBER_OF_DECIMAL_PLACES}f}')

    print("Sigma Model Tested.")

    return total_loss / len(test_data_loader)
