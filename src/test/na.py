from typing import Callable

import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader

from .. import settings


def test_forward_model(
            forward_model: Module,
            criterion: Callable[[Tensor, Tensor], Tensor],
            test_data_loader: DataLoader
        ) -> float:

    print("Testing Forward Model ...")

    forward_model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for data, ground_truth in test_data_loader:
            data = data.to(settings.DEVICE)
            ground_truth = ground_truth.to(settings.DEVICE)

            forward_model_output = forward_model(data)

            loss = criterion(forward_model_output, ground_truth)

            total_loss += loss.item()

        print(f'Loss: {total_loss / len(test_data_loader):.{settings.NUMBER_OF_DECIMAL_PLACES}f}')

    print("Forward Model Tested.")

    return total_loss / len(test_data_loader)
