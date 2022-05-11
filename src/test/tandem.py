from typing import Callable

import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader

from .. import settings


def test_backward_model(
            backward_model: Module,
            forward_model: Module,
            criterion: Callable[[Tensor, Tensor], Tensor],
            test_data_loader: DataLoader
        ) -> float:

    print("Testing Backward Model ...")

    backward_model.eval()
    forward_model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for (data, ) in test_data_loader:
            data = data.to(settings.DEVICE)

            backward_model_output = backward_model(data)
            forward_model_output = forward_model(backward_model_output)

            loss = criterion(forward_model_output, data)

            total_loss += loss.item()

        print(f'Loss: {total_loss / len(test_data_loader):.{settings.NUMBER_OF_DECIMAL_PLACES}f}')

    print("Backward Model Tested.")

    return total_loss / len(test_data_loader)
