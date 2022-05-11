from typing import Callable, List, Tuple

import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader

from .. import settings


def test_backward_model(
            backward_model: Module,
            mu_models: List[Module],
            sigma_models: List[Module],
            epistemic_weight: float,
            aleatory_weight: float,
            criterion: Callable[[Tensor, Tensor, float, Tensor, float, Tensor], Tuple[Tensor, Tensor]],
            test_data_loader: DataLoader
        ) -> float:

    print("Testing Backward Model ...")

    backward_model.eval()
    number_of_models = len(mu_models)
    for i in range(number_of_models):
        mu_models[i].eval()
        sigma_models[i].eval()
    total_loss = 0.0

    with torch.no_grad():
        for data in test_data_loader:
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

            total_loss += loss.item()

        print(f'Loss: {total_loss / len(test_data_loader):.{settings.NUMBER_OF_DECIMAL_PLACES}f}')

    print("Backward Model Tested.")

    return total_loss / len(test_data_loader)
