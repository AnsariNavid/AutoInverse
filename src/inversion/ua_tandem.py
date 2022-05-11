from typing import List, Tuple

import torch
from torch import Tensor
from torch.nn import Module

from .. import settings


def invert_performance(
            backward_model: Module,
            mu_models: List[Module],
            sigma_models: List[Module],
            target: Tensor
        ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:

    print("Inverting Performance ...")

    backward_model.eval()
    number_of_models = len(mu_models)
    for i in range(number_of_models):
        mu_models[i].eval()
        sigma_models[i].eval()

    design = backward_model(target)

    batch_size = target.shape[0]
    output_size = target.shape[1]

    ensembled_outputs = torch.empty(number_of_models, batch_size, output_size * 2).to(settings.DEVICE)

    for i in range(number_of_models):
        ensembled_outputs[i, :, :] = torch.cat((mu_models[i](design), sigma_models[i](design)), 1)

    mu_mean = (1 / number_of_models) * torch.sum(ensembled_outputs[:, :, :output_size], 0)
    sigma_aleatory = (1 / number_of_models) * torch.sum(ensembled_outputs[:, :, output_size:], 0)
    sigma_epistemic = (1 / number_of_models) * \
        torch.sum(ensembled_outputs[:, :, :output_size] ** 2 - mu_mean.repeat(number_of_models, 1, 1) ** 2, 0)

    print("Performance Inverted ...")

    design = design.detach().clone()
    mu_mean = mu_mean.detach().clone()
    sigma_aleatory = sigma_aleatory.detach().clone()
    sigma_epistemic = sigma_epistemic.detach().clone()

    return design, mu_mean, sigma_epistemic, sigma_aleatory
