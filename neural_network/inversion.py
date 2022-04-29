from typing import Callable, List, Tuple

import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer

from . import settings


def _converge_design_one_iteration(
            sigma_models: List[Module],
            mu_models: List[Module],
            epistemic_weight: float,
            aleatory_weight: float,
            criterion: Callable[[Tensor, Tensor, float, Tensor, float, Tensor], Tuple[Tensor, Tensor]],
            optimizer: Optimizer,
            design: Tensor,
            target: Tensor,
            iteration_number: int,
            number_of_iterations: int
        ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:

    for sigma_model in sigma_models:
        sigma_model.eval()
    for mu_model in mu_models:
        mu_model.eval()

    ensembled_outputs = torch.empty(len(sigma_models), design.shape[0], target.shape[1] * 2).to(settings.DEVICE)

    for i in range(len(sigma_models)):
        ensembled_outputs[i, :, :] = torch.cat((mu_models[i](design), sigma_models[i](design)), 1)

    mu_mean = (1 / len(sigma_models)) * torch.sum(ensembled_outputs[:, :, :target.shape[1]], 0)
    sigma_aleatory = (1 / len(sigma_models)) * torch.sum(ensembled_outputs[:, :, target.shape[1]:], 0)
    sigma_epistemic = (1 / len(sigma_models)) * \
        torch.sum(ensembled_outputs[:, :, :target.shape[1]] ** 2 - mu_mean.repeat(10, 1, 1) ** 2, 0)

    mse_loss, uncertainty_loss = criterion(
        mu_mean,
        target,
        epistemic_weight,
        sigma_epistemic,
        aleatory_weight,
        sigma_aleatory
    )
    loss = mse_loss + uncertainty_loss

    print(
        f'Iteration [{iteration_number}/{number_of_iterations}]',
        f'Loss: {loss.item():.{settings.NUMBER_OF_DECIMAL_PLACES}f}',
        f'MSE Loss: {mse_loss.item():.{settings.NUMBER_OF_DECIMAL_PLACES}f}',
        f'Uncertainty Loss: {(sigma_epistemic + sigma_aleatory).mean().item():.{settings.NUMBER_OF_DECIMAL_PLACES}f}',
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss, mu_mean, sigma_epistemic, sigma_aleatory


def converge_design(
            sigma_models: List[Module],
            mu_models: List[Module],
            epistemic_weight: float,
            aleatory_weight: float,
            criterion: Callable[[Tensor, Tensor, float, Tensor, float, Tensor], Tuple[Tensor, Tensor]],
            no_reduction_criterion: Callable[[Tensor, Tensor, float, Tensor, float, Tensor], Tensor],
            optimizer: Optimizer,
            design: Tensor,
            target: Tensor,
            number_of_iterations: int,
            minimum_conversion_length: int,
            conversion_threshold: float,
        ) -> Tuple[float, Tensor, Tensor, Tensor, Tensor, Tensor]:

    print("Converging Design ...")
    losses = []
    best_loss = float('inf')
    best_design = None
    best_mu_mean = None
    best_sigma_epistemic = None
    best_sigma_aleatory = None
    best_no_reduction_loss = None
    for iteration_number in range(1, number_of_iterations + 1, 1):
        loss, mu_mean, sigma_epistemic, sigma_aleatory = _converge_design_one_iteration(
            sigma_models=sigma_models,
            mu_models=mu_models,
            epistemic_weight=epistemic_weight,
            aleatory_weight=aleatory_weight,
            criterion=criterion,
            optimizer=optimizer,
            design=design,
            target=target,
            iteration_number=iteration_number,
            number_of_iterations=number_of_iterations
        )
        if loss.item() <= best_loss:
            best_loss = loss.item()
            best_design = design.detach().clone()
            best_mu_mean = mu_mean.detach().clone()
            best_sigma_epistemic = sigma_epistemic.detach().clone()
            best_sigma_aleatory = sigma_aleatory.detach().clone()
            best_no_reduction_loss = no_reduction_criterion(
                mu_mean,
                target,
                epistemic_weight,
                sigma_epistemic,
                aleatory_weight,
                sigma_aleatory
            ).detach().clone()

        if len(losses) > minimum_conversion_length:
            maximum_difference = max(
                [abs(loss.item() - previous_loss) for previous_loss in losses[-minimum_conversion_length:]]
            )
            if maximum_difference <= conversion_threshold:
                break
        losses.append(loss.item())

    print("Design Converged.")
    print()
    print()
    return best_loss, best_design, best_mu_mean, best_sigma_epistemic, best_sigma_aleatory, best_no_reduction_loss
