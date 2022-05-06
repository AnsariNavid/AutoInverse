from typing import Callable, List, Tuple

import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer

from .. import settings


def _converge_design_one_iteration(
            forward_models: List[Module],
            criterion: Callable[[Tensor, Tensor], Tensor],
            optimizer: Optimizer,
            design: Tensor,
            target: Tensor,
            iteration_number: int,
            number_of_iterations: int
        ) -> Tuple[Tensor, Tensor]:

    number_of_models = len(forward_models)
    output_size = target.shape[1]
    batch_size = design.shape[0]
    for i in range(number_of_models):
        forward_models[i].eval()

    ensembled_outputs = torch.empty(number_of_models, batch_size, output_size).to(settings.DEVICE)

    for i in range(number_of_models):
        ensembled_outputs[i, :, :] = forward_models[i](design)

    forward_mean = (1 / number_of_models) * torch.sum(ensembled_outputs, 0)

    loss = criterion(forward_mean, target)

    print(
        f'Iteration [{iteration_number}/{number_of_iterations}]',
        f'Loss: {loss.item():.{settings.NUMBER_OF_DECIMAL_PLACES}f}',
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss, forward_mean


def converge_design(
            forward_models: List[Module],
            criterion: Callable[[Tensor, Tensor], Tensor],
            no_reduction_criterion: Callable[[Tensor, Tensor], Tensor],
            optimizer: Optimizer,
            design: Tensor,
            target: Tensor,
            number_of_iterations: int,
            minimum_convergence_length: int,
            convergence_threshold: float,
        ) -> Tuple[float, Tensor, Tensor, Tensor]:

    print("Converging Design ...")
    losses = []
    best_loss = float('inf')
    best_design = None
    best_forward_mean = None
    best_no_reduction_loss = None
    for iteration_number in range(1, number_of_iterations + 1, 1):
        loss, forward_mean = _converge_design_one_iteration(
            forward_models=forward_models,
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
            best_forward_mean = forward_mean.detach().clone()
            best_no_reduction_loss = no_reduction_criterion(forward_mean, target).detach().clone()

        if len(losses) > minimum_convergence_length:
            maximum_difference = max(
                [abs(loss.item() - previous_loss) for previous_loss in losses[-minimum_convergence_length:]]
            )
            if maximum_difference <= convergence_threshold:
                break
        losses.append(loss.item())

    print("Design Converged.")
    return best_loss, best_design, best_forward_mean, best_no_reduction_loss
