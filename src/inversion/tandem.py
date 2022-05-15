from typing import Tuple

from torch import Tensor
from torch.nn import Module


def invert_target(
            backward_model: Module,
            forward_model: Module,
            target: Tensor
        ) -> Tuple[Tensor, Tensor]:

    print("Inverting Performance ...")

    backward_model.eval()
    forward_model.eval()

    design = backward_model(target)
    performance = forward_model(design)

    print("Performance Inverted ...")

    design = design.detach().clone()
    performance = performance.detach().clone()

    return design, performance
