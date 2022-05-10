from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor


def softrobot_loss(
            mu_mean: Tensor,
            target: Tensor,
            epistemic_weight: float,
            sigma_epistemic: Tensor,
            aleatory_weight: float,
            sigma_aleatory: Tensor
        ) -> Tuple[Tensor, Tensor]:

    def tip_position(softrobot_vertices: Tensor):
        assert softrobot_vertices.shape[-1] == 206
        if len(softrobot_vertices.shape) == 2:
            return softrobot_vertices[:, 122:124]
        elif len(softrobot_vertices.shape) == 1:
            return softrobot_vertices[122:124]
        else:
            raise Exception()

    mse_loss = (nn.MSELoss())(tip_position(mu_mean), tip_position(target))
    uncertainty_loss = (epistemic_weight * sigma_epistemic + aleatory_weight * sigma_aleatory).mean()
    return mse_loss, uncertainty_loss


def inversion_loss(
            mu_mean: Tensor,
            target: Tensor,
            epistemic_weight: float,
            sigma_epistemic: Tensor,
            aleatory_weight: float,
            sigma_aleatory: Tensor
        ) -> Tuple[Tensor, Tensor]:

    mse_loss = (nn.MSELoss())(mu_mean, target)
    uncertainty_loss = (epistemic_weight * sigma_epistemic + aleatory_weight * sigma_aleatory).mean()
    return mse_loss, uncertainty_loss
