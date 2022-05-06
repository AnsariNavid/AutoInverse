import torch
import torch.nn as nn
from torch import Tensor


def softrobot_loss(
            forward_mean: Tensor,
            target: Tensor
        ) -> Tensor:

    def tip_position(softrobot_vertices: Tensor):
        assert softrobot_vertices.shape[-1] == 206
        if len(softrobot_vertices.shape) == 2:
            return softrobot_vertices[:, 122:124]
        elif len(softrobot_vertices.shape) == 1:
            return softrobot_vertices[122:124]
        else:
            raise Exception()

    return (nn.MSELoss())(tip_position(forward_mean), tip_position(target))


def no_reduction_softrobot_loss(
            forward_mean: Tensor,
            target: Tensor
        ) -> Tensor:

    def tip_position(softrobot_vertices: Tensor):
        assert softrobot_vertices.shape[-1] == 206
        if len(softrobot_vertices.shape) == 2:
            return softrobot_vertices[:, 122:124]
        elif len(softrobot_vertices.shape) == 1:
            return softrobot_vertices[122:124]
        else:
            raise Exception()

    return torch.mean((nn.MSELoss(reduction='none'))(tip_position(forward_mean), tip_position(target)), 1)


def inversion_loss(
            forward_mean: Tensor,
            target: Tensor
        ) -> Tensor:

    return (nn.MSELoss())(forward_mean, target)


def no_reduction_inversion_loss(
            forward_mean: Tensor,
            target: Tensor
        ) -> Tensor:

    return torch.mean((nn.MSELoss(reduction='none'))(forward_mean, target), 1)
