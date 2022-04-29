import scipy.io
import torch
from torch import Tensor
from torch.nn import Module


def save_model(
            model: Module,
            path: str
        ) -> None:

    print("Saving Model ...")
    torch.save(model.state_dict(), path)
    print("Model Saved.")
    print()
    print()


def load_model(
            model: Module,
            path: str
        ) -> Module:

    print("Loading Model ...")
    model.load_state_dict(torch.load(path))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    print("Model Loaded.")
    print()
    print()
    return model


def save_tensor_to_mat_file(
            tensor: Tensor,
            path: str,
            key: str
        ) -> None:

    scipy.io.savemat(path, {key: tensor.detach().to('cpu').numpy()})
