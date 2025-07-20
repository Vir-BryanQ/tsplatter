import abc
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from typing import Literal
from enum import Enum

class L1(nn.Module):
    """L1 loss"""

    def __init__(
        self, implementation: Literal["scalar", "per-pixel"] = "scalar", **kwargs
    ):
        super().__init__()
        self.implementation = implementation

    def forward(self, pred, gt):
        if self.implementation == "scalar":
            return torch.abs(pred - gt).mean()
        else:
            return torch.abs(pred - gt)

class TVLoss(nn.Module):
    """TV loss"""

    def __init__(self):
        super().__init__()

    def forward(self, pred):
        """
        Args:
            pred: [batch, H, W, 3]

        Returns:
            tv_loss: [batch]
        """
        h_diff = pred[..., :, :-1, :] - pred[..., :, 1:, :]
        w_diff = pred[..., :-1, :, :] - pred[..., 1:, :, :]
        return torch.mean(torch.abs(h_diff)) + torch.mean(torch.abs(w_diff))

class NormalLossType(Enum):
    """Enum for specifying depth loss"""

    L1 = "L1"
    Smooth = "Smooth"


class NormalLoss(nn.Module):
    """Factory method class for various depth losses"""

    def __init__(self, normal_loss_type: NormalLossType, **kwargs):
        super().__init__()
        self.normal_loss_type = normal_loss_type
        self.kwargs = kwargs
        self.loss = self._get_loss_instance()

    @abc.abstractmethod
    def forward(self, *args) -> Tensor:
        return self.loss(*args)

    def _get_loss_instance(self) -> nn.Module:
        if self.normal_loss_type == NormalLossType.L1:
            return L1(**self.kwargs)
        elif self.normal_loss_type == NormalLossType.Smooth:
            return TVLoss(**self.kwargs)
        else:
            raise ValueError(f"Unsupported loss type: {self.normal_loss_type}")