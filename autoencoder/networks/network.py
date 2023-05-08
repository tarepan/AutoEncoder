"""The Network"""


from dataclasses import dataclass

from torch import Tensor
import torch.nn as nn
from omegaconf import MISSING, SI

from ..domain import ImageBatched
from .child import Child, ConfChild


@dataclass
class ConfNetwork:
    """Configuration of the Network.

    Args:
        dim_i: Dimension size of input
        dim_i: Dimension size of output
    """
    dim_i: int = MISSING
    dim_o: int = MISSING
    child: ConfChild = ConfChild(
        dim_i=SI("${..dim_i}"),
        dim_o=SI("${..dim_o}"),)

class Network(nn.Module):
    """The Network.
    """
    def __init__(self, conf: ConfNetwork):
        super().__init__()

        # Submodule
        self._child = Child(conf.child)

    # Typing of PyTorch forward API is poor.
    def forward(self, image: ImageBatched) -> Tensor: # pyright: ignore [reportIncompatibleMethodOverride]
        """(PT API) Forward a batch.

        Arguments:
            image :: (B, C=1, W, H) - Input image
        Returns:
            o_pred :: (B, C=1, W, H) - Reconstructed image
        """

        # :: (B, C=1, W, H) -> (B, C=1, W, H)
        o_pred = self._child(image)

        return o_pred

    def generate(self, image: ImageBatched) -> Tensor:
        """Run inference with a batch.

        Arguments:
            image :: (B, C=1, W, H) - Input image
        Returns:
            o_pred :: (B, C=1, W, H) - Reconstructed image
        """

        # :: (B, C=1, W, H) -> (B, C=1, W, H)
        o_pred = self._child(image)

        return o_pred
