"""The Network"""


from dataclasses import dataclass

from torch import Tensor
import torch.nn as nn
from omegaconf import MISSING, SI


from ..domain import ImageBatched
from ..data.transform import inv_flatten3dim
from .child import EncoderFC, ConfEncoderFC, DecoderFC, ConfDecoderFC


@dataclass
class ConfAutoencoder:
    """Configuration of the Network"""
    feat_io:    int = MISSING                # Feature dimension size of CWH-flatten    input/output
    channel_io: int = MISSING                # Channel dimension size of feat-unflatten input/output
    width_io:   int = MISSING                # Width   dimension size of feat-unflatten input/output
    height_io:  int = MISSING                # Height  dimension size of feat-unflatten input/output
    feat_z:     int = MISSING                # Feature dimension size of latent representation
    encoder: ConfEncoderFC = ConfEncoderFC(
        feat_i=SI("${..feat_io}"),
        feat_o=SI("${..feat_z}"),)
    decoder: ConfDecoderFC = ConfDecoderFC(
        feat_i=SI("${..feat_z}"),
        feat_o=SI("${..feat_io}"),)

class Autoencoder(nn.Module):
    """The Autoencoder"""
    def __init__(self, conf: ConfAutoencoder):
        super().__init__() # pyright: ignore [reportUnknownMemberType]; because of PyTorch

        self.enc = EncoderFC(conf.encoder)
        self.dec = DecoderFC(conf.decoder)
        self._unflatten = inv_flatten3dim(conf.channel_io, conf.width_io, conf.height_io)

    def forward(self, image: ImageBatched) -> Tensor: # pyright: ignore [reportIncompatibleMethodOverride]
        """(PT API) Forward a batch.

        Arguments:
            image :: (B, C*W*H) - Image
        Returns:
                  :: (B, C*W*H) - Reconstructed image
        """
        return self.dec(self.enc(image))

    def generate(self, image: ImageBatched) -> Tensor:
        """Run inference with a batch.

        Arguments:
            image :: (B, C*W*H) - Image
        Returns:
                  :: (B, C, W, H) - Reconstructed image, unflatten as 3D (CWH) image
        """
        # :: (B, C*W*H) -> (B, C*W*H) -> (B, C=1, W, H)
        return self._unflatten(self.dec(self.enc(image)))
