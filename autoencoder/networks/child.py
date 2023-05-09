"""The Child sub-module"""


from dataclasses import dataclass

from torch import Tensor
import torch.nn as nn
from omegaconf import MISSING


@dataclass
class ConfEncoderFC:
    """Configuration of EncoderFC"""
    feat_i: int = MISSING # Feature dimension size of input
    feat_h: int = MISSING # Feature dimension size of hidden layer
    feat_o: int = MISSING # Feature dimension size of output

class EncoderFC(nn.Module):
    """The Fully-Connected Encoder, [FC-ReLU]x3"""
    def __init__(self, conf: ConfEncoderFC):
        super().__init__() # pyright: ignore [reportUnknownMemberType]; because of PyTorch

        self.net = nn.Sequential(*[
            nn.Linear(conf.feat_i, conf.feat_h), nn.ReLU(),
            nn.Linear(conf.feat_h, conf.feat_h), nn.ReLU(),
            nn.Linear(conf.feat_h, conf.feat_o), nn.ReLU(),
        ])

    def forward(self, i_pred: Tensor) -> Tensor: # pyright: ignore [reportIncompatibleMethodOverride]
        """(PT API) Forward a batch.

        Arguments:
            i_pred :: (B, Feat) - Input
        Returns:
                   :: (B, Feat) - Latent representation
        """
        return self.net(i_pred)

    def generate(self, i_pred: Tensor) -> Tensor:
        """Run inference with a batch.

        Arguments:
            i_pred :: (B, Feat) - Input
        Returns:
                   :: (B, Feat) - Latent representation
        """
        return self.forward(i_pred)


# class ConfChild:
@dataclass
class ConfDecoderFC:
    """Configuration of EncoderFC"""
    feat_i: int = MISSING # Feature dimension size of input
    feat_h: int = MISSING # Feature dimension size of hidden layer
    feat_o: int = MISSING # Feature dimension size of output

class DecoderFC(nn.Module):
    """The Fully-Connected Decoder, [FC-ReLU]x3"""
    def __init__(self, conf: ConfDecoderFC):
        super().__init__() # pyright: ignore [reportUnknownMemberType]; because of PyTorch

        self.net = nn.Sequential(*[
            nn.Linear(conf.feat_i, conf.feat_h), nn.ReLU(),
            nn.Linear(conf.feat_h, conf.feat_h), nn.ReLU(),
            nn.Linear(conf.feat_h, conf.feat_o), nn.ReLU(),
        ])

    def forward(self, i_pred: Tensor) -> Tensor: # pyright: ignore [reportIncompatibleMethodOverride]
        """(PT API) Forward a batch.

        Arguments:
            i_pred :: (B, Feat) - Latent representation
        Returns:
                   :: (B, Feat) - Reconstructed input
        """
        return self.net(i_pred)

    def generate(self, i_pred: Tensor) -> Tensor:
        """Run inference with a batch.

        Arguments:
            i_pred :: (B, Feat) - Latent representation
        Returns:
                   :: (B, Feat) - Reconstructed input
        """
        return self.forward(i_pred)
