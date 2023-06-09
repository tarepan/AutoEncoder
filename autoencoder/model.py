"""The model"""


from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import torchvision as tv                    # pyright: ignore [reportMissingTypeStubs]
import lightning as L                       # pyright: ignore [reportMissingTypeStubs]
from omegaconf import MISSING

from .domain import ImageNumBatch
from .data.domain import Image
from .data.transform import ConfTransform, augment, collate, load_raw, preprocess
from .networks.network import Autoencoder, ConfAutoencoder


@dataclass
class ConfOptim:
    """Configuration of optimizer"""
    learning_rate:    float = MISSING # Optimizer learning rate
    sched_decay_rate: float = MISSING # LR shaduler decay rate
    sched_decay_step: int   = MISSING # LR shaduler decay step

@dataclass
class ConfModel:
    """Configuration of the Model.
    """
    net: ConfAutoencoder = ConfAutoencoder()
    optim: ConfOptim = ConfOptim()
    transform: ConfTransform = ConfTransform()

class Model(L.LightningModule):
    """The model.
    """

    def __init__(self, conf: ConfModel):
        super().__init__()
        self.save_hyperparameters()
        self._conf = conf
        self.net = Autoencoder(conf.net)

    def forward(self, batch: ImageNumBatch): # pyright: ignore [reportIncompatibleMethodOverride] ; pylint: disable=arguments-differ
        """(PL API) Run inference toward a batch.
        """
        image, _ = batch

        # Inference :: (Batch, C=1, W, H) -> (Batch, C=1, W, H)
        return self.net.generate(image)

    # Typing of PL step API is poor. It is typed as `(self, *args, **kwargs)`.
    def training_step(self, batch: ImageNumBatch): # pyright: ignore [reportIncompatibleMethodOverride] ; pylint: disable=arguments-differ
        """(PL API) Train the model with a batch.
        """

        image_gt, _ = batch

        # Forward :: (B, C*W*H) -> (B, C*W*H)
        image_pred = self.net(image_gt)

        # Loss
        loss = F.mse_loss(image_gt, image_pred)

        self.log('loss', loss) #type: ignore ; because of PyTorch-Lightning
        return {"loss": loss}

    def validation_step(self, batch: ImageNumBatch, batch_idx: int): # pyright: ignore [reportIncompatibleMethodOverride] ; pylint: disable=arguments-differ,unused-argument
        """(PL API) Validate the model with a batch.
        """

        image_gt, num_gt = batch

        # Forward :: (B, C*W*H) -> (B, C*W*H)
        image_pred = self.net(image_gt)
        image_pred_unflatten = self.net.generate(image_gt)

        # Loss
        loss_fwd = F.mse_loss(image_gt, image_pred)

        # Logging
        grid = tv.utils.make_grid(image_pred_unflatten)
        self.logger.experiment.add_image(f"Reconstructed {num_gt.item()}", grid, global_step=self.global_step)

        return {
            "val_loss": loss_fwd,
        }

    # def test_step(self, batch, batch_idx: int): # pyright: ignore [reportIncompatibleMethodOverride] ; pylint: disable=arguments-differ
    #     """(PL API) Test a batch. If not provided, test_step == validation_step."""
    #     return anything_for_`test_epoch_end`

    def configure_optimizers(self): # type: ignore ; because of PyTorch-Lightning (no return typing, so inferred as Void)
        """(PL API) Set up a optimizer.
        """
        conf = self._conf.optim

        optim = Adam(self.net.parameters(), lr=conf.learning_rate)
        sched = {
            "scheduler": StepLR(optim, conf.sched_decay_step, conf.sched_decay_rate),
            "interval": "step",
        }

        return {
            "optimizer": optim,
            "lr_scheduler": sched,
        }

    # def predict_step(self, batch: HogeFugaBatch, batch_idx: int): # pyright: ignore [reportIncompatibleMethodOverride] ; pylint: disable=arguments-differ
    #     """(PL API) Run prediction with a batch. If not provided, predict_step == forward."""
    #     return pred

    def sample(self) -> Image:
        """Acquire sample input toward preprocess."""

        # Audio Example (librosa is not handled by this template)
        import librosa # pyright: ignore [reportMissingImports, reportUnknownVariableType] ; pylint: disable=import-outside-toplevel,import-error
        path: Path = librosa.example("libri2") # pyright: ignore [reportUnknownMemberType, reportUnknownVariableType]

        return load_raw(self._conf.transform.load, path)

    def load(self, path: Path) -> Image:
        """Load raw inputs.
        Args:
            path - Path to the input.
        """
        return load_raw(self._conf.transform.load, path)

    def preprocess(self, piyo: Image, to_device: str | None = None) -> ImageNumBatch:
        """Preprocess raw inputs into model inputs for inference."""

        conf = self._conf.transform
        hoge_fuga = preprocess(conf.preprocess, piyo)
        hoge_fuga_datum = augment(conf.augment, hoge_fuga)
        batch = collate([hoge_fuga_datum])

        # To device
        device = torch.device(to_device) if to_device else torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        batch = (batch[0].to(device), batch[1].to(device), batch[2])

        return batch
