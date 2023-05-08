"""Data wrapper by PL-datamodule"""


from dataclasses import dataclass

import lightning as L                                                # pyright: ignore [reportMissingTypeStubs]
from omegaconf import MISSING, SI
from speechdatasety.helper.loader import generate_loader, ConfLoader # pyright: ignore [reportMissingTypeStubs]
from torch.utils.data import DataLoader

from .domain import ImageNumDatum
from .dataset import ConfHogeFugaDataset
from .corpus import prepare_corpora, ConfCorpora


@dataclass
class ConfData:
    """Configuration of the Data.
    """
    adress_data_root: str | None = MISSING
    corpus: ConfCorpora = ConfCorpora(
        root=SI("${..adress_data_root}"))
    dataset: ConfHogeFugaDataset = ConfHogeFugaDataset(
        adress_data_root=SI("${..adress_data_root}"))
    loader: ConfLoader = ConfLoader()

class Data(L.LightningDataModule):
    """Data wrapper.
    """
    def __init__(self, conf: ConfData):
        super().__init__()
        self._conf = conf

    # def prepare_data(self) -> None:
    #     """(PL-API) Prepare data in dataset.
    #     """
    #     pass

    def setup(self, stage: str | None = None) -> None:
        """(PL-API) Setup train/val/test datasets.
        """

        dataset_train, dataset_val, dataset_test = prepare_corpora(self._conf.corpus)

        # (image, target)
        if stage == "fit" or stage is None:
            self.dataset_train = dataset_train
            self.dataset_val   = dataset_val
        if stage == "test" or stage is None:
            self.dataset_test  = dataset_test

    def train_dataloader(self) -> DataLoader[ImageNumDatum]:
        """(PL-API) Generate training dataloader."""
        return generate_loader(self.dataset_train, self._conf.loader, "train")

    def val_dataloader(self) -> DataLoader[ImageNumDatum]:
        """(PL-API) Generate validation dataloader."""
        return generate_loader(self.dataset_val,   self._conf.loader, "val")

    def test_dataloader(self) -> DataLoader[ImageNumDatum]:
        """(PL-API) Generate test dataloader."""
        return generate_loader(self.dataset_test,  self._conf.loader, "test")
