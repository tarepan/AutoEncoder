"""Data transformation"""


from dataclasses import dataclass
from pathlib import Path

import numpy as np
from omegaconf import MISSING
from torch import from_numpy, stack # pyright: ignore [reportUnknownVariableType] ; because of PyTorch ; pylint: disable=no-name-in-module

from ..domain import NumBatched, ImageBatched, ImageNumBatch
from .domain import NumDatum, ImageDatum, ImageNum, ImageNumDatum, Image, Num, Fuga


# [Data transformation]
#
#      load        preprocessing            augmentation              collation
#     -----> raw -----------------> item -----------------> datum -----------------> batch
#                 before training            in Dataset             in DataLoader

###################################################################################################################################
# [Load]
"""
(delele here when template is used)

[Design Notes - Load as transformation]
    Loading determines data shape, and load utilities frequently modify the data.
    In this meaning, load has similar 'transform' funtionality.
    For this reason, `load_raw` is placed here.
"""

@dataclass
class ConfLoad:
    """
    Configuration of piyo loading.
    Args:
        sampling_rate - Sampling rate
    """
    sampling_rate: int = MISSING

def load_raw(conf: ConfLoad, path: Path) -> Image:
    """Load raw data 'piyo' from the adress."""

    # Audio Example (librosa is not handled by this template)
    import librosa # pyright: ignore [reportMissingImports, reportUnknownVariableType] ; pylint: disable=import-outside-toplevel,import-error
    piyo: Image = librosa.load(path, sr=conf.sampling_rate, mono=True)[0] # pyright: ignore [reportUnknownMemberType]

    return piyo

###################################################################################################################################
# [Preprocessing]

@dataclass
class ConfPiyo2Hoge:
    """
    Configuration of piyo-to-hoge preprocessing.
    Args:
        amp - Amplification factor
    """
    amp: float = MISSING

def piyo_to_hoge(conf: ConfPiyo2Hoge, piyo: Image) -> Num:
    """Convert piyo to hoge.
    """
    # Amplification :: (T,) -> (T,)
    hoge: Num = piyo * conf.amp

    return hoge


@dataclass
class ConfPiyo2Fuga:
    """
    Configuration of piyo-to-fuga preprocessing.
    Args:
        div - Division factor
    """
    div: float = MISSING

def piyo_to_fuga(conf: ConfPiyo2Fuga, piyo: Image) -> Fuga:
    """Convert piyo to fuga.
    """
    # Division :: (T,) -> (T,)
    fuga: Fuga = piyo / conf.div

    return fuga

@dataclass
class ConfPreprocess:
    """
    Configuration of item-to-datum augmentation.
    Args:
        len_clip - Length of clipping
    """
    piyo2hoge: ConfPiyo2Hoge = ConfPiyo2Hoge()
    piyo2fuga: ConfPiyo2Fuga = ConfPiyo2Fuga()

def preprocess(conf: ConfPreprocess, raw: Image) -> ImageNum:
    """Preprocessing (raw_to_item) - Process raw data into item.

    Piyo -> Hoge & Fuga
    """
    return piyo_to_hoge(conf.piyo2hoge, raw), piyo_to_fuga(conf.piyo2fuga, raw)

###################################################################################################################################
# [Augmentation]

@dataclass
class ConfAugment:
    """
    Configuration of item-to-datum augmentation.
    Args:
        len_clip - Length of clipping
    """
    len_clip: int = MISSING

def augment(conf: ConfAugment, hoge_fuga: ImageNum) -> ImageNumDatum:
    """Augmentation (item_to_datum) - Dynamically modify item into datum.

    Clipping + DimensionExpansion
    """
    hoge, fuga = hoge_fuga

    # Clipping
    ## :: (T=t,) -> (T=L,)
    hoge = hoge[:conf.len_clip]
    ## :: (T=t,) -> (T=L,)
    fuga = fuga[:conf.len_clip]

    # Dimension expansion
    ## :: (T,) -> (T, 1)
    hoge_datum: ImageDatum = np.expand_dims(hoge, axis=-1) # pyright: ignore [reportUnknownMemberType] ; because of numpy
    ## :: (T,) -> (T, 1)
    fuga_datum: NumDatum = np.expand_dims(fuga, axis=-1) # pyright: ignore [reportUnknownMemberType]; because of numpy

    return hoge_datum, fuga_datum

###################################################################################################################################
# [collation]

def collate(datums: list[ImageNumDatum]) -> ImageNumBatch:
    """Collation (datum_to_batch) - Bundle multiple datum into a batch."""

    hoge_batched: ImageBatched = stack([from_numpy(datum[0]) for datum in datums])
    fuga_batched: NumBatched = stack([from_numpy(datum[1]) for datum in datums])

    return hoge_batched, fuga_batched

###################################################################################################################################

@dataclass
class ConfTransform:
    """Configuration of data transform."""
    load: ConfLoad = ConfLoad()
    preprocess: ConfPreprocess = ConfPreprocess()
    augment: ConfAugment = ConfAugment()
