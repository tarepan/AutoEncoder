"""Domain"""


from torch import Tensor # pyright: ignore [reportUnknownVariableType] ; because of PyTorch ; pylint: disable=no-name-in-module


# Data batch

## :: (B, C=1, W, H) - Batch of gray images
ImageBatched = Tensor
## :: (B,) - Batch of image class number
NumBatched = Tensor

## the batch
ImageNumBatch = tuple[ImageBatched, NumBatched]
