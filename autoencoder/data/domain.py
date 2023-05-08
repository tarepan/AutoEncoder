"""Data domain"""


import numpy as np
from numpy.typing import NDArray


# `XX_` is for typing

# Statically-preprocessed item
## Image :: (C=1, W, H) - Gray image
Image = NDArray[np.float32]
## Num :: (1,) - Number of image, 0~9
Num = NDArray[np.float32]
Num_: Num = np.array([1.], dtype=np.float32)
## Fuga :: (T,) - fuga fuga
Fuga = NDArray[np.float32]
Fuga_: Fuga = np.array([1.], dtype=np.float32)
## the item
ImageNum = tuple[Image, Num]
ImageNum_: ImageNum = (Num_, Fuga_)

# Dynamically-transformed Dataset datum
## ImageDatum :: (C=1, W, H) - Gray image
ImageDatum = NDArray[np.float32]
## NumDatum :: (1,) - Number of image, 0~9
NumDatum = NDArray[np.float32]
## the datum
ImageNumDatum = tuple[ImageDatum, NumDatum]
