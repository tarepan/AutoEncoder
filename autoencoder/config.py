"""Whole Configuration"""


from dataclasses import dataclass

from omegaconf import MISSING
from configen import generate_conf_loader # pyright: ignore [reportMissingTypeStubs]
from lightlightning import ConfTrain      # pyright: ignore [reportMissingTypeStubs]

from .data.transform import ConfTransform
from .data.datamodule import ConfData
from .model import ConfModel


CONF_DEFAULT_STR = """
seed: 1234
path_extend_conf: null
transform:
    load:
        sampling_rate: 16000
    preprocess:
        piyo2hoge:
            amp: 1.2
        piyo2fuga:
            div: 3.0
    augment:
        len_clip: 10
model:
    net:
        feat_io: 784
        feat_z: 20
        channel_io: 1
        width_io: 28
        height_io: 28
        encoder:
            feat_h: 100
        decoder:
            feat_h: 100
    optim:
        learning_rate: 0.01
        sched_decay_rate: 0.01
        sched_decay_step: 1000
    transform: "${transform}"
data:
    adress_data_root: ""
    corpus:
        train:
            name: "MNIST"
            download: True
        val:
            name: "MNIST"
            download: True
        test:
            name: "MNIST"
            download: True
        n_val: 1
        n_test: 1
    dataset:
        attr1: 1
        transform: "${transform}"
    loader:
        batch_size_train: 100
        batch_size_val: 1
        batch_size_test: 1
        num_workers: null
        pin_memory: null
train:
    gradient_clipping: null
    max_epochs: 10000
    val_interval_epoch: 100
    profiler: null
    ckpt_log:
        dir_root: "."
        name_exp: "default"
        name_version: "version_0"
"""

@dataclass
class ConfGlobal:
    """Configuration of everything.
    Args:
        seed: PyTorch-Lightning's seed for every random system
        path_extend_conf: Path of configuration yaml which extends default config
    """
    seed: int = MISSING
    path_extend_conf: str | None = MISSING
    transform: ConfTransform = ConfTransform()
    model: ConfModel = ConfModel()
    data: ConfData = ConfData()
    train: ConfTrain = ConfTrain()


# Exported
load_conf = generate_conf_loader(CONF_DEFAULT_STR, ConfGlobal)
"""Load configuration type-safely.
"""
