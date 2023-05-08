"Corpus splitting"


from dataclasses import dataclass

from omegaconf import MISSING, SI
from speechcorpusy.interface import ConfCorpus # pyright: ignore [reportMissingTypeStubs]; bacause of library
import torchvision as tv                       # pyright: ignore [reportMissingTypeStubs]; bacause of library
import torch


@dataclass
class ConfCorpora:
    """Configuration of Corpora.

    Args:
        root - Corpus data root
        n_val - The number of validation items, for corpus split
        n_test - The number of test items, for corpus split
    """
    root: str = MISSING
    train: ConfCorpus = ConfCorpus(
        root=SI("${..root}"))
    val: ConfCorpus = ConfCorpus(
        root=SI("${..root}"))
    test: ConfCorpus = ConfCorpus(
        root=SI("${..root}"))
    n_val: int = MISSING
    n_test: int = MISSING

def prepare_corpora(conf: ConfCorpora) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Instantiate corpuses and split them for datasets.

    Returns - CorpusItems for train/val/test
    """

    # Instantiation
    ## No needs of content init. It is a duty of consumer (Dataset).
    assert conf.train.root is not None
    assert conf.val.root   is not None
    assert conf.train.name == "MNIST", f"Currently not supporting '{conf.train.name}' corpus."
    assert conf.val.name   == "MNIST", f"Currently not supporting '{conf.train.name}' corpus."
    assert conf.test.name  == "MNIST", f"Currently not supporting '{conf.train.name}' corpus."
    corpus_train   = tv.datasets.MNIST(root=conf.train.root, train=True,  download=conf.train.download, transform=tv.transforms.ToTensor())
    corpus_valtest = tv.datasets.MNIST(root=conf.val.root,   train=False, download=conf.val.download,   transform=tv.transforms.ToTensor())

    # Split
    #                                                 label  0    1    2    3    4    5    6    7    8    9
    corpus_val  = torch.utils.data.Subset(corpus_valtest, [  3,   2,   1,  18,   4,   8,  11,   0,  61,   7])
    corpus_test = torch.utils.data.Subset(corpus_valtest, [ 69,  74,  72,  63,  65, 102,  66,  64,  84,  62])

    return corpus_train, corpus_val, corpus_test
