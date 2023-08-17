import torch.utils.data as data
import numpy as np
import glob
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


class NoPositiveGraspsException(Exception):
    """raised when there's no positive grasps for an object."""
    pass


class BaseDataset(data.Dataset):
    def __init__(self,
                 opt,
):
        super(BaseDataset, self).__init__()
        self.opt = opt
        self.caching = opt.caching
        self.cache = {}

    def make_dataset(self):
        files = glob.glob(self.opt.dataset_root_folder+"/*")
        return files


def collate_fn(batch):
    """Creates mini-batch tensors
    We should build custom collate_fn rather than using default collate_fn
    """
    batch = list(filter(lambda x: x is not None, batch))  #
    meta = {}
    keys = batch[0].keys()
    for key in keys:
        meta.update({key: np.concatenate([d[key] for d in batch])})
    return meta
