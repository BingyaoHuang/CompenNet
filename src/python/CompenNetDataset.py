'''
CompenNet data loader
'''

import os
from torch.utils.data import Dataset
import cv2 as cv
from utils import fullfile


# Use Pytorch multi-threaded dataloader and opencv to load image faster
class SimpleDataset(Dataset):
    """Simple dataset."""

    def __init__(self, data_root, index=None, size=None):
        self.data_root = data_root
        self.size = size

        # img list
        img_list = sorted(os.listdir(data_root))
        img_list = [img_list[x] for x in index] if index is not None else img_list

        self.img_names = [fullfile(self.data_root, name) for name in img_list]

        # assert len(self.img_names) == len(index), print(
        #     'Dataset Error: image numbers does not match index length!')

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        if self.size is not None:
            im = cv.cvtColor(cv.resize(cv.imread(self.img_names[idx]), self.size[::-1]), cv.COLOR_BGR2RGB)
        else:
            im = cv.cvtColor(cv.imread(self.img_names[idx]), cv.COLOR_BGR2RGB)
        return im