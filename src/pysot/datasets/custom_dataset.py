import os
import torch


class TrackerDataset(torch.utils.data.Dataset):

    def __init__(self, rootdir: str) -> None:
        self.rootdir = rootdir
        self.frames  = sorted(list(filter(lambda x: x.endswith('bmp'), os.listdir(rootdir))))
        self.masks   = sorted(list(filter(lambda x: x.endswith('png'), os.listdir(rootdir))))