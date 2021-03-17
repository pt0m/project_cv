import os
import torch
from skimage import io, transform
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class data_set(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, name, root_dir, transform=None):
        """
        Args:
            csv_file (string): name of the sequence
            root_dir (string): Directory with all the sequence.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        length = {'bag': 196, 'bear': 26, 'book': 51, 'camel': 90, 'rhino': 90, 'swan': 50}
        frames = [name+'-%0*d.bmp'%(3, im_begin) for i in range(1,length[name]+1)]
        masks =  [name+'-%0*d.png'%(3, im_begin) for i in range(1,length[name]+1)]

        self.data  = pd.DataFrame(d={'frames' : frames, 'masks': masks})


        self.path = root_dir

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,self.data.iloc[idx, 0])
        frame = io.imread(img_name)
        mask_name = os.path.join(self.root_dir,self.data.iloc[idx, 1])
        mask =  io.imread(mask_name)
        bb = self.getBB(mask)
        sample = {'frame': frame, 'mask': mask, 'boundbox': bb}

        if self.transform:
            sample = self.transform(sample)

        return sample(idx)

    def getBB(mask):
        (n,m) = mask.shape
        y_max, y_min = m-1,0
        x_max, x_min = n-1,0
        while 255 not in mask[y_min,:] and y_min < m:
            y_min+=1
        while 255 not in mask[:,x_min] and x_min < n:
            x_min+=1
        while 255 not in mask[y_max,:] and y_max >= 0:
            y_max-=1
        while 255 not in mask[:,x_max] and x_max >= 0:
            x_max-=1
        return y_max, x_max, y_min, x_min
