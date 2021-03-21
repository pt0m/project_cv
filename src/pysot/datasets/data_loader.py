import os
import torch
from skimage import io, transform
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from src.pysot.datasets.anchor_target import *
import random

def getBB(mask):
    (m,n) = mask.shape
    print((m,n))
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
    return x_min, y_min, x_max ,y_max




def crop_data_around(template,x,y,size):
    (x_temp, y_temp, _) = template.shape
    x_min = max(0, x-(size//2))
    x_max = x_min + size
    if x_max > x_temp:
        x_max = x_temp
        x_min = x_temp - size
    y_min = max(0, y-(size//2))
    y_max = y_min + size
    if y_max > y_temp:
        y_max = y_temp
        y_min = y_temp - size
    x_min, x_max, y_min, y_max = int(x_min), int(x_max), int(y_min), int(y_max)
    print("dx = ", x_max-x_min)
    print("dy = ", y_max-y_min)
    return template[x_min:x_max,y_min:y_max]

anchor_target = AnchorTarget()
def image_to_dict(image1,image2, image1_mask, image2_mask):
    """
    image1 : image with the template in it
    image2 : image in wich we search
    cls : class
    masks ... the masks
    """
    bb1, bb2 = getBB(image1_mask), getBB(image2_mask)
    #template
    #template = image1[bb1[1]:bb1[3], bb1[0]:bb1[2]]
    template = crop_data_around(image1,(bb1[1]+bb1[3])//2, (bb1[0]+bb1[2])//2, cfg.TRAIN.EXEMPLAR_SIZE)
    search = crop_data_around(image2,(bb1[1]+bb1[3])//2, (bb1[0]+bb1[2])//2, cfg.TRAIN.SEARCH_SIZE)
    cls, delta, delta_weight, overlap = anchor_target(target = bb1, size = cfg.TRAIN.OUTPUT_SIZE)
    return {'template': template,
                'search': search,
                'label_cls': cls,
                'label_loc': delta,
                'label_loc_weight': delta_weight,
                'bbox': np.array(bb1)}


class dataset_loader(Dataset):
    def __init__(self, name, root_dir, transform=None):
        """
        Args:
            csv_file (string): name of the sequence
            root_dir (string): Directory with all the sequence.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        length = {'bag': 196, 'bear': 26, 'book': 51, 'camel': 90, 'rhino': 90, 'swan': 50}
        self.frames = [name+'-%0*d.bmp'%(3, i) for i in range(1,length[name]+1)]
        self.masks =  [name+'-%0*d.png'%(3, i) for i in range(1,length[name]+1)]
        self.data  = pd.DataFrame({'frames' : self.frames, 'masks': self.masks})
        self.path = root_dir
        self.max_id =  length[name]
    def __getitem2__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.path + self.frames[idx] # os.path.join(self.path,self.data.iloc[idx, 0])
        frame = io.imread(img_name)
        mask_name =self.path + self.masks[idx]# os.path.join(self.path,self.data.iloc[idx, 1])
        mask =  io.imread(mask_name)
        bb = self.getBB(mask)
        sample = {'frame': frame, 'mask': mask, 'boundbox': bb}
        return sample

    def getBB(self,mask):
        (m,n) = mask.shape
        print((m,n))
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
        return x_min, y_min, x_max ,y_max

    def __len__(self):
        return self.max_id
    def __getitem__(self,index):
        if(index > 0):
            index_2 = 0
        else:
             index_2 = 1
        id1 = np.random.random_integers(0, self.max_id-1)
        id2 = np.random.random_integers(0, self.max_id-1)
        spl1 = self.__getitem2__(id1)
        spl2 = self.__getitem2__(id2)
        image1 = spl1["frame"]
        mask1 = spl1["mask"]
        image2 = spl2["frame"]
        mask2 = spl2["mask"]
        dict = image_to_dict(image1, image2, mask1, mask2)
        dict['template'] = dict['template'].transpose((2, 0, 1)).astype(np.float32)
        dict['search'] = dict['search'].transpose((2, 0, 1)).astype(np.float32)
        print("template size=", dict['template'].shape)
        print("seach size   =", dict['search'].shape)
        return {
            'template': torch.as_tensor(dict['template'], dtype=torch.double),
            'search': torch.as_tensor(dict['search'], dtype=torch.double),
            'label_cls': torch.from_numpy(dict['label_cls']),
            'label_loc': torch.from_numpy(dict['label_loc']),
            'label_loc_weight': torch.from_numpy(dict['label_loc_weight']),
            'bbox': torch.from_numpy(np.array(dict['bbox']))
        }
