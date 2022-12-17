from torch.utils.data import Dataset

import pandas as pd
from astropy.io import fits
import numpy as np
import torch
import cv2

class myDataset(Dataset):
    def __init__(self, csv_file,):
        self.image = pd.read_csv(csv_file,usecols=[2], header=0,)
        self.image=self.image.values
        self.image=self.image.reshape(-1)
        self.label=pd.read_csv(csv_file,usecols=[3], header=0)
        self.label=self.label.values
        self.label=self.label.reshape(-1)



    def __len__(self):
        return len(self.image)
    def add_mask(self,img):
        center=np.random.randint(20,70,dtype=int)
        img[center:center+10,center:center+10]=0
        return img
    def __getitem__(self, idx):
        img=fits.getdata(self.image[idx])
        img = img.astype(np.float32)
        img = cv2.resize(img, (100, 100))
        img=img.reshape(1,100,100)
        img = (img - img.mean()) / (img.std())
        flag=np.random.randint(2)
        if flag == 1 :
            img=self.add_mask(img)
        img = torch.from_numpy(img)
        if self.label[idx]==1:
            label = np.array([1,0])
        elif self.label[idx]==0:
            label = np.array([0, 1])
        label=label.astype(np.float32)

        label = torch.from_numpy(label)
        return img,label
class my_test_Dataset(Dataset):
    def __init__(self, list,):
        self.image = list


    def __len__(self):
        return len(self.image)
    def __getitem__(self, idx):
        name=self.image[idx]
        img=fits.getdata(self.image[idx])
        img = img.astype(np.float32)
        img = cv2.resize(img, (100, 100))
        img = (img - img.min()) / (img.max()-img.min())
        img=img.reshape(1,100,100)
        return img,name,