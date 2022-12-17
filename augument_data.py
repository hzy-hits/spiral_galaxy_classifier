import numpy as np
import os
import cv2
import uuid
array = [ "0", "1", "2", "3", "4", "5","6", "7", "8", "9","a", "b", "c", "d", "e", "f","g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s","t", "u", "v", "w", "x", "y", "z",
"A", "B", "C", "D", "E", "F", "G", "H", "I","J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V","W", "X", "Y", "Z"]
def get_short_id():
    id = str(uuid.uuid4()).replace("-", '') # 注意这里需要用uuid4
    buffer = []
    for i in range(0, 8):
        start = i *  4
        end = i * 4 + 4
        val = int(id[start:end], 16)
        buffer.append(array[val % 62])
    return "".join(buffer)

import pandas as pd
from astropy.io import fits
path = "D:\\study\\大四上\\thesis\\pre_project\\galaxy_figure\\"
def augument_data(csv_file):
    data=pd.read_csv(csv_file,header=0)

class Origin_img:
    def __init__(self,path,label,save_path):
        self.image=fits.getdata(path)
        self.label=int(label)
        self.flip_label=self.label*-1
        self.save_path=save_path
        self.iter=0
    def save_fits(self,img,label2):

        name=self.getname()
        grey = fits.PrimaryHDU([label2] )
        label1 = fits.ImageHDU(img)
        hdu = fits.HDUList([grey,label1])
        hdu.writeto(os.path.join(self.save_path,name))
        self.iter=self.iter+1
    def rotate(self,img,flag):
        img=np.rot90(img,flag)
        return img
    def flipimage(self,img):
        self.save_fits(img, self.label)
        img=cv2.flip(img,1)
        self.save_fits(img,self.flip_label)

    def addsalt_pepper(self,img, SNR=0.96):
        img_ = img
        mask = np.random.choice((0,1, 2), size=(img.shape[0], img.shape[1]), p=[SNR, (1 - SNR) / 2., (1 - SNR) / 2.])
        img_[mask == 1] = 1  # 盐噪声
        img_[mask == 2] = 0  # 椒噪声
        return img_
    def add_noise(self,img):
        image = np.array(img/255, dtype=float)  # 将像素值归一
        out0=self.addsalt_pepper(image)
        noise = np.random.normal(0.3*image.mean(), 0.1*image.std(), image.shape)  # 产生高斯噪声
        out = (image + noise)
        out=np.clip(out, 0, 1.0)
        out1=out * 255
        out0=out0 * 255
        self.flipimage(out1)
        self.flipimage(out0)
    def getname(self):
        name=get_short_id()
        return name+'.fits'



    def augument_data(self,):
        for i in range (4):
            img=self.rotate(self.image,i)
            self.flipimage(img)
            self.add_noise(img)


train_data=pd.read_csv('D:\\study\\senior_fall\\thesis\\pre_model\\txt\\pre_train_data.csv',header=0)
train_data=train_data.values
train_save_path ='formal/train'
test_save_path='formal/test'
test_data=pd.read_csv('D:\\study\\senior_fall\\thesis\\pre_model\\txt\\pre_test_data.csv',header=0)
test_data=test_data.values
#for x in test_data:
    #aug_data=Origin_img(x[0],x[1],test_save_path)
    #aug_data.augument_data()
for x in train_data:
    aug_data=Origin_img(x[0],x[1],train_save_path)
    aug_data.augument_data()





