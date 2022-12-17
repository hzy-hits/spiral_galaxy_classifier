from model import ResNet18
import torch
import numpy as np
import cv2 as cv
from data_loader import my_test_Dataset
import glob
from dataloader import test
path='D:\\study\\senior_fall\\thesis\\pre_project\\galaxy_figure\\Nicola_fits\\'
dir=['fits_xy_0.200','fits_yz_0.100','fits_yz_0.200','fits_zx_0.100','fits_zx_0.200','fits_xy_0.010','fits_yz_0.010','fits_zx_0.010','fits_xy_0.100']
device=torch.device('cuda:0')
model=ResNet18(classes_num=2).to(device)
weights_dict = torch.load('model/select_spiral.pth', map_location=device)
model.load_state_dict(weights_dict,)
for spe_path in dir:
    file=glob.glob(path+spe_path+'\\*.fits')
    test_data=my_test_Dataset(file)
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=0,
        prefetch_factor=2,
    )
    result = test(model=model,
                      data_loader=test_loader,
                      device=device,)
    np.savetxt('.\\txt\\'+spe_path+'.txt',result,fmt='%s')