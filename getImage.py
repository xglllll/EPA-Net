from utils.data_RGB import get_validation_data
from skimage import img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio as PSNR
import numpy as np
import os
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import utils
from logger import *
import yaml
from PIL import Image
import torchvision

toTensor = torchvision.transforms.ToTensor()
toPil = torchvision.transforms.ToPILImage()

with open('test.yml', mode='r') as f_yml:
    Loader, _ = ordered_yaml()
    opt = yaml.load(f_yml, Loader=Loader)

gpus = ','.join([str(i) for i in opt['GPU']])

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus

model_restoration = utils.get_arch(opt['MODEL'])
model_restoration.eval().cuda()

dir_name = os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.join(dir_name, 'log', opt['MODEL']['NAME'] + '_' + opt['MODEL']['MODE'])
model_dir  = os.path.join(log_dir, 'models')
path_chk_rest = os.path.join(model_dir, opt['VAL']['PRETRAIN_MODEL'])
save_dir = os.path.join(dir_name, 'save')
utils.load_checkpoint(model_restoration, path_chk_rest)
val_epoch = utils.load_start_epoch(path_chk_rest)

img = Image.open('/home/yoga/save_pth/xgl2/EPANet/dxn.jpg')

# img.show()
imgTensor = toTensor(img).unsqueeze(0).cuda()

factor = 64
h,w = imgTensor.shape[2],imgTensor.shape[3]
H,W = ((h+factor)//factor)*factor,((w+factor)//factor)*factor
padh = H-h if h%factor!=0 else 0
padw = W-w if w%factor!=0 else 0
imgTensor = F.pad(imgTensor,(0,padw,0,padh),'reflect')
out = model_restoration(imgTensor)
print(out.shape)
# outPil = toPil(out)
# outPil.show()