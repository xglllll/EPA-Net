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
import cv2

def _generate_3d_gaussian_kernel():
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    kernel_3 = cv2.getGaussianKernel(11, 1.5)
    kernel = torch.tensor(np.stack([window * k for k in kernel_3], axis=0))
    conv3d = torch.nn.Conv3d(1, 1, (11, 11, 11), stride=1, padding=(5, 5, 5), bias=False, padding_mode='replicate')
    conv3d.weight.requires_grad = False
    conv3d.weight[0, 0, :, :, :] = kernel
    return conv3d

def _3d_gaussian_calculator(img, conv3d):
    out = conv3d(img.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
    return out

def _ssim_3d(img1, img2, max_value):
    assert len(img1.shape) == 3 and len(img2.shape) == 3
    """Calculate SSIM (structural similarity) for one channel images.

    It is called by func:`calculate_ssim`.

    Args:
        img1 (ndarray): Images with range [0, 255]/[0, 1] with order 'HWC'.
        img2 (ndarray): Images with range [0, 255]/[0, 1] with order 'HWC'.

    Returns:
        float: ssim result.
    """
    C1 = (0.01 * max_value) ** 2
    C2 = (0.03 * max_value) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    kernel = _generate_3d_gaussian_kernel().cuda()

    img1 = torch.tensor(img1).float().cuda()
    img2 = torch.tensor(img2).float().cuda()


    mu1 = _3d_gaussian_calculator(img1, kernel)
    mu2 = _3d_gaussian_calculator(img2, kernel)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = _3d_gaussian_calculator(img1 ** 2, kernel) - mu1_sq
    sigma2_sq = _3d_gaussian_calculator(img2 ** 2, kernel) - mu2_sq
    sigma12 = _3d_gaussian_calculator(img1*img2, kernel) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))
    return float(ssim_map.mean())

def calculate_ssim(img1,
                   img2,
                   input_order='HWC'):
    """Calculate SSIM (structural similarity).

    Ref:
    Image quality assessment: From error visibility to structural similarity

    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.

    For three-channel images, SSIM is calculated for each channel and then
    averaged.

    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the SSIM calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: ssim result.
    """

    assert img1.shape == img2.shape, (
        f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            '"HWC" and "CHW"')

    if type(img1) == torch.Tensor:
        if len(img1.shape) == 4:
            img1 = img1.squeeze(0)
        img1 = img1.detach().cpu().numpy().transpose(1,2,0)
    if type(img2) == torch.Tensor:
        if len(img2.shape) == 4:
            img2 = img2.squeeze(0)
        img2 = img2.detach().cpu().numpy().transpose(1,2,0)

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    ssims = []
    # ssims_before = []

    # skimage_before = skimage.metrics.structural_similarity(img1, img2, data_range=255., multichannel=True)
    # print('.._skimage',
    #       skimage.metrics.structural_similarity(img1, img2, data_range=255., multichannel=True))
    max_value = 1 if img1.max() <= 1 else 255
    with torch.no_grad():
        final_ssim = _ssim_3d(img1, img2, max_value)
        ssims.append(final_ssim)

    # for i in range(img1.shape[2]):
    #     ssims_before.append(_ssim(img1, img2))

    # print('..ssim mean , new {:.4f}  and before {:.4f} .... skimage before {:.4f}'.format(np.array(ssims).mean(), np.array(ssims_before).mean(), skimage_before))
        # ssims.append(skimage.metrics.structural_similarity(img1[..., i], img2[..., i], multichannel=False))

    return np.array(ssims).mean()


with open('test.yml', mode='r') as f_yml:
    Loader, _ = ordered_yaml()
    opt = yaml.load(f_yml, Loader=Loader)


gpus = ','.join([str(i) for i in opt['GPU']])

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus

model_restoration = utils.get_arch(opt['MODEL'])

dir_name = os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.join(dir_name, 'log', opt['MODEL']['NAME'] + '_' + opt['MODEL']['MODE'])
result_dir = os.path.join(log_dir, 'results')
model_dir  = os.path.join(log_dir, 'models')


path_chk_rest = os.path.join(model_dir, opt['VAL']['PRETRAIN_MODEL'])
utils.load_checkpoint(model_restoration, path_chk_rest)
val_epoch = utils.load_start_epoch(path_chk_rest)

print("===>Testing using weights of epoch: ",val_epoch)
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

val_dataset = get_validation_data(opt['PATH']['VAL_DATASET'], {'patch_size':opt['VAL']['VAL_PS']})
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False, pin_memory=True)

psnr_val_rgb = []
ssim_val_rgb = []

for ii, data_val in enumerate(val_loader, 1):
    target_img = img_as_ubyte(data_val[0].numpy().squeeze().transpose((1,2,0)))
    input_ = data_val[1].cuda()

    factor = 64
    h,w = input_.shape[2],input_.shape[3]
    H,W = ((h+factor)//factor)*factor,((w+factor)//factor)*factor
    padh = H-h if h%factor!=0 else 0
    padw = W-w if w%factor!=0 else 0
    input_ = F.pad(input_,(0,padw,0,padh),'reflect')

    with torch.no_grad():
        restored = model_restoration(input_)

    restored = restored[0]
    restored = torch.clamp(restored,0,1)
    restored = restored[:,:h,:w]
    restored_img = img_as_ubyte(restored.cpu().numpy().squeeze().transpose((1,2,0)))

    ######### release testing gpu memory ##########
    del restored
    del input_
    torch.cuda.empty_cache()

    psnr_val_rgb.append(PSNR(restored_img, target_img))
    ssim_val_rgb.append(calculate_ssim(restored_img, target_img))
    # print(calculate_ssim(restored_img, target_img))
    log('%-6s \t %f \t %f' % (data_val[2][0], psnr_val_rgb[-1], ssim_val_rgb[-1]), os.path.join(log_dir, 'val_' + str(val_epoch)+'.txt'), P=True)
    if opt['VAL']['SAVE_IMG']:
        save_img_path = os.path.join(result_dir, 'epoch_' + str(val_epoch))
        file_name = data_val[2][0] + '_restored.png'
        utils.mkdir(save_img_path)
        utils.save_img(os.path.join(save_img_path, file_name), restored_img)

avg_psnr  = sum(psnr_val_rgb)/ii
avg_ssim  = sum(ssim_val_rgb)/ii
print(avg_ssim)
log('total images = %d \t avg_psnr = %f \t avg_ssim = %f' % (ii,avg_psnr,avg_ssim), os.path.join(log_dir, 'val_' + str(val_epoch)+'.txt'), P=True)