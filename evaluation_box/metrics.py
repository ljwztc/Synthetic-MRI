import os
import cv2
import sys
import math
import torch
import torch.nn.functional as F
import numpy as np
from piq import multi_scale_ssim, fsim, gmsd, vsi, haarpsi, dss, brisque
from piq import ssim as ssim2


from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

from evaluation_box.niqe import niqe
from evaluation_box.piqe import piqe



def PSNR(gt, rec):
    """
    Peak Signal to Noise Ratio
    inputs:
        gt: [slides, w, h]
        rec: reconstruction with the same dimensions as gt
    outputs:
        skimage PSNR score between gt and rec
    """
    return compare_psnr(gt, rec, data_range=gt.max())
    # mse = np.mean((gt[0] - rec[0]) ** 2)
    # return 20 * np.log10(255.0 / np.sqrt(mse))



def SSIM(gt, rec):
    """
    Compute SSIM quality assessment metric
    inputs:
        gt: [slides, w, h]
        rec: reconstruction with the same dimensions as gt
    outputs:
        skimage SSIM score between gt and rec
    """
    gt2 = torch.tensor(gt).unsqueeze(0)
    rec2 = torch.tensor(rec).unsqueeze(0)
    return compare_ssim(gt.transpose(1, 2, 0), rec.transpose(1, 2, 0), channel_axis=2, data_range=gt.max())


def HFEN(gt, rec):
    """
    Calculates high frequency error norm (HFEN) to quantify the quality 
    of reconstruction of edges and fine features. Uses a rotationally 
    symmetric LoG (Laplacian of Gaussian) filter to capture edges.
    inputs:
        gt: [slides, w, h]
        rec: reconstruction with the same dimensions as gt
    outputs:
        hfen score between gt and rec
    """
    rec_max, rec_min = np.max(rec), np.min(rec)
    gt = (gt - rec_min) / rec_max * 255
    rec = (rec - rec_min) / rec_max * 255
    kernel_size = 5
    sigma = 1.0
    log_kernel = cv2.getGaussianKernel(kernel_size, sigma)
    log_kernel = np.outer(log_kernel, log_kernel)
    log_kernel = -1 * ((log_kernel - np.mean(log_kernel)) / np.max(log_kernel))

    gt_output = np.zeros_like(gt)

    for channel in range(gt_output.shape[0]):
        gt_output[channel, :, :] = cv2.filter2D(gt[channel, :, :], -1, log_kernel)
    
    rec_output = np.zeros_like(rec)

    for channel in range(rec_output.shape[0]):
        rec_output[channel, :, :] = cv2.filter2D(rec[channel, :, :], -1, log_kernel)


    hfen = np.mean((gt_output - rec_output) ** 2)

    return hfen


def MS_SSIM(gt, rec):
    """
    Compute multi-scale SSIM quality assessment metric
    inputs:
        gt: [slides, w, h]
        rec: reconstruction with the same dimensions as gt
    outputs:
        piq ms_ssim score between gt and rec
    """
    gt = torch.tensor(gt).unsqueeze(0)
    rec = torch.tensor(rec).unsqueeze(0)
    return multi_scale_ssim(gt, rec, data_range=gt.max(), reduction='mean').numpy()

def FSIM(gt, rec):
    gt = torch.tensor(gt).unsqueeze(1)
    rec = torch.tensor(rec).unsqueeze(1)
    return fsim(gt, rec, data_range=gt.max(), reduction='mean', chromatic=False).numpy()


def GMSD(gt, rec):
    gt = torch.tensor(gt).unsqueeze(1)
    rec = torch.tensor(rec).unsqueeze(1)
    return gmsd(gt, rec, data_range=gt.max(), reduction='mean').numpy()

def HaarPSI(gt, rec):
    gt = torch.tensor(gt).unsqueeze(0)
    rec = torch.tensor(rec).unsqueeze(0)
    return haarpsi(gt, rec, data_range=gt.max()).numpy()

def DSS(gt, rec):
    gt = torch.tensor(gt).unsqueeze(1)
    rec = torch.tensor(rec).unsqueeze(1)
    return dss(gt, rec, data_range=gt.max()).numpy()

def M_BRISQUE(rec):
    rec = torch.tensor(rec).unsqueeze(1)
    return brisque(rec, data_range=rec.max()).numpy()

def NIQE(rec):
    rec_max, rec_min = np.max(rec), np.min(rec)
    rec = (rec - rec_min) / rec_max * 255
    niqe_array = []
    for i in range(rec.shape[0]):
        img = rec[i]
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        niqe_array.append(niqe(img))
    return np.mean(niqe_array)

def PIQE(rec):
    rec_max, rec_min = np.max(rec), np.min(rec)
    rec = (rec - rec_min) / rec_max * 255
    piqe_array = []
    for i in range(rec.shape[0]):
        img = rec[i]
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        piqe_array.append(piqe(img))
    return np.mean(piqe_array)