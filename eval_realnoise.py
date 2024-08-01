import numpy as np
import torch
import warnings

from monai.transforms import (
    Compose,
    LoadImaged,
    ScaleIntensityRanged,
    EnsureTyped,
    ThresholdIntensityd,
    EnsureChannelFirstd,
    Orientationd,
)

from monai.apps.reconstruction.transforms.dictionary import (
    ExtractDataKeyFromMetaKeyd,
    RandomKspaceMaskd,
    EquispacedKspaceMaskd,
    ReferenceBasedSpatialCropd,
    ReferenceBasedNormalizeIntensityd,
)

from evaluation_box.metrics import SSIM, PSNR, HFEN, MS_SSIM, FSIM, GMSD, HaarPSI, DSS, M_BRISQUE, NIQE, PIQE

from monai.networks.nets import BasicUNet

from pathlib import Path
import argparse
from monai.data import CacheDataset, DataLoader, decollate_batch, Dataset
from torch.utils.tensorboard import SummaryWriter

import logging
import os
import sys
from datetime import datetime
import time
from collections import defaultdict
import random
import glob

warnings.filterwarnings("ignore")

dataset_split = {
    0: [1,2,3,4],
    1: [5,6,7,8],
    2: [9,10,11,12],
    3: [13,14,15,16,17],
    4: [18,19,20,21,22]
}

def trainer(args):
    # create the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BasicUNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        features=args.features,
    ).to(device)
    print("#model_params:", np.sum([len(p.flatten()) for p in model.parameters()]))
    print("resume training from checkpoint ", args.checkpoint_dir)

    for fold in range(5):
        load_path = args.checkpoint_dir + str(fold)
        load_path = glob.glob(load_path+'/**')[0]
        load_path = load_path + '/unet_mri_reconstruction.pt'
        model.load_state_dict(torch.load(load_path))

        val_files = [args.data_path_train + '/sub-{:02d}.nii'.format(i) for i in dataset_split[fold]]
        val_files = [dict([("input", val_files[i]), 
                        ("gt", val_files[i].replace('image', 'gt')),
                        ("name", val_files[i].split('/')[-1])]) for i in range(len(val_files))]
        

        train_transforms = Compose(
            [
                LoadImaged(keys=["input", "gt"]),
                # user can also add other random transforms
                # EnsureChannelFirstd(keys=["input", "gt"]),
                Orientationd(keys=["input", "gt"], axcodes="RAS"),
                ReferenceBasedNormalizeIntensityd(
                    keys=["input", "gt"], ref_key="input"
                ),
                ThresholdIntensityd(
                    keys=["input", "gt"], threshold=6.0, above=False, cval=6.0
                ),
                ThresholdIntensityd(
                    keys=["input", "gt"], threshold=-6.0, above=True, cval=-6.0
                ),
                EnsureTyped(keys=["input", "gt"]),
            ]
        )

        val_ds = Dataset(data=val_files, transform=train_transforms)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

        # validation
        model.eval()
        outputs = defaultdict(list)
        targets = defaultdict(list)
        with torch.no_grad():
            val_ssim = list()
            val_msssim = list()
            val_psnr = list()
            val_hfen = list()
            val_fsim = list()
            val_gmsd = list()
            val_dss = list()
            val_hpsi = list()
            val_brisque = list()
            val_niqe = list()

            for val_data in val_loader:
                input, target, fname, mean, std = (
                    val_data["input"],
                    val_data["gt"],
                    val_data["name"],
                    val_data["mean"],
                    val_data["std"],
                )

                # iterate through all slices:
                slice_dim = 3  # change this if another dimension is your slice dimension
                num_slices = input.shape[slice_dim]
                for i in range(num_slices):
                    inp = input[:, :, :, i].unsqueeze(0)
                    tar = target[:, :, :, i].unsqueeze(0)
                    output = model(inp.to(device))

                    _std = std.item()
                    _mean = mean.item()
                    output_scale = output.data.cpu().numpy()[0][0] * _std + _mean
                    outputs[fname[0]].append(output_scale)
                    target_scale = tar.numpy()[0][0] * _std + _mean
                    
                    targets[fname[0]].append(target_scale)

            # compute validation ssims

            for fname in outputs:
                outputs[fname] = np.stack(outputs[fname])
                targets[fname] = np.stack(targets[fname])
                val_ssim.append(SSIM(targets[fname], outputs[fname]))
                val_psnr.append(PSNR(targets[fname], outputs[fname]))
                
            
            for fname in outputs:
                output_scale = np.clip(outputs[fname], 0, 256)
                target_scale = np.clip(targets[fname], 0, 256)
                val_msssim.append(MS_SSIM(target_scale, output_scale))
                # val_hfen.append(HFEN(target_scale, output_scale))
                val_fsim.append(FSIM(target_scale, output_scale))
                val_gmsd.append(GMSD(target_scale, output_scale))
                val_hpsi.append(HaarPSI(target_scale, output_scale))
                val_dss.append(DSS(target_scale, output_scale))
                try:
                    val_brisque.append(M_BRISQUE(output_scale))
                except:
                    val_brisque.append(0)
                    
                # val_niqe.append(NIQE(output_scale))

            ssim_metric = np.mean(val_ssim)
            psnr_metric = np.mean(val_psnr)
            msssim_metrics = np.mean(val_msssim)
            # hfen_metric = np.mean(val_hfen)
            fsim_metric = np.mean(val_fsim)
            gmsd_metric = np.mean(val_gmsd)
            dss_metric = np.mean(val_dss)
            hpsi_metric = np.mean(val_hpsi)

            brisque_metric = np.mean(val_brisque)
            # niqe_metric = np.mean(val_niqe)

            print('ssim:', ssim_metric, 'psnr:', psnr_metric, 'ms_ssim', msssim_metrics , 'fsim:', fsim_metric, 'gmsd:', gmsd_metric, 'hpsi:', hpsi_metric, 'dss:', dss_metric, 'brisque:', brisque_metric)
            # print('ssim:', ssim_metric, 'psnr:', psnr_metric)




def main():
    parser = argparse.ArgumentParser()

    # data loader arguments
    parser.add_argument(
        "--batch_size",
        default=1,
        type=int,
        help="Data loader batch size (batch_size>1 is suitable for varying input size",
    )

    parser.add_argument(
        "--num_workers",
        default=4,
        type=int,
        help="Number of workers to use in data loader",
    )

    parser.add_argument(
        "--cache_rate",
        default=0.0,
        type=float,
        help="The fraction of the data to be cached when being loaded",
    )

    parser.add_argument(
        "--data_path_train",
        default='/NAS_liujie/liujie/Real_Noise/image',
        type=str,
        help="Path to the fastMRI training set",
    )

    parser.add_argument(
        "--sample_rate",
        default=0.8,
        type=float,
        help="what fraction of the dataset to use for training (also, what fraction of validation set to use)",
    )

    # Mask parameters
    parser.add_argument("--accelerations", default=[4], type=list, help="acceleration factors used during training")

    parser.add_argument(
        "--center_fractions",
        default=[0.08],
        type=list,
        help="center fractions used during training (center fraction denotes the center region to exclude from masking)",
    )

    # training params
    parser.add_argument("--num_epochs", default=1000, type=int, help="number of training epochs")

    parser.add_argument("--split", default=0, type=int, help="from 0~4")

    parser.add_argument("--exp_dir", default='./log/', type=Path, help="output directory to save training logs")

    parser.add_argument("--lr", default=0.0001, type=float, help="learning rate")

    parser.add_argument("--lr_step_size", default=500, type=int, help="decay learning rate every lr_step_size epochs")

    parser.add_argument(
        "--lr_gamma",
        default=0.1,
        type=float,
        help="every lr_step_size epochs, decay learning rate by a factor of lr_gamma",
    )

    parser.add_argument("--weight_decay", default=0.0, type=float, help="ridge regularization factor")

    parser.add_argument(
        "--mask_type", default="random", type=str, help="under-sampling mask type: ['random','equispaced']"
    )

    # model specific args
    parser.add_argument("--drop_prob", default=0.0, type=float, help="dropout probability for U-Net")

    parser.add_argument(
        "--features",
        default=[32, 64, 128, 256, 512, 32],
        type=list,
        help="six integers as numbers of features (see monai.networks.nets.basic_unet)",
    )

    parser.add_argument(
        "--resume_checkpoint", default=False, type=bool, help="if True, training statrts from a model checkpoint"
    )

    parser.add_argument(
        "--checkpoint_dir", default=None, type=str, help="model checkpoint path to resume training from"
    )

    args = parser.parse_args()
    trainer(args)

if __name__ == "__main__":
    main()

# CUDA_VISIBLE_DEVICES=1 python train.py --exp real_noise