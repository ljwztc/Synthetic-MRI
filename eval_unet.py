# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
import warnings
from fastmri_ssim import skimage_ssim

from monai.transforms import (
    Compose,
    LoadImaged,
    ScaleIntensityRanged,
    EnsureTyped,
    ThresholdIntensityd,
)

from monai.apps.reconstruction.transforms.dictionary import (
    ExtractDataKeyFromMetaKeyd,
    RandomKspaceMaskd,
    EquispacedKspaceMaskd,
    ReferenceBasedSpatialCropd,
    ReferenceBasedNormalizeIntensityd,
)

from monai.apps.reconstruction.fastmri_reader import FastMRIReader
from monai.networks.nets import BasicUNet

from pathlib import Path
import argparse
from monai.data import CacheDataset, DataLoader, decollate_batch
from torch.utils.tensorboard import SummaryWriter

from evaluation_box.metrics import SSIM, PSNR, HFEN, MS_SSIM, FSIM, GMSD, HaarPSI, DSS, M_BRISQUE, NIQE, PIQE

import logging
import os
import sys
from datetime import datetime
import time
from collections import defaultdict
import random

warnings.filterwarnings("ignore")


def trainer(args):
    val_files = list(Path(args.data_path_val).iterdir())
    ##################################################### temp: dummy file
    val_files = list(filter(lambda f: "file_brain_AXT2_201_2010294.h5" not in str(f), val_files))
    ##################################################### temp: dummy file
    random.shuffle(val_files)
    val_files = val_files[
        : int(args.sample_rate * len(val_files))
    ]  # select a subset of the data according to sample_rate
    val_files = [dict([("kspace", val_files[i])]) for i in range(len(val_files))]

    # define mask transform type (e.g., whether it is equispaced or random)
    if args.mask_type == "random":
        MaskTransform = RandomKspaceMaskd(
            keys=["kspace"],
            center_fractions=args.center_fractions,
            accelerations=args.accelerations,
            spatial_dims=2,
            is_complex=True,
        )
    elif args.mask_type == "equispaced":
        MaskTransform = EquispacedKspaceMaskd(
            keys=["kspace"],
            center_fractions=args.center_fractions,
            accelerations=args.accelerations,
            spatial_dims=2,
            is_complex=True,
        )

    train_transforms = Compose(
        [
            LoadImaged(keys=["kspace"], reader=FastMRIReader, dtype=np.complex64),
            # user can also add other random transforms
            ExtractDataKeyFromMetaKeyd(keys=["reconstruction_rss", "mask"], meta_key="kspace_meta_dict"),
            MaskTransform,
            ReferenceBasedSpatialCropd(keys=["kspace_masked_ifft"], ref_key="reconstruction_rss"),
            ReferenceBasedNormalizeIntensityd(
                keys=["kspace_masked_ifft", "reconstruction_rss"], ref_key="kspace_masked_ifft", channel_wise=True
            ),
            ThresholdIntensityd(
                keys=["kspace_masked_ifft", "reconstruction_rss"], threshold=6.0, above=False, cval=6.0
            ),
            ThresholdIntensityd(
                keys=["kspace_masked_ifft", "reconstruction_rss"], threshold=-6.0, above=True, cval=-6.0
            ),
            EnsureTyped(keys=["kspace", "kspace_masked_ifft", "reconstruction_rss"]),
        ]
    )

    val_ds = CacheDataset(
        data=val_files, transform=train_transforms, cache_rate=args.cache_rate, num_workers=args.num_workers
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # create the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BasicUNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        features=args.features,
    ).to(device)
    print("#model_params:", np.sum([len(p.flatten()) for p in model.parameters()]))

    load_path = args.checkpoint_dir / 'unet_mri_reconstruction.pt'
    model.load_state_dict(torch.load(load_path))
    print("resume training from a given checkpoint...")


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
            input, target, mean, std, fname = (
                val_data["kspace_masked_ifft"],
                val_data["reconstruction_rss"],
                val_data["mean"],
                val_data["std"],
                val_data["kspace_meta_dict"]["filename"],
            )

            # iterate through all slices:
            slice_dim = 1  # change this if another dimension is your slice dimension
            num_slices = input.shape[slice_dim]
            for i in range(num_slices):
                inp = input[:, i, ...].unsqueeze(slice_dim)
                tar = target[:, i, ...].unsqueeze(slice_dim)
                output = model(inp.to(device))

                _std = std[0][i].item()
                _mean = mean[0][i].item()
                outputs[fname[0]].append(output.data.cpu().numpy()[0][0] * _std + _mean)
                targets[fname[0]].append(tar.numpy()[0][0] * _std + _mean)

        for fname in outputs:
            outputs[fname] = np.stack(outputs[fname])
            targets[fname] = np.stack(targets[fname])
            val_ssim.append(SSIM(targets[fname], outputs[fname]))
            val_psnr.append(PSNR(targets[fname], outputs[fname]))
                
        for fname in outputs:
            output_scale = np.clip(outputs[fname], 0, 256)
            target_scale = np.clip(targets[fname], 0, 256)
            try:
                val_dss.append(DSS(target_scale, output_scale))
            except:
                print('dss error')
                val_dss.append(0)
            try:
                val_brisque.append(M_BRISQUE(output_scale))
            except:
                print('brisque error')
                val_brisque.append(0)
                
            # val_niqe.append(NIQE(output_scale))

        ssim_metric = np.mean(val_ssim)
        psnr_metric = np.mean(val_psnr)
        dss_metric = np.mean(val_dss)
        brisque_metric = np.mean(val_brisque)

        print('ssim:', ssim_metric, 'psnr:', psnr_metric, 'dss:', dss_metric, 'brisque:', brisque_metric)




def __main__():
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
        default='/NAS_liujie/liujie/fastMRI/brain/multicoil_train',
        type=Path,
        help="Path to the fastMRI training set",
    )

    parser.add_argument(
        "--data_path_val",
        default='/NAS_liujie/liujie/fastMRI/brain/multicoil_val',
        type=Path,
        help="Path to the fastMRI validation set",
    )

    parser.add_argument(
        "--sample_rate",
        default=1.0,
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
    parser.add_argument("--num_epochs", default=50, type=int, help="number of training epochs")

    parser.add_argument("--exp_dir", default='./log', type=Path, help="output directory to save training logs")

    parser.add_argument("--lr", default=0.0001, type=float, help="learning rate")

    parser.add_argument("--lr_step_size", default=40, type=int, help="decay learning rate every lr_step_size epochs")

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
        "--checkpoint_dir", default=None, type=Path, help="model checkpoint path to resume training from"
    )

    args = parser.parse_args()
    trainer(args)


__main__()