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

import logging
import os
import sys
from datetime import datetime
import time
from collections import defaultdict
import random

import random
from tqdm import tqdm
import h5py
import matplotlib.pyplot as plt

from common.blur_func import MotionModel
from common.noise_func import NoiseModel

#### temporal
from monai.apps.reconstruction.complex_utils import complex_abs, convert_to_tensor_complex
from monai.apps.reconstruction.mri_utils import root_sum_of_squares
from monai.config.type_definitions import NdarrayOrTensor
from monai.data.fft_utils import ifftn_centered, fftn_centered
from monai.utils.type_conversion import convert_to_tensor

warnings.filterwarnings("ignore")


def trainer(args):
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    now = datetime.now()
    date = now.strftime("%m-%d-%y_%H-%M")

    # create training-validation data loaders
    train_files = list(Path(args.data_path_train).iterdir())
    train_files = [dict([("kspace", train_files[i]), ('name', train_files[i].name)]) for i in range(len(train_files))]

    val_files = list(Path(args.data_path_val).iterdir())
    val_files = [dict([("kspace", val_files[i]), ('name', val_files[i].name)]) for i in range(len(val_files))]

    train_transforms = Compose(
        [
            LoadImaged(keys=["kspace"], reader=FastMRIReader, dtype=np.complex64),
            # user can also add other random transforms
            ExtractDataKeyFromMetaKeyd(keys=["reconstruction_rss", "mask"], meta_key="kspace_meta_dict"),
            # MaskTransform,
        ]
    )

    train_ds = CacheDataset(
        data=train_files, transform=train_transforms, cache_rate=args.cache_rate, num_workers=args.num_workers
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    val_ds = CacheDataset(
        data=val_files, transform=train_transforms, cache_rate=args.cache_rate, num_workers=args.num_workers
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)


    for index, batch_data in tqdm(enumerate(train_loader), total=len(train_loader)):
        kspace = batch_data['kspace'][0]
        # print('information for kspace', kspace.shape, kspace.dtype, kspace.mean(), kspace.var())
        # print('information for numpykpace', kspace.numpy().dtype)
        
        motion_dg = MotionModel([0.4, 0.6])
        kspace_noise_blur = motion_dg(kspace.permute(0, 2, 3, 1), 'respiration').permute(0, 3, 1, 2)
        
        noise_dg = NoiseModel(10)
        kspace_cpx = convert_to_tensor_complex(kspace_noise_blur)

        # print('information for kspace_raw', kspace_cpx.shape, kspace_cpx.dtype, kspace_cpx.mean(), kspace_cpx.var())
        image = ifftn_centered(kspace_cpx, spatial_dims=2, is_complex=True)
        degraded_image = noise_dg(image)
        kspace_noise = convert_to_tensor(
            fftn_centered(degraded_image, spatial_dims=2, is_complex=True)
        )
        # print('information for kspace_noise', kspace_noise.shape, kspace_noise.dtype, kspace_noise.mean(), kspace_noise.var())
        kspace_blur_noise = torch.view_as_complex(kspace_noise)
        # print(saved_kspace_noise.dtype, saved_kspace_noise.shape)

        
        saved_kspace_blur_noise = kspace_blur_noise.numpy()

        name = batch_data['name'][0]
        original_file_name = args.data_path_train / name
        with h5py.File(original_file_name, 'r') as original_file:
            ismrmrd_header_data = original_file['ismrmrd_header'][()]
            reconstruction_rss_data = original_file['reconstruction_rss'][()]

            new_file_dir = Path('/NAS_liujie/liujie/fastMRI/brain/multicoil_train_nab')
            if not new_file_dir.exists():
                new_file_dir.mkdir(parents=True, exist_ok=True)

            new_file_name = new_file_dir / name
            
            with h5py.File(new_file_name, 'w') as new_file:
                original_attrs = dict(original_file.attrs)
                for attr_name, attr_value in original_attrs.items():
                    new_file.attrs[attr_name] = attr_value
                new_file.create_dataset('ismrmrd_header', data=ismrmrd_header_data)
                new_file.create_dataset('reconstruction_rss', data=reconstruction_rss_data)
                new_file.create_dataset('kspace', data=saved_kspace_blur_noise)
                print(f"File '{new_file_name}' has been created with the new 'kspace' dataset.")
        


        # degraded_kspace = convert_to_tensor_complex(kspace_noise)
        # degraded_kspace_ifft = convert_to_tensor(
        #     complex_abs(ifftn_centered(degraded_kspace, spatial_dims=2, is_complex=True))
        # )
        # degraded_kspace_ifft_rss = convert_to_tensor(
        #     root_sum_of_squares(degraded_kspace_ifft, spatial_dim=-3)
        # )
        # reconstruction_rss_data = batch_data['reconstruction_rss']
        # x = reconstruction_rss_data
        # y = degraded_kspace_ifft_rss

        # # print(kspace.shape, x.shape, y.shape)

        # import matplotlib.pyplot as plt
        # fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))


        # axes[0].imshow(x[0, x.shape[1]//2, :, :], cmap='gray')
        # axes[0].set_title('Before Reconstruction')
        # axes[0].axis('off')

        # axes[1].imshow(y[0, y.shape[1]//2, :, :], cmap='gray')
        # axes[1].set_title('After Reconstruction')
        # axes[1].axis('off')

        # kspace_npy = kspace.view(torch.float64).numpy()
        # kspace_npy = np.log10(np.abs(kspace_npy) + 1)
        # axes[2].imshow(kspace_npy[x.shape[1]//2, :, :, 0], cmap='gray')
        # axes[2].set_title('k-space')
        # axes[2].axis('off')

        # plt.tight_layout()
        # plt.savefig(f'cache/blur_{index}.png')

    for index, batch_data in tqdm(enumerate(val_loader), total=len(val_loader)):
        
        kspace = batch_data['kspace'][0]
        # print('information for kspace', kspace.shape, kspace.dtype, kspace.mean(), kspace.var())
        # print('information for numpykpace', kspace.numpy().dtype)
        
        motion_dg = MotionModel([0.4, 0.6])
        kspace_noise_blur = motion_dg(kspace.permute(0, 2, 3, 1), 'respiration').permute(0, 3, 1, 2)
        
        noise_dg = NoiseModel(10)
        kspace_cpx = convert_to_tensor_complex(kspace_noise_blur)

        # print('information for kspace_raw', kspace_cpx.shape, kspace_cpx.dtype, kspace_cpx.mean(), kspace_cpx.var())
        image = ifftn_centered(kspace_cpx, spatial_dims=2, is_complex=True)
        degraded_image = noise_dg(image)
        kspace_noise = convert_to_tensor(
            fftn_centered(degraded_image, spatial_dims=2, is_complex=True)
        )
        # print('information for kspace_noise', kspace_noise.shape, kspace_noise.dtype, kspace_noise.mean(), kspace_noise.var())
        kspace_blur_noise = torch.view_as_complex(kspace_noise)
        # print(saved_kspace_noise.dtype, saved_kspace_noise.shape)

        
        saved_kspace_blur_noise = kspace_blur_noise.numpy()

        name = batch_data['name'][0]
        original_file_name = args.data_path_val / name
        with h5py.File(original_file_name, 'r') as original_file:
            ismrmrd_header_data = original_file['ismrmrd_header'][()]
            reconstruction_rss_data = original_file['reconstruction_rss'][()]

            new_file_dir = Path('/NAS_liujie/liujie/fastMRI/brain/multicoil_val_nab')
            if not new_file_dir.exists():
                new_file_dir.mkdir(parents=True, exist_ok=True)

            new_file_name = new_file_dir / name
            
            with h5py.File(new_file_name, 'w') as new_file:
                original_attrs = dict(original_file.attrs)
                for attr_name, attr_value in original_attrs.items():
                    new_file.attrs[attr_name] = attr_value
                new_file.create_dataset('ismrmrd_header', data=ismrmrd_header_data)
                new_file.create_dataset('reconstruction_rss', data=reconstruction_rss_data)
                new_file.create_dataset('kspace', data=saved_kspace_blur_noise)
                print(f"File '{new_file_name}' has been created with the new 'kspace' dataset.")

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

    parser.add_argument(
        "--exp",
        default=None,
        type=str,
        help="experiment name (a folder will be created with this name to store the results)",
    )

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