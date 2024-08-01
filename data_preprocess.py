import numpy as np
import torch
import warnings

from monai.transforms import (
    Compose,
    LoadImaged,
    ScaleIntensityRanged,
    EnsureTyped,
    ThresholdIntensityd,
)

from monai.apps.reconstruction.transforms.dictionary import (
    ExtractDataKeyFromMetaKeyd,
    ReferenceBasedSpatialCropd,
    ReferenceBasedNormalizeIntensityd,
)
from common.dictionary import (
    RadialKspaceMaskd, 
    SpiralKspaceMaskd, 
    RandomKspaceMaskd,
    EquispacedKspaceMaskd,
)

from common.blur_func import MotionModel

from monai.apps.reconstruction.fastmri_reader import FastMRIReader

from pathlib import Path
import argparse
from monai.data import Dataset, DataLoader

import logging
import os
import sys
from datetime import datetime
import time
import random
from tqdm import tqdm
import h5py
import matplotlib.pyplot as plt

#### temporal
from monai.apps.reconstruction.complex_utils import complex_abs, convert_to_tensor_complex
from monai.apps.reconstruction.mri_utils import root_sum_of_squares
from monai.config.type_definitions import NdarrayOrTensor
from monai.data.fft_utils import ifftn_centered
from monai.utils.type_conversion import convert_to_tensor

warnings.filterwarnings("ignore")
    
def data_transform(args, mask: bool = True, kspace_masked_ifft: bool = True, reconstruction_rss: bool = True):
    # create data loaders
    filename = args.datalist

    train_files = list(Path(args.data_path, 'multicoil_train').iterdir())
    train_files = [dict([("kspace", train_files[i])]) for i in range(len(train_files))]
    preprocess_files = [dict([("kspace", train_files[i]), ("name", str(train_files[i]).split('/')[-1])]) for i in range(len(train_files))]

    # with open(filename) as f:
    #     lines = f.readlines()

    # preprocess_files = []
    # for line in lines:
    #     preprocess_files.append(Path(args.data_path, line.strip()))
    # # preprocess_files = list(Path(args.data_path, 'multicoil_train').iterdir()) + list(Path(args.data_path, 'multicoil_val').iterdir())
    # print(len(preprocess_files))
    # preprocess_files = [dict([("kspace", preprocess_files[i]), ("name", str(preprocess_files[i]).split('/')[-1])]) for i in range(len(preprocess_files))]

    if args.accelerations == 4:
        args.center_fractions = 0.08
    elif args.accelerations == 8:
        args.center_fractions = 0.04
    else:
        raise ValueError(f"The acceleration {args.accelerations} is not defined")

    if args.mask_type == "cartesian_random":
        MaskTransform = RandomKspaceMaskd(
            keys=["kspace"],
            center_fractions=[args.center_fractions],
            accelerations=[args.accelerations],
            seed=args.seed,
            spatial_dims=2,
            is_complex=True,
        )
    elif args.mask_type == "cartesian_equispaced":
        MaskTransform = EquispacedKspaceMaskd(
            keys=["kspace"],
            center_fractions=[args.center_fractions],
            accelerations=[args.accelerations],
            seed=args.seed,
            spatial_dims=2,
            is_complex=True,
        )
    elif args.mask_type == 'radial':
        MaskTransform = RadialKspaceMaskd(
            keys=["kspace"],
            accelerations=[args.accelerations],
            seed=args.seed,
            spatial_dims=2,
            is_complex=True,
        )
    elif args.mask_type == 'spiral':
        MaskTransform = SpiralKspaceMaskd(
            keys=["kspace"],
            accelerations=[args.accelerations],
            seed=args.seed,
            spatial_dims=2,
            is_complex=True,
        )

    preprocess_transforms = Compose(
        [
            LoadImaged(keys=["kspace"], reader=FastMRIReader, dtype=np.complex64),
            ExtractDataKeyFromMetaKeyd(keys=["reconstruction_rss", "mask"], meta_key="kspace_meta_dict"),
            # MaskTransform,
            # ReferenceBasedSpatialCropd(keys=["kspace_masked_ifft"], ref_key="reconstruction_rss"),
        ]
    )
    preprocess_ds = Dataset(data=preprocess_files, transform=preprocess_transforms)
    preprocess_loader = DataLoader(preprocess_ds, batch_size=1, shuffle=False, num_workers=args.num_workers)

    if args.noise_level == 0:
        noise_level = 'none'
    
    # foldername = Path(args.data_path, '_'.join([args.mask_type, str(args.accelerations), noise_level, args.blurriness_type]))
    # if not os.path.exists(foldername):
    #     os.makedirs(foldername)
    #     print(f"Folder {foldername} has been created")
    
    # if mask:
    #     mask_path = Path(foldername, 'mask')
    #     if not os.path.exists(mask_path):
    #         os.makedirs(mask_path)
    #         print(f"Folder {mask_path} has been created")
    
    # if kspace_masked_ifft:
    #     kspace_masked_ifft_path = Path(foldername, 'kspace_masked_ifft')
    #     if not os.path.exists(kspace_masked_ifft_path):
    #         os.makedirs(kspace_masked_ifft_path)
    #         print(f"Folder {kspace_masked_ifft_path} has been created")
    
    # if reconstruction_rss:
    #     reconstruction_rss_path = Path(foldername, 'reconstruction_rss')
    #     if not os.path.exists(reconstruction_rss_path):
    #         os.makedirs(reconstruction_rss_path)
    #         print(f"Folder {reconstruction_rss_path} has been created")

    for index, batch_data in tqdm(enumerate(preprocess_loader), total=len(preprocess_loader)):
        name = batch_data['name'][0]
        
        # if mask:
        #     mask_data = batch_data['mask'][0]
        #     # print(mask_data.shape, mask_data.dtype,mask_data.numpy().dtype) # torch.Size([1, 1, 640, 320, 1]), torch.bool bool
        #     file_name = Path(mask_path, name)
        #     with h5py.File(file_name, 'w') as f:
        #         f.create_dataset('data', data=mask_data.numpy())
        #     plt.imshow(mask_data.numpy()[0,0,:,:,0], cmap='gray')
        #     plt.axis('off')
        #     plt.savefig(str(file_name).replace('.h5', '.png'), bbox_inches='tight')

        # if kspace_masked_ifft:
        #     kspace_masked_ifft_data = batch_data['kspace_masked_ifft'][0]
        #     # print(kspace_masked_ifft_data.shape, kspace_masked_ifft_data.dtype) #torch.Size([16, 320, 320])
        #     file_name = Path(kspace_masked_ifft_path, name)
        #     with h5py.File(file_name, 'w') as f:
        #         f.create_dataset('data', data=kspace_masked_ifft_data.numpy())

        # if reconstruction_rss:
        #     reconstruction_rss_data = batch_data['reconstruction_rss'][0]
        #     # print(reconstruction_rss_data.shape, reconstruction_rss_data.dtype) #torch.Size([16, 320, 320])
        #     file_name = Path(reconstruction_rss_path, name)
        #     with h5py.File(file_name, 'w') as f:
        #         f.create_dataset('data', data=reconstruction_rss_data.numpy())
        
        kspace = batch_data['kspace'][0]
        print('information for kspace', kspace.shape, kspace.mean(), kspace.var())
        motion_dg = MotionModel([0,1])
        kspace = kspace.permute(0, 2, 3, 1)
        degraded_kspace = motion_dg(kspace, args.blurriness_type).permute(0, 3, 1, 2).unsqueeze(0)
        degraded_kspace = convert_to_tensor_complex(degraded_kspace)
        degraded_kspace_ifft = convert_to_tensor(
            complex_abs(ifftn_centered(degraded_kspace, spatial_dims=2, is_complex=True))
        )
        degraded_kspace_ifft_rss = convert_to_tensor(
            root_sum_of_squares(degraded_kspace_ifft, spatial_dim=-3)
        )
        reconstruction_rss_data = batch_data['reconstruction_rss']
        x = reconstruction_rss_data
        y = degraded_kspace_ifft_rss

        print(kspace.shape, x.shape, y.shape)

        import matplotlib.pyplot as plt

        plt.imshow(x[0, x.shape[1]//2, :, :], cmap='gray')

        plt.axis('off')

        plt.savefig('gray_image.png', bbox_inches='tight', pad_inches=0)
        plt.close()


        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))


        axes[0].imshow(x[0, x.shape[1]//2, :, :], cmap='gray')
        axes[0].set_title('Before Reconstruction')
        axes[0].axis('off')

        axes[1].imshow(y[0, y.shape[1]//2, :, :], cmap='gray')
        axes[1].set_title('After Reconstruction')
        axes[1].axis('off')

        kspace_npy = kspace.view(torch.float64).numpy()
        kspace_npy = np.log10(np.abs(kspace_npy) + 1)
        axes[2].imshow(kspace_npy[x.shape[1]//2, :, :, 0], cmap='gray')
        axes[2].set_title('k-space')
        axes[2].axis('off')

        plt.tight_layout()
        plt.savefig(f'cache/blur_{index}.png')


        # with h5py.File('example.h5', 'w') as f:
        #     dset = f.create_dataset('data', data=arr.numpy())
            
        # with h5py.File('example.h5', 'r') as f:
        #     arr_h5 = torch.from_numpy(f['data'][()])
            
        # print(torch.allclose(arr, arr_h5))


    

def __main__():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num_workers",
        default=1,
        type=int,
        help="Number of workers to use in data loader",
    )

    parser.add_argument(
        "--data_path",
        default='/NAS_liujie/liujie/fastMRI/brain',
        type=Path,
        help="Path to the fastMRI dataset",
    )
    ## '/home/jliu288/data/fastMRI/brain/multicoil_train'
    ## '/hdd2/lj/fastMRI/brain/multicoil_train'

    parser.add_argument(
        "--datalist",
        default='./data/fastMRI_brain/train_0.txt',
        type=Path,
        help="Path to the fastMRI dataset",
    )

    # Preprocess parameters
    ## Undersampling
    parser.add_argument(
        "--mask_type", default="cartesian_equispaced", type=str, help="under-sampling mask type: ['cartesian_random','cartesian_equispaced', 'radial', 'spiral']"
    )

    parser.add_argument("--accelerations", default=8, type=int, help="acceleration factors used during training")

    parser.add_argument("--seed", default=123, type=int, help="seed for random")
    ## Noise
    parser.add_argument("--noise_level", default=0, type=int, help="noise_level")
    ## Blurriness
    parser.add_argument("--blurriness_type", default="respiration", type=str, help="blurriness type: ['none', 'rigid', 'respiration']")

    args = parser.parse_args()
    data_transform(args, mask=False, kspace_masked_ifft=False, reconstruction_rss=False)


__main__()
