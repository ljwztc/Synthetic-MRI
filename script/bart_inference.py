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
import multiprocessing
from fastmri_ssim import skimage_ssim
import bart

from monai.transforms import (
    Compose,
    LoadImaged,
    ScaleIntensityRanged,
    EnsureTyped,
    ThresholdIntensityd,
    SpatialCrop,
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
import matplotlib.pyplot as plt

from typing import Dict, NamedTuple, Optional, Sequence, Tuple, Union

from pathlib import Path

import h5py

warnings.filterwarnings("ignore")

def tensor_to_complex_np(data: torch.Tensor) -> np.ndarray:
    """
    Converts a complex torch tensor to numpy array.

    Args:
        data: Input data to be converted to numpy.

    Returns:
        Complex numpy version of data.
    """
    return torch.view_as_complex(data).numpy()


def save_reconstructions(reconstructions: Dict[str, np.ndarray], out_dir: Path):
    """
    Save reconstruction images.

    This function writes to h5 files that are appropriate for submission to the
    leaderboard.

    Args:
        reconstructions: A dictionary mapping input filenames to corresponding
            reconstructions.
        out_dir: Path to the output directory where the reconstructions should
            be saved.
    """
    out_dir.mkdir(exist_ok=True, parents=True)
    for fname, recons in reconstructions.items():
        with h5py.File(out_dir / fname, "w") as hf:
            hf.create_dataset("reconstruction", data=recons)


def trainer(args):
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    outpath = os.path.join(args.exp_dir, args.exp)
    Path(outpath).mkdir(parents=True, exist_ok=True)  # create output directory to store model checkpoints
    now = datetime.now()
    date = now.strftime("%m-%d-%y_%H-%M")
    writer = SummaryWriter(
        outpath + "/" + date
    )  # create a date directory within the output directory for storing training logs

    # val_files = list(Path(args.data_path_val).iterdir())
    # ##################################################### temp: dummy file
    # val_files = list(filter(lambda f: "file_brain_AXT2_201_2010294.h5" not in str(f), val_files))
    # ##################################################### temp: dummy file
    # random.shuffle(val_files)
    # # val_files = val_files[
    # #     : 2
    # # ]  # select a subset of the data according to sample_rate
    # val_files = val_files[
    #     : int(args.sample_rate * len(val_files))
    # ]  # select a subset of the data according to sample_rate

    val_files = [args.data_path_val / 'file_brain_AXT2_201_2010106.h5']
    print("val_file length", len(val_files))
    val_files = [dict([("kspace", val_files[i]), ("name", val_files[i].name)]) for i in range(len(val_files))]

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
            # user can also add other random transforms but remember to disable randomness for val_transforms
            ExtractDataKeyFromMetaKeyd(keys=["reconstruction_rss", "mask"], meta_key="kspace_meta_dict"),
            MaskTransform,
            EnsureTyped(keys=["kspace", "kspace_masked_ifft", "reconstruction_rss"]),
        ]
    )

    val_ds = CacheDataset(
        data=val_files, transform=train_transforms, cache_rate=args.cache_rate, num_workers=args.num_workers
    )
    # val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    def cs_total_variation(kspace, reg_wt, croped_size):
        """
        Run ESPIRIT coil sensitivity estimation and Total Variation Minimization
        based reconstruction algorithm using the BART toolkit.

        Args:
            reg_wt (float): Regularization parameter.
            crop_size (tuple): Size to crop final image to.

        Returns:
            np.array: Reconstructed image.
        """

        kspace = kspace.permute(1, 2, 0, 3).unsqueeze(0)
        kspace = tensor_to_complex_np(kspace)

        # estimate sensitivity maps
        sens_maps = bart.bart(1, "ecalib -d0 -m1", kspace)

        # use Total Variation Minimization to reconstruct the image
        pred = bart.bart(
            1, f"pics -d0 -S -R T:7:0:{reg_wt} -i 200", kspace, sens_maps
        )
        pred = torch.from_numpy(np.abs(pred[0]))

        roi_center = tuple(i // 2 for i in pred.shape[-2:])
        cropper = SpatialCrop(roi_center=roi_center, roi_size=croped_size)
        pred = pred.unsqueeze(0)
        output_crp = cropper(pred)
        output_crp = output_crp.squeeze(0)

        return output_crp



    def run_model(idx):
        """
        Run BART on idx index from dataset.

        Args:
            idx (int): The index of the dataset.

        Returns:
            tuple: tuple with
                fname: Filename
                slice_num: Slice number.
                prediction: Reconstructed image.
        """
        data_dic = val_ds[idx]

        prediction_list = []

        masked_kspace, reg_wt, full_mri, masked_mri = data_dic['kspace_masked'], 0.01, data_dic['reconstruction_rss'], data_dic['kspace_masked_ifft']
        # print(full_mri.shape, masked_mri.shape)
        name = data_dic['name']

        croped_size = full_mri.shape[1:]
        zero_fill_list = []

        for slice, zero_fill_slide in zip(masked_kspace, masked_mri):
            recover_slide = cs_total_variation(slice, reg_wt, croped_size).numpy()
            prediction_list.append(recover_slide)
            roi_center = tuple(i // 2 for i in zero_fill_slide.shape[-2:])
            cropper = SpatialCrop(roi_center=roi_center, roi_size=croped_size)
            zero_fill_slide = zero_fill_slide.unsqueeze(0)
            zero_fill_crp = cropper(zero_fill_slide)
            zero_fill_crp = zero_fill_crp.squeeze(0)
            zero_fill_list.append(zero_fill_crp)
        
        prediction = np.stack(prediction_list)
        zero_fill = np.stack(zero_fill_list)
        full_mri = full_mri.numpy()

        # print(prediction.shape, prediction.dtype)
        # print(full_mri.shape, full_mri.dtype)
        # print(zero_fill.shape, zero_fill.dtype)

        cs_ssim, zero_ssim = skimage_ssim(prediction, full_mri), skimage_ssim(zero_fill, full_mri)

        save_reconstructions({name: prediction}, Path('./mri_results/cs_noise'))
        save_reconstructions({name: zero_fill}, Path('./mri_results/zero_noise'))


        return cs_ssim, zero_ssim

    def run_bart():
        start_time = time.perf_counter()
        ssim_list = []
        for i in range(len(val_ds)):
            # outputs.append(run_model(i))
            cs_ssim, zero_ssim = run_model(i)
            
            time_taken = time.perf_counter() - start_time
            logging.info(f"Run Time = {time_taken:} s")

            ssim_list.append((cs_ssim, zero_ssim))

        # with multiprocessing.Pool(args.num_workers) as pool:
        #     start_time = time.perf_counter()
        #     ssim_list = pool.map(run_model, range(len(val_ds)))
        #     time_taken = time.perf_counter() - start_time
        # logging.info(f"Run Time = {time_taken:} s")
        cs_list = list()
        zero_list = list()
        for cs_ssim, zero_ssim in ssim_list:
            cs_list.append(cs_ssim)
            zero_list.append(zero_ssim)
        print('cs ssim', np.mean(cs_list))
        print('zero fill ssim', np.mean(zero_list))

        
        # save_outputs(outputs, args.output_path)
    run_bart()
            
    # with torch.no_grad():
    #     val_ssim = list()
    #     val_loss = list()
    #     for val_data in val_loader:
    #         input, target, mean, std, fname = (
    #             val_data["kspace_masked_ifft"],
    #             val_data["reconstruction_rss"],
    #             val_data["mean"],
    #             val_data["std"],
    #             val_data["kspace_meta_dict"]["filename"],
    #         )

            # # iterate through all slices:
            # slice_dim = 1  # change this if another dimension is your slice dimension
            # num_slices = input.shape[slice_dim]
            # for i in range(num_slices):
            #     inp = input[:, i, ...].unsqueeze(slice_dim)
            #     tar = target[:, i, ...].unsqueeze(slice_dim)
            #     output = model(inp.to(device))

            #     vloss = loss_function(output, tar.to(device))
            #     val_loss.append(vloss.item())

            #     _std = std[0][i].item()
            #     _mean = mean[0][i].item()
            #     outputs[fname[0]].append(output.data.cpu().numpy()[0][0] * _std + _mean)
            #     targets[fname[0]].append(tar.numpy()[0][0] * _std + _mean)

    #     # compute validation ssims
    #     for fname in outputs:
    #         outputs[fname] = np.stack(outputs[fname])
    #         targets[fname] = np.stack(targets[fname])
    #         val_ssim.append(skimage_ssim(targets[fname], outputs[fname]))

    #     metric = np.mean(val_ssim)

        
    #     writer.add_scalar("val_mean_ssim", metric, epoch + 1)

    # writer.close()


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
        default=1,
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