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

from monai.networks.nets import BasicUNet

from pathlib import Path
import argparse
from monai.data import CacheDataset, DataLoader, decollate_batch, Dataset
from torch.utils.tensorboard import SummaryWriter

from data.dataloader import CC_Dataset

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

dim = (218,170)
n_channels = 24 # 12-channels*2 (real and imaginary)
nslices = 256
crop = (30,30) # Crops slices with little anatomy

def trainer(args):
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    outpath = os.path.join(args.exp_dir, args.exp)
    Path(outpath).mkdir(parents=True, exist_ok=True)  # create output directory to store model checkpoints
    now = datetime.now()
    date = now.strftime("%m-%d-%y_%H-%M")
    writer = SummaryWriter(
        outpath + "/" + date
    )  # create a date directory within the output directory for storing training logs

    ## create dataset
    # require input image and target image
    h5_list = glob.glob(("/NAS_liujie/liujie/calgary-campinas/*.h5").__str__())
    train = h5_list[: int(len(h5_list)*0.9)]
    test = h5_list[int(len(h5_list)*0.9) :]
    print("Train:",len(train))
    under_masks = np.load("./common/R5_218x170.npy")

    train_dataset = CC_Dataset(train, dim = dim, under_masks = under_masks,  crop = crop,\
                                batch_size = 1, n_channels = n_channels,nslices = nslices, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_dataset = CC_Dataset(test, dim = dim, under_masks = under_masks,  crop = crop,\
                                batch_size = 1, n_channels = n_channels,nslices = nslices, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # create the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BasicUNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        features=args.features,
    ).to(device)
    print("#model_params:", np.sum([len(p.flatten()) for p in model.parameters()]))
    if args.resume_checkpoint:
        model.load_state_dict(torch.load(args.checkpoint_dir))
        print("resume training from a given checkpoint...")

    # create the loss function
    loss_function = torch.nn.L1Loss()

    # create the optimizer and the learning rate scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_step_size, args.lr_gamma)

    # start a typical PyTorch training loop
    val_interval = 5  # doing validation every 2 epochs
    best_metric = -1
    best_metric_epoch = -1
    tic = time.time()
    for epoch in range(args.num_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{args.num_epochs}")
        model.train()
        epoch_loss = 0
        input_num = 32
        step = 0
        for batch_data in train_loader:
            step+=1
            input, target = (
                batch_data["input"],
                batch_data["gt"],
            )

            optimizer.zero_grad()

            inp = input.float()
            tar = target.float()
            # print(inp.shape, tar.shape)
            output = model(inp.to(device))

            loss = loss_function(output, tar.to(device))

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print(f"{step}, train_loss: {epoch_loss/step:.4f}", "\r", end="")
        scheduler.step()
        epoch_loss /= step
        writer.add_scalar("train_loss", epoch_loss, epoch + 1)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f} time elapsed: {(time.time()-tic)/60:.2f} mins")
        torch.save(model.state_dict(), os.path.join(outpath + "/" + date, "unet_mri_reconstruction.pt"))

        # # validation
        # if (epoch + 1) % val_interval == 0:
        #     model.eval()
        #     with torch.no_grad():
        #         val_ssim = list()
        #         val_loss = list()
        #         val_step = 0
        #         for val_data in val_loader:
        #             input, target = (
        #                 val_data["input"],
        #                 val_data["gt"],
        #             )


        #             inp = input.float()
        #             tar = target.float()
        #             output = model(inp.to(device))

        #             vloss = loss_function(output, tar.to(device))
        #             val_loss.append(vloss.item())

        #             val_step += 1
        #             ssim = skimage_ssim(target[:, 0, :, :].cpu().numpy(), output[:, 0, :, :].cpu().numpy())
        #             print(f"{val_step}, ssim: {ssim:.4f}", "\r", end="")
        #             val_ssim.append(ssim)

        #         metric = np.mean(val_ssim)

        #         # save the best checkpoint so far
        #         if metric > best_metric:
        #             best_metric = metric
        #             best_metric_epoch = epoch + 1
        #             torch.save(model.state_dict(), os.path.join(outpath + "/" + date, "unet_mri_reconstruction.pt"))
        #             print("saved new best metric model")
        #         print(
        #             "current epoch: {} current mean ssim: {:.4f} best mean ssim: {:.4f} at epoch {}".format(
        #                 epoch + 1, metric, best_metric, best_metric_epoch
        #             )
        #         )
        #         writer.add_scalar("val_mean_ssim", metric, epoch + 1)

    print(f"training completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    writer.close()


def main():
    parser = argparse.ArgumentParser()

    # data loader arguments
    parser.add_argument(
        "--batch_size",
        default=64,
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
    parser.add_argument("--num_epochs", default=100, type=int, help="number of training epochs")

    parser.add_argument("--split", default=0, type=int, help="from 0~4")

    parser.add_argument("--exp_dir", default='./log/', type=Path, help="output directory to save training logs")

    parser.add_argument(
        "--exp",
        default=None,
        type=str,
        help="experiment name (a folder will be created with this name to store the results)",
    )

    parser.add_argument("--lr", default=0.0001, type=float, help="learning rate")

    parser.add_argument("--lr_step_size", default=200, type=int, help="decay learning rate every lr_step_size epochs")

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

if __name__ == "__main__":
    main()

# CUDA_VISIBLE_DEVICES=1 python train.py --exp real_noise