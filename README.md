# alpha-brain

## Dataset Process

### fastMRI

#### Download
https://fastmri.med.nyu.edu/

please scroll to the bottom and fill the information, then click `submit`. You will revieve an email with download tutorial. 

Only following files are required to download:
```
Brain MRI:
brain_multicoil_train_batch_0 (~98.5 GB)
brain_multicoil_train_batch_1 (~92.1 GB)
brain_multicoil_train_batch_2 (~92.6 GB)
brain_multicoil_train_batch_3 (~95.5 GB)
brain_multicoil_train_batch_4 (~92.7 GB)
brain_multicoil_train_batch_5 (~94.3 GB)
brain_multicoil_train_batch_6 (~99.1 GB)
brain_multicoil_train_batch_7 (~95.7 GB)
brain_multicoil_train_batch_8 (~97.5 GB)
brain_multicoil_train_batch_9 (~88.3 GB)
brain_multicoil_val_batch_0 (~93.5 GB)
brain_multicoil_val_batch_1 (~88.7 GB)
brain_multicoil_val_batch_2 (~93.8 GB)
SHA256 Hash (0.5 KB)
```

```
brain
\multicoil_train
\multicoil_val
```

#### Analysis
AXFLAIR, AXT1, AXT1POST, AXT1PRE, AXT2 for brain

#### Data Format
(number of slices, number of coils, height, width)

### calgary-campinas_version-1.0
https://portal.conp.ca/dataset?id=projects/calgary-campinas

```
wget https://portal.conp.ca/data/calgary-campinas_version-1.0.tar.gz --no-check-certificate
tar -zxvf calgary-campinas_version-1.0.tar.gz
```
Only CC359/Raw-data/Multi-channel/12-channel/train_val_12_channel.zip is required.
```
mkdir calgary-campinas
cd calgary-campinas_version-1.0/CC359/Raw-data/Multi-channel/12-channel
unzip train_val_12_channel.zip
mv Train/** ../../../../../calgary-campinas
mv Val/** ../../../../../calgary-campinas
```

`python script/pretrain_cc.py --exp pretrain_cc`

### RealNoiseMRI
https://realnoisemri.grand-challenge.org/

`aws s3 sync --no-sign-request s3://openneuro.org/ds004332 ds004332-download/` 
`python real_noise_process.py`
Restore the nii.gz files in image directory as nii files.

More detail please refere to [link](https://openneuro.org/datasets/ds004332/versions/1.0.2)


## Training Code
### Requirements
```
apt install net-tools
conda create -n udmri python=3.9
conda activate udmri
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install monai==1.0
pip install 'monai[nibabel, skimage, pillow, tensorboard, gdown, itk, tqdm, lmdb, psutil, cucim, openslide, pandas, einops, transformers, mlflow, matplotlib, tensorboardX, tifffile, imagecodecs, pyyaml, fire, jsonschema, ninja, pynrrd, pydicom, h5py, nni, optuna]'
pip install numba
pip install opencv-python
pip install fastmri
pip install timm==0.4.12
pip install piq

```

### Synthetic MRI data
`python synthesis_blur.py`
`python synthesis_noise.py`
`python synthesis_nab.py`

### Sim2Real
`conda activate udmri`

For fastmri without any synthetic degradation pretrain.   
`CUDA_VISIBLE_DEVICES=0 python rn_script/train_realnoise.py --exp realnoise_fastmri_non0 --split 0 --resume_checkpoint True --checkpoint_dir pre_trained/unet_fastmri_brain_base.pt`
`CUDA_VISIBLE_DEVICES=0 python rn_script/train_realnoise.py --exp realnoise_fastmri_non1 --split 1 --resume_checkpoint True --checkpoint_dir pre_trained/unet_fastmri_brain_base.pt`
`CUDA_VISIBLE_DEVICES=0 python rn_script/train_realnoise.py --exp realnoise_fastmri_non2 --split 2 --resume_checkpoint True --checkpoint_dir pre_trained/unet_fastmri_brain_base.pt`
`CUDA_VISIBLE_DEVICES=0 python rn_script/train_realnoise.py --exp realnoise_fastmri_non3 --split 3 --resume_checkpoint True --checkpoint_dir pre_trained/unet_fastmri_brain_base.pt`
`CUDA_VISIBLE_DEVICES=0 python rn_script/train_realnoise.py --exp realnoise_fastmri_non4 --split 4 --resume_checkpoint True --checkpoint_dir pre_trained/unet_fastmri_brain_base.pt`

Eval: `CUDA_VISIBLE_DEVICES=0 python eval_realnoise.py --resume_checkpoint True --checkpoint_dir log/realnoise_fastmri_non`

For fastmri with noise pretrain.   done
`CUDA_VISIBLE_DEVICES=1 python rn_script/train_realnoise.py --exp realnoise_fastmri_noise0 --split 0 --resume_checkpoint True --checkpoint_dir pre_trained/unet_fastmri_brain_noise.pt`
`CUDA_VISIBLE_DEVICES=1 python rn_script/train_realnoise.py --exp realnoise_fastmri_noise1 --split 1 --resume_checkpoint True --checkpoint_dir pre_trained/unet_fastmri_brain_noise.pt`
`CUDA_VISIBLE_DEVICES=2 python rn_script/train_realnoise.py --exp realnoise_fastmri_noise2 --split 2 --resume_checkpoint True --checkpoint_dir pre_trained/unet_fastmri_brain_noise.pt`
`CUDA_VISIBLE_DEVICES=2 python rn_script/train_realnoise.py --exp realnoise_fastmri_noise3 --split 3 --resume_checkpoint True --checkpoint_dir pre_trained/unet_fastmri_brain_noise.pt`
`CUDA_VISIBLE_DEVICES=5 python rn_script/train_realnoise.py --exp realnoise_fastmri_noise4 --split 4 --resume_checkpoint True --checkpoint_dir pre_trained/unet_fastmri_brain_noise.pt`

Eval: `CUDA_VISIBLE_DEVICES=0 python eval_realnoise.py --resume_checkpoint True --checkpoint_dir log/realnoise_fastmri_noise`

For fastmri with blur pretrain. 
`CUDA_VISIBLE_DEVICES=0 python rn_script/train_realnoise.py --exp realnoise_fastmri_blur0 --split 0 --resume_checkpoint True --checkpoint_dir pre_trained/unet_fastmri_brain_blur.pt`
`CUDA_VISIBLE_DEVICES=0 python rn_script/train_realnoise.py --exp realnoise_fastmri_blur1 --split 1 --resume_checkpoint True --checkpoint_dir pre_trained/unet_fastmri_brain_blur.pt`
`CUDA_VISIBLE_DEVICES=1 python rn_script/train_realnoise.py --exp realnoise_fastmri_blur2 --split 2 --resume_checkpoint True --checkpoint_dir pre_trained/unet_fastmri_brain_blur.pt`
`CUDA_VISIBLE_DEVICES=1 python rn_script/train_realnoise.py --exp realnoise_fastmri_blur3 --split 3 --resume_checkpoint True --checkpoint_dir pre_trained/unet_fastmri_brain_blur.pt`
`CUDA_VISIBLE_DEVICES=2 python rn_script/train_realnoise.py --exp realnoise_fastmri_blur4 --split 4 --resume_checkpoint True --checkpoint_dir pre_trained/unet_fastmri_brain_blur.pt`

Eval: `CUDA_VISIBLE_DEVICES=0 python eval_realnoise.py --resume_checkpoint True --checkpoint_dir log/realnoise_fastmri_blur`

For fastmri with blur and noise pretrain.   
`CUDA_VISIBLE_DEVICES=2 python rn_script/train_realnoise.py --exp realnoise_fastmri_nab0 --split 0 --resume_checkpoint True --checkpoint_dir pre_trained/unet_fastmri_brain_nab.pt`
`CUDA_VISIBLE_DEVICES=5 python rn_script/train_realnoise.py --exp realnoise_fastmri_nab1 --split 1 --resume_checkpoint True --checkpoint_dir pre_trained/unet_fastmri_brain_nab.pt`
`CUDA_VISIBLE_DEVICES=5 python rn_script/train_realnoise.py --exp realnoise_fastmri_nab2 --split 2 --resume_checkpoint True --checkpoint_dir pre_trained/unet_fastmri_brain_nab.pt`
`CUDA_VISIBLE_DEVICES=6 python rn_script/train_realnoise.py --exp realnoise_fastmri_nab3 --split 3 --resume_checkpoint True --checkpoint_dir pre_trained/unet_fastmri_brain_nab.pt`
`CUDA_VISIBLE_DEVICES=6 python rn_script/train_realnoise.py --exp realnoise_fastmri_nab4 --split 4 --resume_checkpoint True --checkpoint_dir pre_trained/unet_fastmri_brain_nab.pt`

Eval: `CUDA_VISIBLE_DEVICES=0 python eval_realnoise.py --resume_checkpoint True --checkpoint_dir log/realnoise_fastmri_nab`

For train from scratch.  
`CUDA_VISIBLE_DEVICES=0 python rn_script/train_realnoise.py --exp real_noise0 --split 0`
`CUDA_VISIBLE_DEVICES=0 python rn_script/train_realnoise.py --exp real_noise1 --split 1`
`CUDA_VISIBLE_DEVICES=0 python rn_script/train_realnoise.py --exp real_noise2 --split 2`
`CUDA_VISIBLE_DEVICES=0 python rn_script/train_realnoise.py --exp real_noise3 --split 3`
`CUDA_VISIBLE_DEVICES=0 python rn_script/train_realnoise.py --exp real_noise4 --split 4`

Eval: `CUDA_VISIBLE_DEVICES=0 python eval_realnoise.py --resume_checkpoint True --checkpoint_dir log/real_noise`

### Compressed sensing (bart)
```
wget https://github.com/mrirecon/bart/archive/refs/tags/v0.8.00.tar.gz
tar xzvf bart-0.9.00.tar.gz
cd bart-0.9.00
make
export TOOLBOX_PATH=/data/liujie/bart/bart-0.9.00
export PYTHONPATH=${TOOLBOX_PATH}/python:${PYTHONPATH}
python script/bart_inference.py --exp compressed_sensing
python script/bart_inference.py --exp compressed_sensing --data_path_val /NAS_liujie/liujie/fastMRI/brain/multicoil_val_blur
python script/bart_inference.py --exp compressed_sensing --data_path_val /NAS_liujie/liujie/fastMRI/brain/multicoil_val_noise
python script/bart_inference.py --exp compressed_sensing --data_path_val /NAS_liujie/liujie/fastMRI/brain/multicoil_val_nab
```

### Deep Learning Model
Train unet for fastmri without synthetic degradation
`CUDA_VISIBLE_DEVICES=7 python script/unet_fastmri_non.py --exp pretrain_unet_fastmri_non`

Train unet for fastmri with synthetic blur
`CUDA_VISIBLE_DEVICES=7 python script/unet_fastmri_non.py --exp pretrain_unet_fastmri_blur --data_path_train /NAS_liujie/liujie/fastMRI/brain/multicoil_train_blur --data_path_val /NAS_liujie/liujie/fastMRI/brain/multicoil_val_blur` 

Train unet for fastmri with synthetic noise
`CUDA_VISIBLE_DEVICES=7 python script/unet_fastmri_non.py --exp pretrain_unet_fastmri_noise --data_path_train /NAS_liujie/liujie/fastMRI/brain/multicoil_train_noise --data_path_val /NAS_liujie/liujie/fastMRI/brain/multicoil_val_noise` 

Train unet for fastmri with synthetic noise and blur
`CUDA_VISIBLE_DEVICES=7 python script/unet_fastmri_non.py --exp pretrain_unet_fastmri_nab --data_path_train /NAS_liujie/liujie/fastMRI/brain/multicoil_train_nab --data_path_val /NAS_liujie/liujie/fastMRI/brain/multicoil_val_nab` 

Train varnet for fastmri without synthetic degradation
`CUDA_VISIBLE_DEVICES=6 python script/varnet_fastmri_non.py --exp pretrain_varnet_non`

Train varnet for fastmri with synthetic blur
`CUDA_VISIBLE_DEVICES=6 python script/varnet_fastmri_non.py --exp pretrain_varnet_blur --data_path_train /NAS_liujie/liujie/fastMRI/brain/multicoil_train_blur --data_path_val /NAS_liujie/liujie/fastMRI/brain/multicoil_val_blur` 

Train varnet for fastmri with synthetic noise
`CUDA_VISIBLE_DEVICES=4 python script/varnet_fastmri_non.py --exp pretrain_varnet_noise --data_path_train /NAS_liujie/liujie/fastMRI/brain/multicoil_train_noise --data_path_val /NAS_liujie/liujie/fastMRI/brain/multicoil_val_noise` 

Train varnet for fastmri with synthetic noise and blur
`CUDA_VISIBLE_DEVICES=3 python script/varnet_fastmri_non.py --exp pretrain_varnet_nab --data_path_train /NAS_liujie/liujie/fastMRI/brain/multicoil_train_nab --data_path_val /NAS_liujie/liujie/fastMRI/brain/multicoil_val_nab` 

For Vision Transformer
`git clone https://github.com/MLI-lab/transformers_for_imaging`

Train transformer for fastmri without synthetic degradation
`CUDA_VISIBLE_DEVICES=0 python script/swin_fastmri_non.py --exp swim_non --exp_dir swin_log`

Train transformer for fastmri with synthetic blur
`CUDA_VISIBLE_DEVICES=5 python script/swin_fastmri_non.py --exp swim_blur --exp_dir swin_log --data_path_train /NAS_liujie/liujie/fastMRI/brain/multicoil_train_blur --data_path_val /NAS_liujie/liujie/fastMRI/brain/multicoil_val_blur` 

Train transformer for fastmri with synthetic noise
`CUDA_VISIBLE_DEVICES=5 python script/swin_fastmri_non.py --exp swim_noise --exp_dir swin_log --data_path_train /NAS_liujie/liujie/fastMRI/brain/multicoil_train_noise --data_path_val /NAS_liujie/liujie/fastMRI/brain/multicoil_val_noise` 

Train transformer for fastmri with synthetic noise and blur
`CUDA_VISIBLE_DEVICES=5 python script/swin_fastmri_non.py --exp swim_nab --exp_dir swin_log --data_path_train /NAS_liujie/liujie/fastMRI/brain/multicoil_train_nab --data_path_val /NAS_liujie/liujie/fastMRI/brain/multicoil_val_nab` 


Evaluation
`CUDA_VISIBLE_DEVICES=0 python eval_swin.py --data_path_val /NAS_liujie/liujie/fastMRI/brain/multicoil_val --resume_checkpoint True --checkpoint_dir swin_log/swim_non`
 
`CUDA_VISIBLE_DEVICES=0 python eval_swin.py --data_path_val /NAS_liujie/liujie/fastMRI/brain/multicoil_val_blur --resume_checkpoint True --checkpoint_dir swin_log/swim_non`
 
`CUDA_VISIBLE_DEVICES=0 python eval_swin.py --data_path_val /NAS_liujie/liujie/fastMRI/brain/multicoil_val_noise --resume_checkpoint True --checkpoint_dir swin_log/swim_non`
 
`CUDA_VISIBLE_DEVICES=0 python eval_swin.py --data_path_val /NAS_liujie/liujie/fastMRI/brain/multicoil_val_nab --resume_checkpoint True --checkpoint_dir swin_log/swim_non`
 

`CUDA_VISIBLE_DEVICES=0 python eval_unet.py --data_path_val /NAS_liujie/liujie/fastMRI/brain/multicoil_val --resume_checkpoint True --checkpoint_dir log/pretrain_unet_fastmri_non`
 
`CUDA_VISIBLE_DEVICES=0 python eval_unet.py --data_path_val /NAS_liujie/liujie/fastMRI/brain/multicoil_val_blur --resume_checkpoint True --checkpoint_dir log/pretrain_unet_fastmri_non`
 
`CUDA_VISIBLE_DEVICES=0 python eval_unet.py --data_path_val /NAS_liujie/liujie/fastMRI/brain/multicoil_val_noise --resume_checkpoint True --checkpoint_dir log/pretrain_unet_fastmri_non`
 
`CUDA_VISIBLE_DEVICES=0 python eval_unet.py --data_path_val /NAS_liujie/liujie/fastMRI/brain/multicoil_val_nab --resume_checkpoint True --checkpoint_dir log/pretrain_unet_fastmri_non`
 

`CUDA_VISIBLE_DEVICES=1 python eval_varnet.py --data_path_val /NAS_liujie/liujie/fastMRI/brain/multicoil_val --resume_checkpoint True --checkpoint_dir varnet_log/varnet_non`
 
`CUDA_VISIBLE_DEVICES=1 python eval_varnet.py --data_path_val /NAS_liujie/liujie/fastMRI/brain/multicoil_val_blur --resume_checkpoint True --checkpoint_dir varnet_log/varnet_non`
 
`CUDA_VISIBLE_DEVICES=2 python eval_varnet.py --data_path_val /NAS_liujie/liujie/fastMRI/brain/multicoil_val_noise --resume_checkpoint True --checkpoint_dir varnet_log/varnet_non`
 
`CUDA_VISIBLE_DEVICES=2 python eval_varnet.py --data_path_val /NAS_liujie/liujie/fastMRI/brain/multicoil_val_nab --resume_checkpoint True --checkpoint_dir varnet_log/varnet_non`
 

Enhancement
`CUDA_VISIBLE_DEVICES=0 python eval_swin.py --data_path_val /NAS_liujie/liujie/fastMRI/brain/multicoil_val_nab --resume_checkpoint True --checkpoint_dir swin_log/swim_non`

`CUDA_VISIBLE_DEVICES=0 python eval_swin.py --data_path_val /NAS_liujie/liujie/fastMRI/brain/multicoil_val_nab --resume_checkpoint True --checkpoint_dir swin_log/swim_noise`

`CUDA_VISIBLE_DEVICES=0 python eval_swin.py --data_path_val /NAS_liujie/liujie/fastMRI/brain/multicoil_val_nab --resume_checkpoint True --checkpoint_dir swin_log/swim_blur`

`CUDA_VISIBLE_DEVICES=0 python eval_swin.py --data_path_val /NAS_liujie/liujie/fastMRI/brain/multicoil_val_nab --resume_checkpoint True --checkpoint_dir swin_log/swim_nab`


`CUDA_VISIBLE_DEVICES=0 python eval_unet.py --data_path_val /NAS_liujie/liujie/fastMRI/brain/multicoil_val_nab --resume_checkpoint True --checkpoint_dir log/pretrain_unet_fastmri_non`

`CUDA_VISIBLE_DEVICES=0 python eval_unet.py --data_path_val /NAS_liujie/liujie/fastMRI/brain/multicoil_val_nab --resume_checkpoint True --checkpoint_dir log/pretrain_unet_fastmri_noise`

`CUDA_VISIBLE_DEVICES=0 python eval_unet.py --data_path_val /NAS_liujie/liujie/fastMRI/brain/multicoil_val_nab --resume_checkpoint True --checkpoint_dir log/pretrain_unet_fastmri_blur`

`CUDA_VISIBLE_DEVICES=0 python eval_unet.py --data_path_val /NAS_liujie/liujie/fastMRI/brain/multicoil_val_nab --resume_checkpoint True --checkpoint_dir log/pretrain_unet_fastmri_nab`


`CUDA_VISIBLE_DEVICES=1 python eval_varnet.py --data_path_val /NAS_liujie/liujie/fastMRI/brain/multicoil_val_nab --resume_checkpoint True --checkpoint_dir varnet_log/varnet_non`

`CUDA_VISIBLE_DEVICES=1 python eval_varnet.py --data_path_val /NAS_liujie/liujie/fastMRI/brain/multicoil_val_nab --resume_checkpoint True --checkpoint_dir varnet_log/pretrain_varnet_noise`

`CUDA_VISIBLE_DEVICES=2 python eval_varnet.py --data_path_val /NAS_liujie/liujie/fastMRI/brain/multicoil_val_nab --resume_checkpoint True --checkpoint_dir varnet_log/pretrain_varnet_blur`

`CUDA_VISIBLE_DEVICES=2 python eval_varnet.py --data_path_val /NAS_liujie/liujie/fastMRI/brain/multicoil_val_nab --resume_checkpoint True --checkpoint_dir varnet_log/pretrain_varnet_nab`