method = 'zero'
de_type = 'blur'
# target = 'file_brain_AXT2_200_2000568.h5'
target = 'file_brain_AXT2_201_2010106.h5'

reconstruct_file = f'mri_results/{method}_{de_type}/' + target
print(reconstruct_file)
raw_file = '/NAS_liujie/liujie/fastMRI/brain/multicoil_val/' + target

import h5py
from monai.apps.reconstruction.complex_utils import complex_abs, convert_to_tensor_complex
from monai.apps.reconstruction.mri_utils import root_sum_of_squares
from monai.utils.type_conversion import convert_to_tensor
from monai.data.fft_utils import ifftn_centered
from evaluation_box.metrics import SSIM, PSNR, HFEN, MS_SSIM, FSIM, GMSD, HaarPSI, DSS, M_BRISQUE, NIQE, PIQE

with h5py.File(reconstruct_file, 'r') as original_file:
    reconstruction_rss_data = original_file['reconstruction'][()]
print(reconstruction_rss_data.shape)

with h5py.File(raw_file, 'r') as original_file:
    raw_rss_data = original_file['reconstruction_rss'][()]

print(reconstruction_rss_data.shape)

x = reconstruction_rss_data

print(SSIM(reconstruction_rss_data, raw_rss_data))
print(PSNR(reconstruction_rss_data, raw_rss_data))

import matplotlib.pyplot as plt
slide = x[x.shape[0]//2, :, :]
# a,b = slide.shape
# cropped_shape = reconstruction_rss_data.shape[-2:]
# c,d = cropped_shape
# start_x = (a - c) // 2
# start_y = (b - d) // 2
# center_region = slide[start_x:start_x + c, start_y:start_y + d]

plt.imshow(slide, cmap='gray')
plt.axis('off')

name = target.split('.')[0].split('_')[-1]
plt.savefig(f'cache/{method}_{de_type}_{name}.png', bbox_inches='tight', pad_inches=0)
plt.close()

x = raw_rss_data

slide = x[x.shape[0]//2, :, :]
plt.imshow(slide, cmap='gray')
plt.axis('off')

plt.savefig(f'cache/{name}.png', bbox_inches='tight', pad_inches=0)
plt.close()