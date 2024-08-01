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

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence

import numpy as np
from torch import Tensor

from monai.apps.reconstruction.complex_utils import complex_abs, convert_to_tensor_complex
from monai.apps.reconstruction.mri_utils import root_sum_of_squares
from monai.config.type_definitions import NdarrayOrTensor
from monai.data.fft_utils import ifftn_centered
from monai.transforms.transform import RandomizableTransform
from monai.utils.enums import TransformBackends
from monai.utils.type_conversion import convert_to_tensor

from common.mask_func import RadialMaskFunc, SpiralMaskFunc
# from common.mask_function import RadialMaskFunction, SpiralMaskFunction


class KspaceCIRCUSMask(RandomizableTransform):
    """
    A basic class for under-sampling mask setup. It provides common
    features for under-sampling mask generators.
    For example, RandomMaskFunc and EquispacedMaskFunc (two mask
    transform objects defined right after this module)
    both inherit MaskFunc to properly setup properties like the
    acceleration factor.
    """

    def __init__(
        self,
        accelerations: Sequence[float],
        spatial_dims: int = 2,
        is_complex: bool = True,
    ):
        """
        Args:
            accelerations: Amount of under-sampling. This should have the
                same length as center_fractions. If multiple values are
                provided, then one of these is chosen uniformly each time.
            spatial_dims: Number of spatial dims (e.g., it's 2 for a 2D data;
                it's also 2 for pseudo-3D datasets like the fastMRI dataset).
                The last spatial dim is selected for sampling. For the fastMRI
                dataset, k-space has the form (...,num_slices,num_coils,H,W)
                and sampling is done along W. For a general 3D data with the
                shape (...,num_coils,H,W,D), sampling is done along D.
            is_complex: if True, then the last dimension will be reserved for
                real/imaginary parts.
        """

        self.accelerations = accelerations
        self.spatial_dims = spatial_dims
        self.is_complex = is_complex

    @abstractmethod
    def __call__(self, kspace: NdarrayOrTensor) -> Sequence[Tensor]:
        """
        This is an extra instance to allow for defining new mask generators.
        For creating other mask transforms, define a new class and simply
        override __call__. See an example of this in
        :py:class:`monai.apps.reconstruction.transforms.array.RandomKspacemask`.

        Args:
            kspace: The input k-space data. The shape is (...,num_coils,H,W,2)
                for complex 2D inputs and (...,num_coils,H,W,D) for real 3D
                data.
        """
        raise NotImplementedError

    def randomize_choose_acceleration(self) -> Sequence[float]:
        """
        If multiple values are provided for center_fractions and
        accelerations, this function selects one value uniformly
        for each training/test sample.

        Returns:
            A tuple containing
                (1) center_fraction: chosen fraction of center kspace
                lines to exclude from under-sampling
                (2) acceleration: chosen acceleration factor
        """
        choice = self.R.randint(0, len(self.accelerations))
        acceleration = self.accelerations[choice]
        return acceleration

class RadialKspaceMask(KspaceCIRCUSMask):
    """Implementation of adial using CIRCUS as shown in [1]_. It creates radial masks for acquired data on a grid.

    References
    ----------

    .. [1] Liu J, Saloner D. Accelerated MRI with CIRcular Cartesian UnderSampling (CIRCUS): a variable density
        Cartesian sampling strategy for compressed sensing and parallel imaging. Quant Imaging Med Surg.
        2014 Feb;4(1):57-67. doi: 10.3978/j.issn.2223-4292.2014.02.01. PMID: 24649436; PMCID: PMC3947985.
    """
    backend = [TransformBackends.TORCH]

    def __init__(
        self,
        accelerations: Sequence[float],
        spatial_dims: int = 2,
        is_complex: bool = True,
    ):
        """
        Args:
            accelerations: Amount of under-sampling. This should have the
                same length as center_fractions. If multiple values are
                provided, then one of these is chosen uniformly each time.
            spatial_dims: Number of spatial dims (e.g., it's 2 for a 2D data;
                it's also 2 for pseudo-3D datasets like the fastMRI dataset).
                The last spatial dim is selected for sampling. For the fastMRI
                dataset, k-space has the form (...,num_slices,num_coils,H,W)
                and sampling is done along W. For a general 3D data with the
                shape (...,num_coils,H,W,D), sampling is done along D.
            is_complex: if True, then the last dimension will be reserved for
                real/imaginary parts.
        """
        super().__init__(
            accelerations=accelerations,
            spatial_dims=spatial_dims,
            is_complex=is_complex,
        )

        self.mask_func = RadialMaskFunc(
            accelerations=accelerations,
        )


    def __call__(self, kspace: NdarrayOrTensor, seed: int) -> Sequence[Tensor]:
        """
        Args:
            kspace: The input k-space data. The shape is (...,num_coils,H,W,2)
                for complex 2D inputs and (...,num_coils,H,W,D) for real 3D
                data. The last spatial dim is selected for sampling. For the
                fastMRI multi-coil dataset, k-space has the form
                (...,num_slices,num_coils,H,W) and sampling is done along W.
                For a general 3D data with the shape (...,num_coils,H,W,D),
                sampling is done along D.

        Returns:
            A tuple containing
                (1) the under-sampled kspace
                (2) absolute value of the inverse fourier of the under-sampled kspace
        
        Notes: only implement is_complex version
        """
        kspace_t = convert_to_tensor_complex(kspace)
        # print(kspace.shape, kspace_t.shape) # torch.Size([16, 4, 640, 320]) torch.Size([16, 4, 640, 320, 2])
        spatial_size = kspace_t.shape
        num_cols = spatial_size[-1]
        num_rows = spatial_size[-2]
        if self.is_complex:  # for complex data
            num_cols = spatial_size[-2]
            num_rows = spatial_size[-3]

        ## generate mask

        mask = self.mask_func(spatial_size[2:], seed=seed)

        # Reshape the mask
        mask_shape = [1 for _ in spatial_size]
        if self.is_complex:
            mask_shape[-2] = num_cols
            mask_shape[-3] = num_rows
        else:
            mask_shape[-1] = num_cols
            mask_shape[-2] = num_rows


        mask = convert_to_tensor(mask.reshape(*mask_shape))

        # under-sample the ksapce
        masked = mask * kspace_t
        masked_kspace: Tensor = convert_to_tensor(masked)
        self.mask = mask

        # visualize the mask
        # import matplotlib.pyplot as plt
        # import numpy as np
        # plt.imshow(mask[0,0,:,:,0], cmap='gray')
        # plt.axis('off')
        # plt.savefig('cache/mask_radial.png')

        # compute inverse fourier of the masked kspace
        masked_kspace_ifft: Tensor = convert_to_tensor(
            complex_abs(ifftn_centered(masked_kspace, spatial_dims=self.spatial_dims, is_complex=self.is_complex))
        )

        # combine coil images (it is assumed that the coil dimension is
        masked_kspace_ifft_rss: Tensor = convert_to_tensor(
            root_sum_of_squares(masked_kspace_ifft, spatial_dim=-self.spatial_dims - 1)
        )
        return masked_kspace, masked_kspace_ifft_rss


class SpiralKspaceMask(KspaceCIRCUSMask):
    """Implementation of adial using CIRCUS as shown in [1]_. It creates radial masks for acquired data on a grid.

    References
    ----------

    .. [1] Liu J, Saloner D. Accelerated MRI with CIRcular Cartesian UnderSampling (CIRCUS): a variable density
        Cartesian sampling strategy for compressed sensing and parallel imaging. Quant Imaging Med Surg.
        2014 Feb;4(1):57-67. doi: 10.3978/j.issn.2223-4292.2014.02.01. PMID: 24649436; PMCID: PMC3947985.
    """

    backend = [TransformBackends.TORCH]

    def __init__(
        self,
        accelerations: Sequence[float],
        spatial_dims: int = 2,
        is_complex: bool = True,
    ):
        """
        Args:
            accelerations: Amount of under-sampling. This should have the
                same length as center_fractions. If multiple values are
                provided, then one of these is chosen uniformly each time.
            spatial_dims: Number of spatial dims (e.g., it's 2 for a 2D data;
                it's also 2 for pseudo-3D datasets like the fastMRI dataset).
                The last spatial dim is selected for sampling. For the fastMRI
                dataset, k-space has the form (...,num_slices,num_coils,H,W)
                and sampling is done along W. For a general 3D data with the
                shape (...,num_coils,H,W,D), sampling is done along D.
            is_complex: if True, then the last dimension will be reserved for
                real/imaginary parts.
        """
        super().__init__(
            accelerations=accelerations,
            spatial_dims=spatial_dims,
            is_complex=is_complex,
        )

        self.mask_func = SpiralMaskFunc(
            accelerations=accelerations,
        )

    def __call__(self, kspace: NdarrayOrTensor, seed: int) -> Sequence[Tensor]:
        """
        Args:
            kspace: The input k-space data. The shape is (...,num_coils,H,W,2)
                for complex 2D inputs and (...,num_coils,H,W,D) for real 3D
                data. The last spatial dim is selected for sampling. For the
                fastMRI multi-coil dataset, k-space has the form
                (...,num_slices,num_coils,H,W) and sampling is done along W.
                For a general 3D data with the shape (...,num_coils,H,W,D),
                sampling is done along D.

        Returns:
            A tuple containing
                (1) the under-sampled kspace
                (2) absolute value of the inverse fourier of the under-sampled kspace
        
        Notes: only implement is_complex version
        """
        kspace_t = convert_to_tensor_complex(kspace)
        # print(kspace.shape, kspace_t.shape) # torch.Size([16, 4, 640, 320]) torch.Size([16, 4, 640, 320, 2])
        spatial_size = kspace_t.shape
        num_cols = spatial_size[-1]
        num_rows = spatial_size[-2]
        if self.is_complex:  # for complex data
            num_cols = spatial_size[-2]
            num_rows = spatial_size[-3]

        ## generate mask

        mask = self.mask_func(spatial_size[2:], seed=seed)

        # Reshape the mask
        mask_shape = [1 for _ in spatial_size]
        if self.is_complex:
            mask_shape[-2] = num_cols
            mask_shape[-3] = num_rows
        else:
            mask_shape[-1] = num_cols
            mask_shape[-2] = num_rows

        mask = convert_to_tensor(mask.reshape(*mask_shape))

        # under-sample the ksapce
        masked = mask * kspace_t
        masked_kspace: Tensor = convert_to_tensor(masked)
        self.mask = mask

        # # visualize the mask
        # import matplotlib.pyplot as plt
        # import numpy as np
        # plt.imshow(mask[0,0,:,:,0], cmap='gray')
        # plt.axis('off')
        # plt.savefig('cache/mask_spiral.png')

        # compute inverse fourier of the masked kspace
        masked_kspace_ifft: Tensor = convert_to_tensor(
            complex_abs(ifftn_centered(masked_kspace, spatial_dims=self.spatial_dims, is_complex=self.is_complex))
        )

        # combine coil images (it is assumed that the coil dimension is
        masked_kspace_ifft_rss: Tensor = convert_to_tensor(
            root_sum_of_squares(masked_kspace_ifft, spatial_dim=-self.spatial_dims - 1)
        )
        return masked_kspace, masked_kspace_ifft_rss
