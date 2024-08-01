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

from collections.abc import Hashable, Mapping, Sequence

import numpy as np
from numpy import ndarray
from torch import Tensor

from common.array import RadialKspaceMask, SpiralKspaceMask
from monai.config import DtypeLike, KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms.croppad.array import SpatialCrop
from monai.transforms.croppad.dictionary import Cropd
from monai.transforms.intensity.array import NormalizeIntensity
from monai.transforms.transform import MapTransform, RandomizableTransform
from monai.utils import FastMRIKeys
from monai.utils.type_conversion import convert_to_tensor
from monai.apps.reconstruction.transforms.array import EquispacedKspaceMask, RandomKspaceMask

class RandomKspaceMaskd(RandomizableTransform, MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.apps.reconstruction.transforms.array.RandomKspacemask`.
    Other mask transforms can inherit from this class, for example:
    :py:class:`monai.apps.reconstruction.transforms.dictionary.EquispacedKspaceMaskd`.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: monai.transforms.MapTransform
        center_fractions: Fraction of low-frequency columns to be retained.
            If multiple values are provided, then one of these numbers is
            chosen uniformly each time.
        accelerations: Amount of under-sampling. This should have the
            same length as center_fractions. If multiple values are provided,
            then one of these is chosen uniformly each time.
        seed: set the random seed.
        spatial_dims: Number of spatial dims (e.g., it's 2 for a 2D data; it's
            also 2 for pseudo-3D datasets like the fastMRI dataset).
            The last spatial dim is selected for sampling. For the fastMRI
            dataset, k-space has the form (...,num_slices,num_coils,H,W)
            and sampling is done along W. For a general 3D data with the
            shape (...,num_coils,H,W,D), sampling is done along D.
        is_complex: if True, then the last dimension will be reserved
            for real/imaginary parts.
        allow_missing_keys: don't raise exception if key is missing.
    """

    backend = RandomKspaceMask.backend

    def __init__(
        self,
        keys: KeysCollection,
        center_fractions: Sequence[float],
        accelerations: Sequence[float],
        seed: int,
        spatial_dims: int = 2,
        is_complex: bool = True,
        allow_missing_keys: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        self.masker = RandomKspaceMask(
            center_fractions=center_fractions,
            accelerations=accelerations,
            spatial_dims=spatial_dims,
            is_complex=is_complex,
        )
        self.set_random_state(seed = seed)

    def set_random_state(
        self, seed: int | None = None, state: np.random.RandomState | None = None
    ) -> RandomKspaceMaskd:
        super().set_random_state(seed, state)
        self.masker.set_random_state(seed, state)
        return self

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, Tensor]:
        """
        Args:
            data: is a dictionary containing (key,value) pairs from the
                loaded dataset

        Returns:
            the new data dictionary
        """
        d = dict(data)
        for key in self.key_iterator(d):
            d[key + "_masked"], d[key + "_masked_ifft"] = self.masker(d[key])
            d[FastMRIKeys.MASK] = self.masker.mask

        return d  # type: ignore


class EquispacedKspaceMaskd(RandomKspaceMaskd):
    """
    Dictionary-based wrapper of
    :py:class:`monai.apps.reconstruction.transforms.array.EquispacedKspaceMask`.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: monai.transforms.MapTransform
        center_fractions: Fraction of low-frequency columns to be retained.
            If multiple values are provided, then one of these numbers is
            chosen uniformly each time.
        accelerations: Amount of under-sampling. This should have the same
            length as center_fractions. If multiple values are provided,
            then one of these is chosen uniformly each time.
        seed: set the random seed.
        spatial_dims: Number of spatial dims (e.g., it's 2 for a 2D data;
            it's also 2 for  pseudo-3D datasets like the fastMRI dataset).
            The last spatial dim is selected for sampling. For the fastMRI
            dataset, k-space has the form (...,num_slices,num_coils,H,W)
            and sampling is done along W. For a general 3D data with the shape
            (...,num_coils,H,W,D), sampling is done along D.
        is_complex: if True, then the last dimension will be reserved
            for real/imaginary parts.
        allow_missing_keys: don't raise exception if key is missing.
    """

    backend = EquispacedKspaceMask.backend

    def __init__(
        self,
        keys: KeysCollection,
        center_fractions: Sequence[float],
        accelerations: Sequence[float],
        seed: int,
        spatial_dims: int = 2,
        is_complex: bool = True,
        allow_missing_keys: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        self.masker = EquispacedKspaceMask(  # type: ignore
            center_fractions=center_fractions,
            accelerations=accelerations,
            spatial_dims=spatial_dims,
            is_complex=is_complex,
        )
        self.set_random_state(seed = seed)

    def set_random_state(
        self, seed: int | None = None, state: np.random.RandomState | None = None
    ) -> EquispacedKspaceMaskd:
        super().set_random_state(seed, state)
        self.masker.set_random_state(seed, state)
        return self


class RadialKspaceMaskd(RandomizableTransform, MapTransform):
    """
    Dictionary-based wrapper of `common.array.RadialKspaceMask`.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: monai.transforms.MapTransform
        accelerations: Amount of under-sampling. This should have the
            same length as center_fractions. If multiple values are provided,
            then one of these is chosen uniformly each time.
        spatial_dims: Number of spatial dims (e.g., it's 2 for a 2D data; it's
            also 2 for pseudo-3D datasets like the fastMRI dataset).
            The last spatial dim is selected for sampling. For the fastMRI
            dataset, k-space has the form (...,num_slices,num_coils,H,W)
            and sampling is done along W. For a general 3D data with the
            shape (...,num_coils,H,W,D), sampling is done along D.
        is_complex: if True, then the last dimension will be reserved
            for real/imaginary parts.
        allow_missing_keys: don't raise exception if key is missing.
    """

    backend = RadialKspaceMask.backend

    ## replace RandomKspaceMask

    def __init__(
        self,
        keys: KeysCollection,
        accelerations: Sequence[float],
        seed: int,
        spatial_dims: int = 2,
        is_complex: bool = True,
        allow_missing_keys: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        self.masker = RadialKspaceMask(
            accelerations=accelerations,
            spatial_dims=spatial_dims,
            is_complex=is_complex,
        )
        self.seed = seed
        self.set_random_state(seed = seed)

    def set_random_state(
        self, seed: int | None = None, state: np.random.RandomState | None = None
    ) -> RadialKspaceMaskd:
        super().set_random_state(seed, state)
        self.masker.set_random_state(seed, state)
        return self

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, Tensor]:
        """
        Args:
            data: is a dictionary containing (key,value) pairs from the
                loaded dataset

        Returns:
            the new data dictionary
        """
        d = dict(data)
        for key in self.key_iterator(d):
            d[key + "_masked"], d[key + "_masked_ifft"] = self.masker(d[key], self.seed)
            d[FastMRIKeys.MASK] = self.masker.mask

        return d  # type: ignore


class SpiralKspaceMaskd(RandomizableTransform, MapTransform):
    """
    Dictionary-based wrapper of `common.array.SpiralKspaceMask`.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: monai.transforms.MapTransform
        accelerations: Amount of under-sampling. This should have the
            same length as center_fractions. If multiple values are provided,
            then one of these is chosen uniformly each time.
        spatial_dims: Number of spatial dims (e.g., it's 2 for a 2D data; it's
            also 2 for pseudo-3D datasets like the fastMRI dataset).
            The last spatial dim is selected for sampling. For the fastMRI
            dataset, k-space has the form (...,num_slices,num_coils,H,W)
            and sampling is done along W. For a general 3D data with the
            shape (...,num_coils,H,W,D), sampling is done along D.
        is_complex: if True, then the last dimension will be reserved
            for real/imaginary parts.
        allow_missing_keys: don't raise exception if key is missing.
    """

    backend = SpiralKspaceMask.backend

    ## replace RandomKspaceMask

    def __init__(
        self,
        keys: KeysCollection,
        accelerations: Sequence[float],
        seed: int,
        spatial_dims: int = 2,
        is_complex: bool = True,
        allow_missing_keys: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        self.masker = SpiralKspaceMask(
            accelerations=accelerations,
            spatial_dims=spatial_dims,
            is_complex=is_complex,
        )
        self.seed = seed
        self.set_random_state(seed = seed)

    def set_random_state(
        self, seed: int | None = None, state: np.random.RandomState | None = None
    ) -> SpiralKspaceMaskd:
        super().set_random_state(seed, state)
        self.masker.set_random_state(seed, state)
        return self

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, Tensor]:
        """
        Args:
            data: is a dictionary containing (key,value) pairs from the
                loaded dataset

        Returns:
            the new data dictionary
        """
        d = dict(data)
        for key in self.key_iterator(d):
            d[key + "_masked"], d[key + "_masked_ifft"] = self.masker(d[key], self.seed)
            d[FastMRIKeys.MASK] = self.masker.mask

        return d  # type: ignore