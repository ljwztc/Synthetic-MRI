# Code and comments can be shared with code of FastMRI under the same MIT license:
# https://github.com/facebookresearch/fastMRI/
# The code can have been adjusted to our needs.

import contextlib
import logging
from abc import abstractmethod
from enum import Enum
from typing import Iterable, List, Optional, Tuple, Union, Callable

import numpy as np
import torch
from numba import njit

__all__ = (
    "RadialMaskFunc",
    "SpiralMaskFunc",
)

Number = Union[float, int]

logger = logging.getLogger(__name__)
GOLDEN_RATIO = (1 + np.sqrt(5)) / 2


def center_crop(data: torch.Tensor, shape: Union[List[int], Tuple[int, ...]]) -> torch.Tensor:
    """Apply a center crop along the last two dimensions.

    Parameters
    ----------
    data: torch.Tensor
    shape: List or tuple of ints
        The output shape, should be smaller than the corresponding data dimensions.

    Returns
    -------
    torch.Tensor: The center cropped data.
    """
    # TODO: Make dimension independent.
    if not (0 < shape[0] <= data.shape[-2]) or not (0 < shape[1] <= data.shape[-1]):
        raise ValueError(f"Crop shape should be smaller than data. Requested {shape}, got {data.shape}.")

    width_lower = (data.shape[-2] - shape[0]) // 2
    width_upper = width_lower + shape[0]
    height_lower = (data.shape[-1] - shape[1]) // 2
    height_upper = height_lower + shape[1]

    return data[..., width_lower:width_upper, height_lower:height_upper]


@contextlib.contextmanager
def temp_seed(rng, seed):
    state = rng.get_state()
    rng.seed(seed)
    try:
        yield
    finally:
        rng.set_state(state)


@njit
def get_square_ordered_idxs(square_side_size: int, square_id: int):
    """Returns ordered (clockwise) indices of a sub-square of a square matrix.

    Parameters
    ----------
    square_side_size: int
        Square side size. Dim of array.
    square_id: int
        Number of sub-square. Can be 0, ..., square_side_size // 2.

    Returns
    -------
    ordered_idxs: List of tuples.
        Indices of each point that belongs to the square_id-th sub-square
        starting from top-left point clockwise.
    """
    assert square_id in range(square_side_size // 2)

    ordered_idxs = list()

    for col in range(square_id, square_side_size - square_id):
        ordered_idxs.append((square_id, col))

    for row in range(square_id + 1, square_side_size - (square_id + 1)):
        ordered_idxs.append((row, square_side_size - (square_id + 1)))

    for col in range(square_side_size - (square_id + 1), square_id, -1):
        ordered_idxs.append((square_side_size - (square_id + 1), col))

    for row in range(square_side_size - (square_id + 1), square_id, -1):
        ordered_idxs.append((row, square_id))

    return ordered_idxs


@njit
def accelerated_loop_spiral(shape, acceleration, c):
    max_dim = max(shape) - max(shape) % 2
    min_dim = min(shape) - min(shape) % 2

    num_nested_squares = max_dim // 2

    M = int(shape[0] * shape[1] / (acceleration * (max_dim / 2 - (max_dim - min_dim) * (1 + min_dim / max_dim) / 4)))
    
    mask = np.zeros((max_dim, max_dim), dtype=np.float32)

    for square_id in range(num_nested_squares):
        ordered_indices = get_square_ordered_idxs(
            square_side_size=max_dim,
            square_id=square_id,
        )

        # J: size of the square, J=2,…,N, i.e., the number of points along one side of the square
        J = 2 * (num_nested_squares - square_id)
        # K: total number of points along the perimeter of the square K=4·J-4;
        K = 4 * (J - 1)

        for m in range(M):
            i = np.floor(np.mod(m / GOLDEN_RATIO, 1) * K)
            indices_idx = int(np.mod((i + np.ceil(J**c) - 1), K))

            mask[ordered_indices[indices_idx]] = 1.0
    
    return mask


@njit
def accelerated_loop_radial(shape, acceleration, t):
    max_dim = max(shape) - max(shape) % 2
    min_dim = min(shape) - min(shape) % 2
    
    M = int(shape[0] * shape[1] / (acceleration * (max_dim / 2 - (max_dim - min_dim) * (1 + min_dim / max_dim) / 4)))

    num_nested_squares = max_dim // 2
    
    mask = np.zeros((max_dim, max_dim), dtype=np.float32)

    for square_id in range(num_nested_squares):
        ordered_indices = get_square_ordered_idxs(
            square_side_size=max_dim,
            square_id=square_id,
        )
        # J: size of the square, J=2,…,N, i.e., the number of points along one side of the square
        J = 2 * (num_nested_squares - square_id)
        # K: total number of points along the perimeter of the square K=4·J-4;
        K = 4 * (J - 1)

        for m in range(M):
            indices_idx = int(np.floor(np.mod((m + t * M) / GOLDEN_RATIO, 1) * K))
            mask[ordered_indices[indices_idx]] = 1.0

    return mask


class BaseMaskFunc:
    """BaseMaskFunc is the base class to create a sub-sampling mask of a given shape."""

    def __init__(
        self,
        accelerations: Union[List[Number], Tuple[Number, ...]],
        center_fractions: Optional[Union[List[float], Tuple[float, ...]]] = None,
        uniform_range: bool = True,
    ):
        """
        Parameters
        ----------
        accelerations: Union[List[Number], Tuple[Number, ...]]
            Amount of under-sampling_mask. An acceleration of 4 retains 25% of the k-space, the method is given by
            mask_type. Has to be the same length as center_fractions if uniform_range is not True.
        center_fractions: Optional[Union[List[float], Tuple[float, ...]]]
            Fraction of low-frequency columns to be retained.
            If multiple values are provided, then one of these numbers is chosen uniformly each time. If uniform_range
            is True, then two values should be given. Default: None.
        uniform_range: bool
            If True then an acceleration will be uniformly sampled between the two values. Default: True.
        """
        if center_fractions is not None:
            if len([center_fractions]) != len([accelerations]):
                raise ValueError(
                    f"Number of center fractions should match number of accelerations. "
                    f"Got {len([center_fractions])} {len([accelerations])}."
                )

        self.center_fractions = center_fractions
        self.accelerations = accelerations

        self.uniform_range = uniform_range

        self.rng = np.random.RandomState()

    def choose_acceleration(self):
        if not self.accelerations:
            return None

        if not self.uniform_range:
            choice = self.rng.randint(0, len(self.accelerations))
            acceleration = self.accelerations[choice]
            if self.center_fractions is None:
                return acceleration

            center_fraction = self.center_fractions[choice]
            return center_fraction, acceleration
        raise NotImplementedError("Uniform range is not yet implemented.")

    @abstractmethod
    def mask_func(self, *args, **kwargs):
        raise NotImplementedError("This method should be implemented by a child class.")

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        """Produces a sampling mask by calling class method :meth:`mask_func`.

        Parameters
        ----------
        *args
        **kwargs

        Returns
        -------
        mask: torch.Tensor
            Sampling mask.
        """
        mask = self.mask_func(*args, **kwargs)
        return mask


class CIRCUSSamplingMode(str, Enum):
    circus_radial = "circus-radial"
    circus_spiral = "circus-spiral"

class CIRCUSMaskFunc(BaseMaskFunc):
    """Implementation of Cartesian undersampling (radial or spiral) using CIRCUS as shown in [1]_. It creates radial or
    spiral masks for Cartesian acquired data on a grid.

    References
    ----------

    .. [1] Liu J, Saloner D. Accelerated MRI with CIRcular Cartesian UnderSampling (CIRCUS): a variable density
        Cartesian sampling strategy for compressed sensing and parallel imaging. Quant Imaging Med Surg.
        2014 Feb;4(1):57-67. doi: 10.3978/j.issn.2223-4292.2014.02.01. PMID: 24649436; PMCID: PMC3947985.
    """

    def __init__(
        self,
        accelerations: Union[List[Number], Tuple[Number, ...]],
        subsampling_scheme: CIRCUSSamplingMode,
        **kwargs,
    ):
        super().__init__(
            accelerations=accelerations,
            center_fractions=tuple(0 for _ in range(len(accelerations))),
            uniform_range=False,
        )
        if subsampling_scheme not in ["circus-spiral", "circus-radial"]:
            raise NotImplementedError(
                f"Currently CIRCUSMaskFunc is only implemented for 'circus-radial' or 'circus-spiral' "
                f"as a subsampling_scheme. Got subsampling_scheme={subsampling_scheme}."
            )

        self.subsampling_scheme = "circus-radial" if subsampling_scheme is None else subsampling_scheme


    def circus_radial_mask(self, shape, acceleration):
        """Implements CIRCUS radial undersampling."""
        t = self.rng.randint(low=0, high=1e4, size=1, dtype=int).item()
        
        mask = accelerated_loop_radial(shape, acceleration, t)

        pad = ((shape[0] % 2, 0), (shape[1] % 2, 0))

        mask = np.pad(mask, pad, constant_values=0)
        mask = center_crop(torch.from_numpy(mask.astype(bool)), shape)

        return mask

    def circus_spiral_mask(self, shape, acceleration):
        """Implements CIRCUS spiral undersampling."""

        c = self.rng.uniform(low=1.1, high=1.3, size=1).item()

        mask = accelerated_loop_spiral(shape, acceleration, c)

        pad = ((shape[0] % 2, 0), (shape[1] % 2, 0))

        mask = np.pad(mask, pad)
        mask = center_crop(torch.from_numpy(mask.astype(bool)), shape)

        return mask

    @staticmethod
    def circular_centered_mask(mask, eps=0.1):
        shape = mask.shape
        center = np.asarray(shape) // 2
        Y, X = np.ogrid[: shape[0], : shape[1]]
        Y, X = torch.tensor(Y), torch.tensor(X)
        radius = 1

        # Finds the maximum (unmasked) disk in mask given a tolerance.
        while True:
            # Creates a disk with R=radius and finds intersection with mask
            disk = (Y - center[0]) ** 2 + (X - center[1]) ** 2 <= radius**2
            intersection = disk & mask
            ratio = disk.sum() / intersection.sum()
            if ratio > 1.0 + eps:
                return intersection
            radius += eps

    def mask_func(
        self,
        shape: Union[List[int], Tuple[int, ...]],
        return_acs: bool = False,
        seed: Optional[Union[int, Iterable[int]]] = None,
    ) -> torch.Tensor:
        """Produces :class:`CIRCUSMaskFunc` sampling masks.

        Parameters
        ----------
        shape: list or tuple of ints
            The shape of the mask to be created. The shape should at least 3 dimensions.
            Samples are drawn along the second last dimension.
        return_acs: bool
            Return the autocalibration signal region as a mask.
        seed: int or iterable of ints or None (optional)
            Seed for the random number generator. Setting the seed ensures the same mask is generated
             each time for the same shape. Default: None.

        Returns
        -------
        mask: torch.Tensor
            The sampling mask.
        """

        if len(shape) < 3:
            raise ValueError("Shape should have 3 or more dimensions")

        with temp_seed(self.rng, seed):
            num_rows = shape[-3]
            num_cols = shape[-2]
            acceleration = self.choose_acceleration()[1]

            if self.subsampling_scheme == "circus-radial":
                mask = self.circus_radial_mask(
                    shape=(num_rows, num_cols),
                    acceleration=acceleration,
                )
            elif self.subsampling_scheme == "circus-spiral":
                mask = self.circus_spiral_mask(
                    shape=(num_rows, num_cols),
                    acceleration=acceleration,
                )

            if return_acs:
                return self.circular_centered_mask(mask).unsqueeze(0).unsqueeze(-1)

            return mask.unsqueeze(0).unsqueeze(-1)


class RadialMaskFunc(CIRCUSMaskFunc):
    """Computes radial masks for Cartesian data."""

    def __init__(
        self,
        accelerations: Union[List[Number], Tuple[Number, ...]],
        **kwargs,
    ):
        super().__init__(
            accelerations=accelerations,
            subsampling_scheme=CIRCUSSamplingMode.circus_radial,
            **kwargs,
        )


class SpiralMaskFunc(CIRCUSMaskFunc):
    """Computes spiral masks for Cartesian data."""

    def __init__(
        self,
        accelerations: Union[List[Number], Tuple[Number, ...]],
        **kwargs,
    ):
        super().__init__(
            accelerations=accelerations,
            subsampling_scheme=CIRCUSSamplingMode.circus_spiral,
            **kwargs,
        )

