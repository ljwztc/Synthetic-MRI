from typing import Sequence, Union

import torch

from common import complex as cplx
import numpy as np
import bart


class NoiseModel:
    """A model that adds inhomogeneous noise.

    Attributes:
        std_devs (Tuple[float]): The range of standard deviations used for
            noise model. If a single value is provided on initialization,
            it is stored as a tuple with 1 element.
        warmup_method (str): The method that is being used for warmup.
        warmup_iters (int): Number of iterations to use for warmup.
        generator (torch.Generator): The generator that should be used for all
            random logic in this class.

    Note:
        This class is functionally deprecated and will not be maintained.
        Use :cls:`RandomNoise` instead.

    Note:
        We do not store this as a module or else it would be saved to the model
        definition, which we dont want.

    Note:
        There is a known bug that the warmup method does not clamp the upper
        bound at the appropriate value. Thus the upper bound of the range keeps
        growing. We have elected not to correct for this to preserve
        reproducibility for older results. To use schedulers with corrected
        functionality, see :cls:`RandomNoise` instead.
    """

    def __init__(
        self,
        std_devs: Union[float, Sequence[float]],
        scheduler=None,
        mask=None,
        seed=None,
        device=None,
    ):
        """
        Args:
            std_devs (float, Tuple[float, float]): The  noise difficulty range.
            scheduler (CfgNode, optional): Config detailing scheduler.
            mask (CfgNode, optional): Config for masking method. Should have
                an attribute ``RHO`` that specifies the extent of masking.
            seed (int, optional): A seed for reproducibility.
        """
        if not isinstance(std_devs, Sequence):
            std_devs = (std_devs,)
        elif len(std_devs) > 2:
            raise ValueError("`std_devs` must have 2 or fewer values")
        self.std_devs = std_devs

        self.warmup_method = None
        self.warmup_iters = 0
        if scheduler is not None:
            self.warmup_method = scheduler.WARMUP_METHOD
            self.warmup_iters = scheduler.WARMUP_ITERS

        # Amount of the kspace to augment with noise.
        self.rho = None
        if mask is not None:
            self.rho = mask.RHO

        # For reproducibility.
        g = torch.Generator(device=device)
        if seed:
            g = g.manual_seed(seed)
        self.generator = g

    def choose_std_dev(self):
        """Chooses a random acceleration rate given a range."""
        if not isinstance(self.std_devs, Sequence):
            return self.std_devs
        elif len(self.std_devs) == 1:
            return self.std_devs[0]


        std_range = self.std_devs[1] - self.std_devs[0]

        g = self.generator
        std_dev = self.std_devs[0] + std_range * torch.rand(1, generator=g, device=g.device).item()
        return std_dev

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, kspace, seed=None, clone=True) -> torch.Tensor:
        """Performs augmentation on undersampled kspace mask."""
        if clone:
            kspace = kspace.clone()
        mask = cplx.get_mask(kspace)
        # print('mask information', mask.mean(), mask.var())

        g = (
            self.generator
            if seed is None
            else torch.Generator(device=kspace.device).manual_seed(seed)
        )
        # noise_std = self.choose_std_dev()
        
        sensi_kspace = torch.view_as_complex(kspace)
        sensi_kspace = sensi_kspace.permute(1, 2, 3, 0).numpy()
        sensi_map = np.zeros(sensi_kspace.shape)
        for index, slice in enumerate(sensi_kspace):
            sens_map = bart.bart(1, "ecalib -d0 -m1", slice)
            sens_map = 1 / (np.abs(sens_map)+ 1e-6)
            sensi_map[index] = sens_map

        sensi_map = torch.tensor(sensi_map)
        noise_std = torch.sqrt(sensi_map)
        noise_std = noise_std.permute(3,0,1,2).unsqueeze(4)

        noise = kspace.mean().abs() * noise_std * torch.randn(kspace.shape, generator=g, device=kspace.device)

        masked_noise = noise * mask
        aug_kspace = kspace + masked_noise

        return aug_kspace
