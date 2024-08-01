from typing import Sequence, Union

import numpy as np
import torch

class MotionModel:
    """A model that corrupts kspace inputs with motion.

    Motion is a common artifact experienced during the MR imaging forward problem.
    When a patient moves, the recorded (expected) location of the kspace sample is
    different than the actual location where the kspace sample that was acquired.
    This module is responsible for simulating different motion artifacts.

    Attributes:
        motion_type (str): The typr of motion [rigid,  respriration]
        motion_range (Tuple[float]): The range of motion difficulty.
            If a single value is provided on initialization, it is stored
            as a tuple with 1 element.
        generator (torch.Generator): The generator that should be used for all
            random logic in this class.
    """

    def __init__(self, motion_range: Union[float, Sequence[float]], seed=None):
        """
        Args:
            motion_range (float, Tuple[float, float]): The range of motion difficulty.
            seed (int, optional): A seed for reproducibility.
        """
        self.motion_range = motion_range

        g = torch.Generator()
        if seed:
            g = g.manual_seed(seed)
        self.generator = g

    def choose_motion_range(self):
        motion_range = self.motion_range[1] - self.motion_range[0]

        g = self.generator
        motion_range = (
            self.motion_range[0] + motion_range * torch.rand(1, generator=g, device=g.device).item()
        )
        return motion_range

    def __call__(self, kspace, blurriness_type, seed=None, clone=True) -> torch.Tensor:
        """Performs motion corruption on kspace image.
        Args:
            kspace (torch.Tensor): The complex tensor. Shape ``(N, Y, X, #coils, [2])``.
            seed (int, optional): Fixed seed at runtime (useful for generating testing vals).
            clone (bool, optional): If ``True``, return a cloned tensor.

        Returns:
            torch.Tensor: The motion corrupted kspace.
        """
        # is_complex = False
        if clone:
            kspace = kspace.clone()
        phase_matrix = torch.zeros(kspace.shape, dtype=torch.complex64)
        width = kspace.shape[2]
        # width = kspace.shape[3]
        g = self.generator if seed is None else torch.Generator().manual_seed(seed)
        scale = self.choose_motion_range()


        odd_err = scale * torch.rand(1).numpy()
        even_err = scale * torch.rand(1).numpy()

        line_0 = width * torch.rand(1, generator=g)
        if blurriness_type == 'rigid':
            for line in range(width):
                if line % 2 == 0:
                    rand_err = even_err
                else:
                    rand_err = odd_err
                phase_error = torch.from_numpy(np.exp(-1j * rand_err))
                phase_matrix[:, :, line] = phase_error
        elif blurriness_type == 'respiration':
            for line in range(width):
                if line % 2 == 0:
                    rand_err = scale * np.sin(2 * np.pi * even_err * np.abs(line - width/2) + scale)
                else:
                    rand_err = scale * np.sin(2 * np.pi * odd_err * np.abs(line - width/2) + scale)
                phase_error = torch.from_numpy(np.exp(-1j * rand_err))
                phase_matrix[:, :, line] = phase_error

        
        aug_kspace = kspace * phase_matrix
        return aug_kspace