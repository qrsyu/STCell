import rtgym
import numpy as np
from .sm_base import SMBase
from .weak_sm_cell import WeakSMCell
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt


class WeakHDSMCell(SMBase):
    """ Head direction conditioned weak spatially modulated cells """
    sens_type = 'weak_hd_sm_cell'
    def __init__(self, arena, **kwargs):
        """
        @kwarg n_cells: number of spatially modulated non-grid/place cellcells
        @kwarg sigma: sigma of the gaussian filter(s) (cm)
        @kwarg magnitude: maximum magnitude of the cell responses

        @attr sm_responses: spatially modulated responses, shape (n_cells, *arena_dimensions)
        """
        super().__init__(arena, **kwargs)
        
        # parameters
        self.n_directions = kwargs.get("n_directions", 4)
        self.sm_list = [WeakSMCell(arena=self.arena, **kwargs) for _ in range(self.n_directions)]

    def get_responses(self, traj: rtgym.trial.Trajectory):
        """
        Return the interpolated responses based on head direction.
        """
        coords, coords_f, hds = traj.coords, traj.coords.f, traj.hds
        
        # Normalize and scale head directions to [0, n_directions)
        norm_hds = (hds + np.pi) % (2 * np.pi)
        scaled_hds = norm_hds / (2 * np.pi) * self.n_directions

        # Calculate indices for interpolation
        lower_indices = np.floor(scaled_hds).astype(int)  # Shape: (n_batch, n_timesteps)
        upper_indices = (lower_indices + 1) % self.n_directions
        lower_weights = 1 - (scaled_hds - lower_indices)  # Shape: (n_batch, n_timesteps)

        # Prepare responses array
        responses = np.array([sm_.get_responses(coords, coords_f, hds) for sm_ in self.sm_list])  # Shape: (n_directions, n_batch, n_timesteps, n_dim)
        n_batch, n_timesteps, n_dim = responses[0].shape
        responses_reshaped = responses.reshape(self.n_directions, -1, n_dim)  # Shape: (n_directions, n_batch * n_timesteps, n_dim)

        # Flatten indices for direct indexing
        flat_lower_indices = lower_indices.ravel()  # Shape: (n_batch * n_timesteps,)
        flat_upper_indices = upper_indices.ravel()  # Shape: (n_batch * n_timesteps,)
        flat_lower_weights = lower_weights.ravel()  # Shape: (n_batch * n_timesteps,)

        # Get lower and upper responses for interpolation
        flat_lower_responses = responses_reshaped[flat_lower_indices, np.arange(flat_lower_indices.size)]
        flat_upper_responses = responses_reshaped[flat_upper_indices, np.arange(flat_upper_indices.size)]

        # Interpolate between lower and upper responses
        flat_interpolated_responses = (flat_lower_weights[:, None] * flat_lower_responses +
                                    (1 - flat_lower_weights[:, None]) * flat_upper_responses)  # Shape: (n_batch * n_timesteps, n_dim)

        # Reshape to original batch and timestep dimensions
        interpolated_responses = flat_interpolated_responses.reshape(n_batch, n_timesteps, n_dim)
        return interpolated_responses

    def vis(self, N=10, cmap='jet', *args, **kwargs):
        """
        Visualize the spatially modulated cells
        """
        N_shortend = min(5, N//2)
        return [_sm.vis(N_shortend, cmap, *args, **kwargs) for _sm in self.sm_list]

    def get_specs(self):
        specs = super().get_specs()
        specs['n_directions'] = self.n_directions
        specs['sm_list'] = [sm_.get_specs() for sm_ in self.sm_list]
