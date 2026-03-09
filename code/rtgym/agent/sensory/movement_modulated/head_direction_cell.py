import rtgym
import numpy as np
from .displacement_abs import DisplacementAbs
from scipy.ndimage import gaussian_filter1d


class HeadDirectionCell(DisplacementAbs):
    """
    Direction cell
    """
    sens_type = 'head_direction_cell'
    def __init__(self, arena, **kwargs):
        """
        @kwarg n_cells: number of spatially modulated non-grid/place cellcells
        @kwarg magnitude: maximum magnitude of the cell responses
        @kwarg normalize: normalize the cell responses
        @kwarg n_bins: number of bins for the direction
        @kwarg sigma: standard deviation of the Gaussian kernel
        """
        super().__init__(arena, **kwargs)

        # parameters
        self.sigma = kwargs.get('sigma', 2)
        self.n_bins = kwargs.get('n_bins', 360)
        self.sorted = kwargs.get('sorted', True)
        self.normalize = kwargs.get('normalize', False)
        self.magnitude = kwargs.get('magnitude', 1)

        # check parameters
        self._check_params()
        self._init_mm_responses()

    def _check_params(self):
        """ Check parameters """
        assert self.n_cells > 0, "n_cells <= 0"

    def _init_mm_responses(self):
        """
        Initialize the movement modulated responses with wrapped Gaussian distributions.
        """
        # Define the range of directions
        self.dirs = self.rng.uniform(-np.pi, np.pi, self.n_bins)  # Shape: (n_bins,)
        
        # Assign random preferred directions to each cell
        self.hd_dirs = self.rng.uniform(-np.pi, np.pi, self.n_cells)  # Shape: (n_cells,)
        
        if self.sorted:
            self.hd_dirs = np.sort(self.hd_dirs)
        
        # Compute the angular distance between each cell's preferred direction and each bin
        angular_dist = np.mod(self.hd_dirs[:, np.newaxis] - self.dirs[np.newaxis, :] + np.pi, 2 * np.pi) - np.pi
        
        # Compute the Gaussian response based on angular distance
        self.mm_responses = np.exp(-0.5 * (angular_dist / self.sigma) ** 2)  # Shape: (n_cells, n_bins)
        
        if self.normalize:
            # zero-one normalization
            self.mm_responses = (self.mm_responses - np.min(self.mm_responses)) / (np.max(self.mm_responses) - np.min(self.mm_responses))

        self.mm_responses *= self.magnitude


    def get_responses(self, traj: rtgym.trial.Trajectory):
        """
        Get the movement modulated responses for the given trajectory.
        
        @param traj: trajectory
        @return: movement modulated responses of shape (n_batches, n_steps, n_cells)
        """
        # Get the head direction from the trajectory
        hd = traj.hds  # Expected shape: (n_batches, n_steps, 1)
        
        # Remove the last dimension if it's singleton
        if hd.shape[-1] == 1:
            hd = hd.squeeze(-1)  # Now shape: (n_batches, n_steps)
        
        # Compute the closest direction index for each head direction
        # Resulting shape: (n_batches, n_steps)
        closest_indices = np.argmin(np.abs(hd[:, :, np.newaxis] - self.dirs[np.newaxis, np.newaxis, :]), axis=-1)
        
        # Use np.take to gather responses from mm_responses
        # self.mm_responses shape: (n_cells, n_bins)
        # We take along axis=1 (bins) for the closest_indices
        # Resulting shape after np.take: (n_cells, n_batches, n_steps)
        gathered_responses = np.take(self.mm_responses, closest_indices, axis=1)
        
        # Transpose to get shape: (n_batches, n_steps, n_cells)
        responses = gathered_responses.transpose(1, 2, 0)
        
        return responses


    def get_specs(self):
        specs = super().get_specs()
        specs['magnitude'] = self.magnitude
        specs['normalize'] = self.normalize
        return specs

    def vis(self, N, **kwargs):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.imshow(self.mm_responses[:N], aspect='auto', cmap='jet')
        ax.set_xticks([0, self.n_bins//4, self.n_bins//2, 3*self.n_bins//4, self.n_bins-1])
        ax.set_xticklabels(['-π', '-π/2', '0', 'π/2', 'π (wrap around)'])
        ax.set_xlabel('Direction (radian)')
        ax.set_ylabel('Cells')
        ax.set_title('Head Direction Cells')
        plt.show()