import numpy as np
from .sm_base import SMBase
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
import pickle
import rtgym.utils as utils


class BoundaryVecCell(SMBase):
    """
    Spatially modulated behavior with border vector responses
    @attr sm_responses: spatially modulated responses, shape (n_cells, arena_height, arena_width)
    """
    sens_type = 'boundary_vec_cell'
    def __init__(self, arena, **kwargs):
        """
        @kwarg n_cells: number of spatially modulated non-grid/place cellcells
        @kwarg r_max: maximum distance of the receptive field (radial extent)
        @kwarg angle_width: width of the angle in radians
        @kwarg normalize: normalize all responses to the same range ([0, 1] before scaled by magnitude)
        @kwarg magnitude: maximum magnitude of the cell responses
        """
        super().__init__(arena, **kwargs)
        self.r_max = kwargs.get('r_max', 10)
        self.angle_width = kwargs.get('angle_width', np.pi/6)
        self.normalize = kwargs.get('normalize', True)
        self.magnitude = kwargs.get('magnitude', 10)
        
        # check parameters and initialize responses
        self._check_params()
        self._init_sm_responses()

    def _check_params(self):
        """ Check parameters """
        assert self.n_cells > 0, "num of cells cannot be less than 0"
        assert self.magnitude > 0, "magnitude cannot be less than 0"
        assert self.r_max > 0, "r_max cannot be less than 0"
        # assert self.r_max * 2 <= self.arena.arena_width, "r_max is too large for the arena"
        # assert self.r_max * 2 <= self.arena.arena_height, "r_max is too large for the arena"
        assert self.angle_width > 0, "angle_width cannot be less than 0"

    def _init_sm_responses(self):
        """ Initialize sm_responses """
        self._generate_receptive_fields()
        self._compute_res()

    def _generate_receptive_fields(self):
        """ Generate receptive fields for all cells """
        n_bins = self.r_max * 2 // self.arena.spatial_resolution
        thetas = np.linspace(0, 2*np.pi, self.n_cells, endpoint=False)
        receptive_fields = np.zeros((self.n_cells, n_bins, n_bins))
        for i in range(self.n_cells):
            receptive_fields[i] = self._generate_receptive_field(thetas[i], n_bins)
        self.receptive_fields = receptive_fields * self.magnitude

    def _generate_receptive_field(self, theta_c, n_bins):
        """
        Generate a receptive field for a single cell
        
        Parameters:
            - theta_c: center angle of the receptive field
            - n_bins: number of bins in the receptive field
        
        Returns:
            - receptive field of the cell
        """
        x = np.linspace(-self.r_max, self.r_max, n_bins)
        y = np.linspace(-self.r_max, self.r_max, n_bins)
        x, y = np.meshgrid(x, y)

        # Convert to polar coordinates relative to the center
        r = np.sqrt(x**2 + y**2)  # Each point's distance from the center
        theta = np.arctan2(y, x)  # Each point's angle relative to the center

        # Radial Gaussian decay
        radial_decay = np.exp(-r**2 / (2 * (self.r_max*0.7)**2))

        # Angular cosine taper
        half_angle_width = self.angle_width / 2
        theta_min, theta_max = theta_c - half_angle_width, theta_c + half_angle_width

        delta_theta = (theta_max - theta_min) / 2  # Angular half-width
        angular_taper = np.maximum(0, np.cos((theta - theta_c) * np.pi / (2 * delta_theta)))

        # Mask the angular region outside the slice, considering wrapping around -pi to pi
        angular_mask = ((theta >= theta_min) & (theta <= theta_max)) | \
                    ((theta + 2 * np.pi >= theta_min) & (theta + 2 * np.pi <= theta_max)) | \
                    ((theta - 2 * np.pi >= theta_min) & (theta - 2 * np.pi <= theta_max))

        angular_weight = angular_taper * angular_mask

        # Combine radial and angular components
        receptive_field = radial_decay * angular_weight

        return receptive_field

    def _compute_res(self):
        """ 
        Generate spatially modulated cells
        """
        arena_dims = self.arena.dimensions
        cells = np.zeros((self.n_cells, *arena_dims))
        for i in range(self.n_cells):
            # Convolve the receptive field (kernel) with the arena map
            cell = convolve(self.arena.arena_map, self.receptive_fields[i], mode='constant', cval=1)
            if self.normalize:
                cell = (cell - cell.min()) / (cell.max() - cell.min())
            cells[i] = cell
        self.sm_responses = cells * self.magnitude

    def plot_recetive_field(self):
        """ Plot the receptive field of a cell """
        return utils.visualize_fields(self.receptive_fields)

    def get_specs(self):
        specs = super().get_specs()
        specs['cell_max_avg'] = self.sm_responses.max(axis=(1, 2)).mean()
        specs['cell_min_avg'] = self.sm_responses.min(axis=(1, 2)).mean()
        specs['cell_mean_avg'] = self.sm_responses.mean(axis=(1, 2)).mean()
        return specs

    def state_dict(self):
        return {
            'sens_type': self.sens_type,
            'n_cells': self.n_cells,
            'r_max': self.r_max,
            'angle_width': self.angle_width,
            'normalize': self.normalize,
            'magnitude': self.magnitude,
            'receptive_fields': self.receptive_fields
        }

    @classmethod
    def load_from_dict(cls, state_dict, arena):
        """
        Load the object from a dictionary and reconstruct it.
        
        Parameters:
            - state_dict: Dictionary containing the object's state.
            - arena: Arena object to reinitialize the class.
            
        Returns:
            - Reconstructed BoundaryVecCell object.
        """
        obj = cls(
            arena,
            n_cells=state_dict['n_cells'],
            r_max=state_dict['r_max'],
            angle_width=state_dict['angle_width'],
            normalize=state_dict['normalize'],
            magnitude=state_dict['magnitude']
        )
        
        obj.receptive_fields = state_dict['receptive_fields']
        obj._compute_res()
        
        return obj
