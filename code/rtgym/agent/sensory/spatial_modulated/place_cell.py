import numpy as np
from .sm_base import SMBase
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt


class PlaceCell(SMBase):
    """
    Place cell responses
    """
    sens_type = 'place_cell'
    def __init__(self, arena, **kwargs):
        """
        @kwarg n_cells: number of cells
        @kwarg sigma: sigma of the gaussian filter(s) (cm)
        @kwarg ssigma: standard deviation of the sigma (cm)
        @kwarg dg_ratio: difference of gaussian ratio, default is 1, i.e. no diff of gaussian.
                         dg_ratio must be greater than or equal to 1
        @kwarg magnitude: maximum magnitude of the cell responses

        @attr sm_responses: spatially modulated responses, shape (n_cells, *arena_dimensions)
        """
        super().__init__(arena, **kwargs)
        
        # parameters
        self.sigma = kwargs.get('sigma', 8)
        self.ssigma = kwargs.get('ssigma', 0)
        self.dg_ratio = kwargs.get('dg_ratio', 1)
        self.magnitude = kwargs.get('magnitude', None)
        self.normalize = kwargs.get('normalize', False)
        
        # check parameters and initialize responses
        self._check_params()
        self._init_params()
        self._init_sm_responses()

    def _check_params(self):
        """ Check parameters """
        assert self.n_cells > 0, "n_cells <= 0"
        # sigma can be a list or an integer
        if isinstance(self.sigma, int):
            assert self.sigma > 0, "sigma <= 0"
        elif isinstance(self.sigma, list):
            for s in self.sigma:
                assert s > 0, "sigma <= 0"

    def _init_params(self):
        if isinstance(self.sigma, int):
            self.sigma = int(self.sigma/self.arena.spatial_resolution)
            self.ssimga = int(self.ssigma/self.arena.spatial_resolution)
        elif isinstance(self.sigma, list):
            self.sigma = [int(s/self.arena.spatial_resolution) for s in self.sigma]
            if self.ssigma != 0:
                print("Warning: ssigma is not used when sigma is a list")

    def _init_sm_responses(self):
        """
        Generate a spatially modulated non-grid/place cell response field 
        """
        super()._init_sm_responses()

        arena_dims = self.arena.dimensions
        arena_free_space = self.arena.free_space
        place_centers = arena_free_space[self.rng.choice(arena_free_space.shape[0], size=self.n_cells, replace=False)]
        cells = np.zeros((self.n_cells, *arena_dims))
        # generate a list of sigmas, with mean = sigma and std = ssigma
        sigma_list = self.rng.normal(self.sigma, self.ssigma, self.n_cells) if self.ssigma != 0 else [self.sigma] * self.n_cells
        
        # filter each cell response field with a 2d gaussian filter
        for i in range(self.n_cells):
            cells[i, place_centers[i][0], place_centers[i][1]] = 1
            cells[i] = gaussian_filter(cells[i], sigma_list[i])
            if self.dg_ratio > 1:
                cells[i] -= gaussian_filter(cells[i], sigma_list[i]*self.dg_ratio)

        # min-max normalize the cells to [0, 1]
        if self.normalize:
            if self.dg_ratio > 1:
                cells = cells * self.magnitude if self.magnitude is not None else cells
                cells = (cells - cells.mean(axis=(1, 2), keepdims=True))
            else:
                # Normalize to 0-1
                cells = (cells - cells.min()) / (cells.max() - cells.min())
                cells = cells * self.magnitude if self.magnitude is not None else cells
        self.sm_responses = cells

    def get_specs(self):
        specs = super().get_specs()
        specs['cell_max_avg'] = self.sm_responses.max(axis=(1, 2)).mean()
        specs['cell_min_avg'] = self.sm_responses.min(axis=(1, 2)).mean()
        specs['cell_mean_avg'] = self.sm_responses.mean(axis=(1, 2)).mean()
        return specs
