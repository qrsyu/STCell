import numpy as np
from .sm_base import SMBase
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt


class WeakSMCell(SMBase):
    """ Spatially modulated non-grid/place responses """
    sens_type = 'weak_sm_cell'
    def __init__(self, arena, **kwargs):
        """
        @kwarg n_cells: number of spatially modulated non-grid/place cellcells
        @kwarg sigma: sigma of the gaussian filter(s) (cm)
        @kwarg magnitude: maximum magnitude of the cell responses

        @attr sm_responses: spatially modulated responses, shape (n_cells, *arena_dimensions)
        """
        super().__init__(arena, **kwargs)
        
        # parameters
        self.sigma = kwargs.get('sigma', 8)
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
        elif isinstance(self.sigma, list):
            self.sigma = [int(s/self.arena.spatial_resolution) for s in self.sigma]

    def _init_sm_responses(self):
        """ Initialize sm_responses """
        super()._init_sm_responses()
        # border padding is also included in the arena map
        self.sm_responses = self._generate_smcells(self.n_cells, self.sigma)

    def _generate_smcells(self, n, sig):
        """ 
        Generate a spatially modulated non-grid/place cell response field 
        @param n: number of cells
        @param sig: sigma of the gaussian filter (pixels)
        """
        arena_dims = self.arena.dimensions
        cells = self.rng.normal(0, 1, (n, *arena_dims))
        # filter each cell response field with a 2d gaussian filter
        for i in range(n):
            cell = gaussian_filter(cells[i], sig, mode='constant')
            if self.normalize:
                cell = (cell - cell.min()) / (cell.max() - cell.min())
            cells[i] = cell
        return cells * self.magnitude if self.magnitude is not None else cells # (n, *arena_dims)

    def get_specs(self):
        specs = super().get_specs()
        specs['cell_max_avg'] = self.sm_responses.max(axis=(1, 2)).mean()
        specs['cell_min_avg'] = self.sm_responses.min(axis=(1, 2)).mean()
        specs['cell_mean_avg'] = self.sm_responses.mean(axis=(1, 2)).mean()
        return specs
