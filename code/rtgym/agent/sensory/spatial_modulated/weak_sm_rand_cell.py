import numpy as np
from .weak_sm_cell import WeakSMCell
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d


class WeakSMRandCell(WeakSMCell):
    """ Spatially modulated non-grid/place-like responses """
    sens_type = 'weak_sm_rand_cell'
    def __init__(self, arena, **kwargs):
        """
        @kwarg n_cells: number of spatially modulated non-grid/place cellcells
        @kwarg sigma: sigma of the gaussian filter(s) (cm)
        @kwarg magnitude: maximum magnitude of the cell responses

        @attr sm_responses: spatially modulated responses, shape (n_cells, arena_height, arena_width)
        """
        super(WeakSMRandCell, self).__init__(arena, **kwargs)
        self.variation = kwargs.get('variation', 1)

    def _generate_smcells(self, n, sig):
        """ 
        Generate a spatially modulated non-grid/place cell response field 
        @param n: number of cells
        @param sig: sigma of the gaussian filter (pixels)
        """
        arena_dims = self.arena.dimensions
        cells = self.rng.normal(0, 1, (n, *arena_dims))
        sigma_list = self.rng.normal(sig, self.variation, n)
        
        # filter each cell response field with a 2d gaussian filter
        for i in range(n):
            sig_ = sigma_list[i]
            cells[i] = gaussian_filter(cells[i], sig_)
        
        # normalize the cells
        cells = cells - np.min(cells)
        if self.normalize:
            cells = cells / np.max(cells)
        return cells * self.magnitude if self.magnitude is not None else cells

    def get_specs(self):
        specs = super().get_specs()
        specs['variation'] = self.variation
        return specs
