import numpy as np
from .sm_base import SMBase
from scipy.signal import fftconvolve


class BoundaryCell(SMBase):
    """
    Boundary cells (this is potentially not realistic), the responses are the distance to the boundary of all directions
    """
    sens_type = 'boundary_cell'
    def __init__(self, arena, **kwargs):
        """
        @kwarg n_cells: number of spatially modulated non-grid/place cells
        @kwarg res_dist: distance of the cell responses from the boundary (cm)
        @kwarg magnitude: maximum magnitude of the cell responses
        @kwarg normalize: normalize the cell responses
        @kwarg center_border_ratio: ratio of center cells to border cells (i.e., gradient from center to border or vice versa)
        """
        super(BoundaryCell, self).__init__(arena, **kwargs)
        
        # parameters
        self.res_dist = kwargs.get('res_dist', 10) / self.arena.spatial_resolution
        self.magnitude = kwargs.get('magnitude', None)
        self.normalize = kwargs.get('normalize', False)
        self.center_border_ratio = kwargs.get('center_border_ratio', 0.5)
        
        # check parameters and initialize responses
        self._check_params()
        self._init_sm_responses()

    def _check_params(self):
        """ Check parameters """
        assert self.n_cells > 0, "n_cells <= 0"
        # check res_dist
        assert self.res_dist > 0, "res_dist <= 0"
        assert self.res_dist < self.arena.dimensions[1], "res_dist >= arena.dimensions[1]"
        assert self.res_dist < self.arena.dimensions[0], "res_dist >= arena.dimensions[0]"
    
    def _init_sm_responses(self):
        """ Initialize sm_responses """
        super(BoundaryCell, self)._init_sm_responses()
        
        # initialize sm_responses
        cell_res_dist = self.rng.normal(self.res_dist, self.res_dist/4, self.n_cells).astype(int)
        cell_res_dist = np.clip(cell_res_dist, 0, None)
        for i in range(self.n_cells):
            kernal = np.ones((cell_res_dist[i]*2, cell_res_dist[i]*2))
            self.sm_responses[i, :, :] = -fftconvolve(self.arena.inv_arena_map, kernal, mode='same')/kernal.sum()
            self.sm_responses[i, :, :] *= self.rng.choice([-1, 1], 1, p=[self.center_border_ratio, 1-self.center_border_ratio])
            self.sm_responses[i, :, :] -= self.sm_responses[i, :, :].min()
        
        # normalize sm_responses
        if self.normalize:
            self.sm_responses = (self.sm_responses - self.sm_responses.min()) / (self.sm_responses.max() - self.sm_responses.min())
        
        # set magnitude
        if self.magnitude is not None:
            self.sm_responses = self.magnitude * self.sm_responses

    def get_specs(self):
        specs = super().get_specs()
        specs['cell_max_avg'] = self.sm_responses.max(axis=(1, 2)).mean()
        specs['cell_min_avg'] = self.sm_responses.min(axis=(1, 2)).mean()
        specs['cell_mean_avg'] = self.sm_responses.mean(axis=(1, 2)).mean()
        return specs
