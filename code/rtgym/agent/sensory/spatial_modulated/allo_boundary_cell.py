import numpy as np
from .sm_base import SMBase
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt


class AlloBoundaryCell(SMBase):
    """
    Spatially modulated non-grid/place-like responses
    """
    def __init__(self, arena, **kwargs):
        """
        @kwarg n_cells: number of spatially modulated non-grid/place cells
        @kwarg res_dist: distance of the cell responses from the boundary (cm)
        @kwarg magnitude: maximum magnitude of the cell responses
        @kwarg normalize: normalize the cell responses
        @kwarg direction_biases: direction biases of the cell responses, needs 4 values
                    must also sum to 1, in the order of [top-down, left-right, bottom-up, right-left]
        """
        super(AlloBoundaryCell, self).__init__(arena, **kwargs)
        
        # parameters
        self.res_dist = kwargs.get('res_dist', 10) / self.arena.spatial_resolution
        self.magnitude = kwargs.get('magnitude', None)
        self.normalize = kwargs.get('normalize', False)
        self.direction_biases = np.array(kwargs.get('direction_biases', [0.25, 0.25, 0.25, 0.25]))
        
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

        # check direction_biases
        assert len(self.direction_biases) == 4, "direction_biases must have length 4"
        assert np.all(self.direction_biases >= 0), "direction_biases must be non-negative"
        assert np.sum(self.direction_biases) > 0, "direction_biases sum must be greater than 0"
        if np.sum(self.direction_biases) != 1:
            self.direction_biases = self.direction_biases / np.sum(self.direction_biases)
            print("direction_biases do not sum to 1, normalized to {}".format(self.direction_biases))
    

    def _init_sm_responses(self):
        """ Initialize sm_responses """
        super(AlloBoundaryCell, self)._init_sm_responses()
        
        # randomly generate the distance of the responses from the boundary (i.e., how far away from the boundary)
        cell_res_dist = self.rng.normal(self.res_dist, self.res_dist/3, self.n_cells).astype(int)
        cell_res_dist = np.clip(cell_res_dist, 0, None)

        for i in range(self.n_cells):
            # 0: top-down, 1: left-right, 2: bottom-up, 3: right-left
            direction = self.rng.choice([0, 1, 2, 3], 1, p=self.direction_biases)
            kernal = np.zeros((cell_res_dist[i]*2, 1))
            kernal[cell_res_dist[i]:, :] = 1
            kernal = np.rot90(kernal, direction) # rotate the kernal by 90 degrees clockwise `direction` times
            self.sm_responses[i, :, :] = -fftconvolve(self.arena.inv_arena_map, kernal, mode='same')/kernal.sum()
            self.sm_responses[i, :, :] -= self.sm_responses[i, :, :].min()

        # normalize sm_responses
        if self.normalize:
            self.sm_responses = (self.sm_responses - self.sm_responses.min()) / (self.sm_responses.max() - self.sm_responses.min())

        # set magnitude
        if self.magnitude is not None:
            self.sm_responses = self.magnitude * self.sm_responses

    def get_specs(self):
        # TODO: might be good add this for all cells
        specs = super().get_specs()
        specs['cell_max_avg'] = self.sm_responses.max(axis=(1, 2)).mean()
        specs['cell_min_avg'] = self.sm_responses.min(axis=(1, 2)).mean()
        specs['cell_mean_avg'] = self.sm_responses.mean(axis=(1, 2)).mean()
        return specs
