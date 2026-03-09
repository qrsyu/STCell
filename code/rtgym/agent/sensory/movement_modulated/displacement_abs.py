import numpy as np
from .mm_base import MMBase
import rtgym


class DisplacementAbs(MMBase):
    """
    Absolute displacement signal, this may include negative values
    """
    sens_type = 'displacement_abs'
    def __init__(self, arena, **kwargs):
        """
        @kwarg n_cells: number of spatially modulated non-grid/place cellcells
        @kwarg magnitude: maximum magnitude of the cell responses
        @kwarg sigma_s: sigma_s for the Gaussian smoothing, default is 0.0 sec
        @kwarg ssigma_s: sigma_s for the spatial smoothing, default is 0.0 sec
        """
        super().__init__(arena, **kwargs)

        # parameters
        self.magnitude = kwargs.get('magnitude', 1)
        self.sigma_ts = self._t_to_ts(kwargs.get('sigma_s', 0.0))
        self.ssigma_ts = self._t_to_ts(kwargs.get('ssigma_s', 0.0))

        # check parameters
        self._check_params()

    def _check_params(self):
        """ Check parameters """
        assert self.n_cells > 0, "n_cells <= 0"
        assert self.n_cells % 2 == 0, "n_cells must be even"

    def _duplicate_res(self, res):
        """
        Duplicate the displacement to the number of cells
        """
        return np.tile(res, (1, 1, self.n_cells//2))

    def get_responses(self, traj: rtgym.trial.Trajectory):
        dup_res = self._duplicate_res(traj.disps)
        return self._smooth_res(dup_res)

    def get_specs(self):
        specs = super().get_specs()
        specs['magnitude'] = self.magnitude
        return specs
