import numpy as np
from .displacement_abs import DisplacementAbs
import rtgym


class DirectionRad(DisplacementAbs):
    """
    Direction cell
    """
    sens_type = 'direction_rad'
    def __init__(self, arena, **kwargs):
        """
        @kwarg n_cells: number of spatially modulated non-grid/place cellcells
        @kwarg magnitude: maximum magnitude of the cell responses
        @kwarg normalize: normalize the cell responses
        @kwarg sigma_s: sigma for the Gaussian smoothing, default is 0.0 sec
        @kwarg ssigma_s: sigma for the spatial smoothing, default is 0.0 sec
        """
        super().__init__(arena, **kwargs)

        # parameters
        self.magnitude = kwargs.get('magnitude', None)
        self.sigma_ts = self._t_to_ts(kwargs.get('sigma_s', 0.0))
        self.ssigma_ts = self._t_to_ts(kwargs.get('ssigma_s', 0.0))
        self.normalize = kwargs.get('normalize', False)

        # check parameters
        self._check_params()

    def _check_params(self):
        """ Check parameters """
        assert self.n_cells > 0, "n_cells <= 0"

    @staticmethod
    def _displacement_to_hd(displacement):
        """
        Convert displacement to direction
        """
        # displacement: (n_time, n_batch, 2), last dimension is x, y
        # direction: (n_time, n_batch, 1), the radian angle within [0, 2pi]
        dirs = np.arctan2(displacement[:, :, 0], displacement[:, :, 1])
        return dirs[:, :, np.newaxis]

    def _duplicate_res(self, res):
        """
        Duplicate the direction to the number of cells
        """
        return np.tile(res, (1, 1, self.n_cells))

    def get_responses(self, traj: rtgym.trial.Trajectory):
        disps = traj.disps  # [batch, timesteps, 2]
        dirs = self._displacement_to_hd(disps)
        dup_res = self._duplicate_res(dirs)
        return self._smooth_res(dup_res)

    def get_specs(self):
        specs = super().get_specs()
        specs['magnitude'] = self.magnitude
        specs['normalize'] = self.normalize
        return specs
