import numpy as np
from .displacement_abs import DisplacementAbs
import rtgym


class VelocityAbs(DisplacementAbs):
    """
    Absolute velocity signal
    """
    sens_type = 'velocity_abs'
    def __init__(self, arena, **kwargs):
        """
        @kwarg n_cells: number of spatially modulated non-grid/place cellcells
        @kwarg magnitude: maximum magnitude of the cell responses
        @kwarg sigma_s: sigma_s for the Gaussian smoothing, default is 0.0 sec
        @kwarg ssigma_s: sigma_s for the spatial smoothing, default is 0.0 sec
        @kwarg normalize: normalize the cell responses
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
    def _displacement_to_velocity(displacement):
        """
        Convert displacement to velocity
        """
        # displacement: (n_batch, n_time, 2), last dimension is x, y
        # velocity: (n_batch, n_time, 1), the magnitude of the velocity
        velocity = np.linalg.norm(displacement, axis=2)
        velocity = velocity[:, :, np.newaxis]
        return velocity

    def _duplicate_res(self, res):
        """
        Duplicate the velocity to the number of cells
        """
        res = np.tile(res, (1, 1, self.n_cells))
        return res

    def get_responses(self, traj: rtgym.trial.Trajectory):
        vel = self._displacement_to_velocity(traj.disps)
        dup_res = self._duplicate_res(vel)
        return self._smooth_res(dup_res)

    def get_specs(self):
        specs = super().get_specs()
        specs['magnitude'] = self.magnitude
        specs['normalize'] = self.normalize
        return specs
