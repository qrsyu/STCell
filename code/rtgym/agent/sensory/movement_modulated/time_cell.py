import numpy as np
from .mm_base import MMBase
from scipy.ndimage import gaussian_filter
import rtgym


class TimeCell(MMBase):
    """
    Absolute time signal, this may include negative values
    """
    sens_type = 'time_cell'
    def __init__(self, arena, **kwargs):
        """
        @kwarg n_cells: number of spatially modulated non-grid/place cellcells
        @kwarg magnitude: maximum magnitude of the cell responses
        @kwarg sigma_s: sigma_s for the Gaussian smoothing, default is 0.0 sec
        @kwarg ssigma_s: sigma_s for the spatial smoothing, default is 0.0 sec
        """
        super().__init__(arena, **kwargs)

        # parameters
        self.event_onset = kwargs.get('event_onset', [0.25, 0.75]) # in percentage
        self.event_onset_sigma = kwargs.get('event_onset_sigma', [0.01, 0.01]) # in percentage
        self.event_width = kwargs.get('event_width', [0.05, 0.05]) # in percentage
        # self.event_width_sigma = kwargs.get('event_width_sigma', [0.01, 0.01]) # in percentage

        self.temp_events = kwargs.get('temp_events', [np.zeros((self.n_cells,)), 
                                                      np.zeros((self.n_cells,))])        
        # self.magnitude = kwargs.get('mag', 1)
        # self.mag_func  = kwargs.get('mag_func', lambda x: x)
        # self.mag_sigma = kwargs.get('mag_sigma', 0.2)
        
        # Gaussian noise
        self.sigma       = kwargs.get('sigma', 0.5) 
        self.ssigma      = self._t_to_ts(kwargs.get('ssigma', 0.5)) # sigma of Gaussian noise smoothing 
        
        self.bias        = kwargs.get('bias', 0.0) 
        
        # check parameters
        self._check_params()

    def _check_params(self):
        """ Check parameters """
        assert self.n_cells > 0, "n_cells <= 0"
        assert self.n_cells % 2 == 0, "n_cells must be even"
        
        # Check temporal events onsets
        # ----------------------------------------------------------------------
        assert isinstance(self.event_onset, list), "onsets must be a list"
        assert len(self.event_onset) >= 1, "onsets must contain at least one values"
        assert all(0 <= onset <= 1 for onset in self.event_onset), "onsets must between 0 and 1"
        # ----------------------------------------------------------------------
        
        # Check temporal events widths
        # ----------------------------------------------------------------------
        assert isinstance(self.event_width, list), "widths must be a list"
        assert len(self.event_width) >= 1, "widths must contain at least one values"
        assert all(0 <= width <= 1 for width in self.event_width), "widths must between 0 and 1"
        # ----------------------------------------------------------------------

    # Generate time cell responses
    # -------------------------------------------------------------------------------------
    def _duplicate_res(self, res): # res: (time_steps,)
        """
        Duplicate the time responses to the number of cells and add fluctuations
        """
        # Generate step heights std
        # step_heights = np.random.normal(self.magnitude, self.mag_sigma, self.n_cells)  # (n_cells,)
        dup_res = np.tile(res.reshape(1, res.shape[0], res.shape[1]), (self.batch_size, 1, 1)) # (batch, time_steps, n_neurons) 
        
        # Generate random Gaussian noise
        noise = np.random.normal(self.bias, self.sigma, dup_res.shape)  # Shape matches dup_res
        noisy_res = dup_res + noise

        # Smooth out the temporal profile
        smoothed_res = gaussian_filter(noisy_res, sigma=(0, self.ssigma, 0))  # Smooth along the time dimension only

        return smoothed_res
    
    def get_responses(self, traj: rtgym.trial.Trajectory):
        self.batch_size = traj.disps.shape[0]
        trial_time_dims = traj.disps.shape[1]  # Total number of time steps

        # Determine the starts and width in unit of time steps (integer)
        event_onsets = [int(trial_time_dims * onset) for onset in self.event_onset]
        event_widths = [int(trial_time_dims * width) for width in self.event_width]
        event_onset_sigmas = [int(trial_time_dims * onset_sigma) for onset_sigma in self.event_onset_sigma]
        # Generate steplift time responses to events
        time_series = np.zeros((trial_time_dims, self.n_cells))
        
        for idx, onset in enumerate(event_onsets):
        # for onset, width, event in zip(event_onsets, event_widths, self.temp_events):
            # Set onsets and ends std
            onsets = np.random.normal(onset, event_onset_sigmas[idx], self.n_cells)
            # print(onsets)
            ends = onsets + event_widths[idx] + np.random.normal(0, event_onset_sigmas[idx], self.n_cells)  # Add some noise to the end time
            # Set the temporal event
            for cell in range(self.n_cells):
                onset = int(max(0, min(trial_time_dims-1, onsets[cell])))
                end   = int(max(0, min(trial_time_dims-1, ends[cell])))
                time_series[onset:end, cell] = self.temp_events[idx][cell]
                
        dup_res = self._duplicate_res(time_series)
        return dup_res
    #-------------------------------------------------------------------------------------
        
    def get_specs(self):
        specs = super().get_specs()
        specs['magnitude'] = self.magnitude
        return specs