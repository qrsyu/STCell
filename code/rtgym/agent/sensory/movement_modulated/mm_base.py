import numpy as np
from scipy.ndimage import convolve1d
import matplotlib.pyplot as plt
from numpy.random import default_rng


class MMBase():
    """ Movement modulated behavior base class """
    sens_type = 'mm_base'
    def __init__(self, arena, n_cells, tr, sensory_key, seed=None, **kwargs):
        """
        @kwarg arena: arena object
        """
        self.arena = arena
        self.tr = tr  # temporal resolution
        self.sensory_key = sensory_key
        self.n_cells = n_cells

        # Initialize random number generator
        self.rng = default_rng(seed) if seed is not None else default_rng()

        # check if the class is the base class
        assert type(self) != MMBase, "MMBase is an abstract class"
        assert self.n_cells > 0, "n_cells <= 0"

    def _smooth_res(self, res):
        """
        Smooth the input with a causal (one-sided) Gaussian-like kernel

        @param res: input, (n_batch, n_time, n_cells)
        """
        if self.sigma_ts == 0:
            return_res = res
        else:
            n = res.shape[2]
            sigs = np.clip(np.random.normal(self.sigma_ts, self.ssigma_ts, n), a_min=5e-2, a_max=None)
            return_res = np.empty_like(res)
            
            for i in range(n):
                # Constructing a one-sided Gaussian-like kernel
                half_size = int(3 * sigs[i])  # To approximate a Gaussian
                kernel = np.exp(-np.arange(half_size + 1) ** 2 / (2 * sigs[i] ** 2))
                kernel = kernel / kernel.sum()  # Normalize to make it a probability distribution

                # Convolve with the kernel along the time dimension with 'valid' padding
                filtered = convolve1d(res[:, :-1, i], weights=kernel, axis=1, mode='nearest')
                return_res[:, :-1, i] = filtered  # The last time step is 0.
                return_res[:, -1, i] = res[:, -1, i]  # Pad the last time step with the last value
            
        # Scale the responses if magnitude is not None
        if self.magnitude is not None:
            return_res *= self.magnitude
        
        return return_res

    def _t_to_ts(self, t):
        """
        Convert time to time step
        """
        return int(t*1e3/self.tr)

    def vis(self, traj, N, *args, **kwargs):
        """
        Visualize the direction cells
        """
        responses = self.get_responses(traj)
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.plot(responses[0, :-1, :N]) # Last dimension will be 0.
        ax.set_title(self.__class__.__name__)
        ax.set_xlabel('Time')
        ax.set_ylabel('Response')
        return fig, ax

    def get_specs(self):
        return {
            'n_cells': self.n_cells,
        }
    
    def print_specs(self):
        """ Print specs """
        print_dict(self.get_specs())
