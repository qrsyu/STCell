import numpy as np
from .weak_sm_cell import WeakSMCell


class WeakSMBinaryCell(WeakSMCell):
    """ 
    Spatially modulated sensory with binary responses 
    i.e. response to an object at a location is either 1 or 0
    @attr sm_responses: spatially modulated responses, shape (n_cells, arena_height, arena_width)
    """
    sens_type = 'weak_sm_binary_cell'
    def __init__(self, **kwargs):
        kwargs['magnitude'] = 1
        self.threshold = kwargs.pop('threshold', 0.3)
        super().__init__(**kwargs)

    def _check_params(self):
        super()._check_params()
        assert self.threshold >= 0 and self.threshold <= 1, "threshold must be between 0 and 1"

    def _init_sm_responses(self):
        super()._init_sm_responses()
        self.sm_responses = np.where(self.sm_responses > self.threshold, 1, 0)

    def get_specs(self):
        specs = super().get_specs()
        specs['threshold'] = self.threshold
        return specs
