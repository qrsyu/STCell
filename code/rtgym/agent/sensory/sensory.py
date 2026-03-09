"""
The sensory module contains the Sensory class which is responsible for 
creating and managing spatially and movement-modulated sensory cells of the agent.
"""


import rtgym
import pickle
import numpy as np
from .spatial_modulated import *
from .movement_modulated import *


class Sensory:
    """
    Sensory class for spatially and movement-modulated sensory cells.
    This class is usually not directly initialized but initialized by the rtgym.

    **Parameters**
        temporal_resolution: temporal resolution of the sensory cells (ms)
        spatial_resolution: spatial resolution of the sensory cells (cm)
        sensory_profile: profile of the sensory cells.
    """
    def __init__(self, temporal_resolution, spatial_resolution):
        self.temporal_resolution = temporal_resolution
        self.spatial_resolution = spatial_resolution
        self.sensories = {}
        self.ranges = None  # Keep track of the indices of the sensory cells
        self.sensories = {}

    def set_arena(self, arena):
        """
        Set the arena for the sensory cells.

        Parameters:
            arena: rtgym.arena.Arena object
        """
        self._arena = arena

    def list_all(self):
        """
        List all the sensory cells.
        """
        return list(self.sensories.keys())

    def init_from_profile(self, sensory_profile):
        # Initialize spatial and movement modulated cells
        self.sensories = self._initialize_sensories(sensory_profile if sensory_profile is not None else {})
        _ranges = np.cumsum([_sens.n_cells for _sens in self.sensories.values()])
        _ranges = np.insert(_ranges, 0, 0).tolist()
        self.ranges = {key: (_ranges[i], _ranges[i+1]) for i, key in enumerate(self.sensories.keys())}

    def _initialize_sensories(self, profile_list):
        """
        Initializes the cells based on the provided profile list.
        """
        sensories = {}
        common_params = {'arena': self._arena, 'tr': self.temporal_resolution}
        for key, value in profile_list.items():
            sensory_class = Sensory._get_sensory_class(value['type'])
            sensories[key] = sensory_class(sensory_key=key, **common_params, **value)
        return sensories
    
    def filter_sensories(self, keys=None, str_filter=None, type_filter=None):
        """
        This helps to find the keys of the sensory cells that match the given criteria.

        It will prioritize the most specific filter. The specificity from most to least is:
            keys > str_filter > type_filter
        """
        if keys is not None:
            if isinstance(keys, str):
                return [keys]
            elif isinstance(keys, list):
                return keys
            else:
                raise ValueError(f"Unknown keys: {keys}")
        elif str_filter is not None:
            return [key for key in self.sensories.keys() if str_filter in key]
        elif type_filter is not None:
            return [key for key, sensory_item in self.sensories.items() if type_filter == sensory_item.sens_type]
        else:
            return list(self.sensories.keys())

    def num_sensories(self, keys=None, str_filter=None, type_filter=None):
        keys = self.filter_sensories(keys, str_filter, type_filter)
        return sum([self.sensories[key].n_cells for key in keys])

    @staticmethod
    def _get_sensory_class(sensory_type):
        sensory_classes = {cls.sens_type: cls for cls in 
                           [WeakSMCell, PlaceCell, WeakSMRandCell, WeakSMBinaryCell, 
                            BoundaryVecCell, BoundaryCell, AlloBoundaryCell, WeakHDSMCell,
                            VelocityAbs, DirectionRad, DisplacementAbs, TimeCell, 
                            AccelerationAbs, GridCell, HeadDirectionCell]}
        
        assert sensory_type in sensory_classes, f"Unknown sensory type: {sensory_type}"
        return sensory_classes[sensory_type]

    def get_responses(self, traj: rtgym.trial.Trajectory, return_format='dict', keys=None, str_filter=None, type_filter=None):
        """
        Get sensory responses for the given trajectory.

        Parameters:
            - traj: rtgym.trial.Trajectory object
            - return_format: format of the returned responses. Can be 'dict' or 'array'.
            - keys: list of sensory keys to get responses. If None, get responses for all.
            - str_filter: filter the sensory keys by the given string.
            - type_filter: filter the sensory keys by the given type.

        Returns:
            - responses: sensory responses. The responses are of shape (n_cells, *arena_dimensions).
                         After indexing, it will be of shape (n_cells, n_batch).
                         When return_format is 'dict', it will be a dictionary of responses.
                         When return_format is 'array', it will be a numpy array of responses.
        """
        # Set filter_keys if not provided
        keys = self.filter_sensories(keys, str_filter, type_filter)
        if return_format == 'dict':
            return {key: self.sensories[key].get_responses(traj) for key in keys}
        elif return_format == 'array':
            res_list = [self.sensories[key].get_responses(traj) for key in keys]
            return np.concatenate(res_list, axis=-1)
        else:
            raise ValueError(f"Unknown return format: {return_format}")

    def observe(self, state: rtgym.trial.AgentState, keys=None, return_dict=False):
        """
        Observe the state and update the responses. This is just a fixed-time version of get_responses.

        Parameters:
            state: rtgym.trial.AgentState object
            keys: list of sensory keys to observe. If None, observe all.
            return_dict: return the responses as a dictionary
        
        Returns:
            responses: sensory responses. The responses are of shape (n_cells, *arena_dimensions).
                         After indexing, it will be of shape (n_cells, n_batch).
                         When return, reshape it to (n_batch, 1, n_cells).
        """
        if keys is not None:
            if return_dict:
                return {key: self.sensories[key].observe(state) for key in keys}
            else:
                arr = [self.sensories[key].observe(state) for key in keys]
                return np.concatenate(arr, axis=-1)
        else:
            if return_dict:
                return {key: _sens.observe(state) for key, _sens in self.sensories.items()}
            else:
                arr = [_sens.observe(state) for _sens in self.sensories.values()]
                return np.concatenate(arr, axis=-1)

    def compute_res(self):
        for _sens in self.sensories.values():
            _sens._compute_res()

    def save(self, file_path):
        """
        Save the sensory cells to a file.

        Parameters:
            file_path: Path to the file where the sensory cells will be saved.
        """
        all_sensory_data = {}
        for key, _sens in self.sensories.items():
            all_sensory_data[key] = _sens.state_dict()
        with open(file_path, 'wb') as f:
            pickle.dump(all_sensory_data, f)

    def load_from_state_dict(self, state_dict, append=True):
        """
        Load the sensory cells from a state dictionary.

        Parameters:
            state_dict: state dictionary of the sensory cells.
            append: if True, append the sensory cells to the existing sensory cells.
                    if False, replace the existing sensory cells.
        """
        if not append:
            self.sensories = {}
        for key in state_dict.keys():
            _sens_data = state_dict[key]
            _sens_class = self._get_sensory_class(_sens_data.pop('sens_type'))
            _sens = _sens_class.load_from_dict(_sens_data, self._arena)
            self.sensories[key] = _sens

    def load(self, file_path):
        """
        Load the sensory cells from a file.

        Parameters:
            file_path: Path to the file where the sensory cells are saved.
        """
        with open(file_path, 'rb') as f:
            all_sensory_data = pickle.load(f)
        self.load_from_state_dict(all_sensory_data)
