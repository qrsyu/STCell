"""
RatatouGym is a python package that provides a simple interface to generate sensory responses
from a virtual agent in a virtual environment.

This file contains more documentation as it serves as the main API for the package.
"""

import os
import numpy as np
from typing import Dict
from rtgym.agent import Agent
from rtgym.arena import Arena
from rtgym.trial import Trial
import torch


class RatatouGym():
    def __init__(
            self,
            temporal_resolution,
            spatial_resolution,
            **kwargs
        ):
        self.temporal_resolution = temporal_resolution
        self.spatial_resolution = spatial_resolution
        self.arena = Arena(spatial_resolution=self.spatial_resolution)
        self.agent = Agent(
            temporal_resolution=self.temporal_resolution,
            spatial_resolution=self.spatial_resolution,
        )
        self.trial = Trial(agent=self.agent, arena=self.arena)

    # Basic get methods without trial generation
    # ======================================================================================
    @property
    def t_res(self):
        return self.temporal_resolution

    @property
    def s_res(self):
        return self.spatial_resolution

    def to_ts(self, t):
        if isinstance(t, (int, float)):
            return int(np.round(t * 1e3 / self.t_res))
        elif isinstance(t, np.ndarray):
            return np.round(t * 1e3 / self.t_res).astype(int)
        elif 'torch' in str(type(t)):
            return torch.round(t * 1e3 / self.t_res).int()
        else:
            raise TypeError("Unsupported type for t")

    def to_sec(self, ts):
        if isinstance(ts, (int, float)):
            return ts * self.t_res / 1e3
        elif isinstance(ts, np.ndarray):
            return ts * self.t_res / 1e3
        elif 'torch' in str(type(ts)):
            return ts * self.t_res / 1e3
        else:
            raise TypeError("Unsupported type for ts")

    @property
    def arena_map(self):
        return self.arena.arena_map

    def random_pos(self, n):
        return self.arena.generate_random_pos(n)
    # ======================================================================================
    
    # Set methods to initialize the arena and agent
    # ======================================================================================
    def load_arena(self, path):
        """
        Load an arena from a file
        """
        self.arena.load(path)
        self.agent.set_arena(self.arena)

    def save_arena(self, path):
        """
        Save the arena to a file
        """
        # file extension is npz
        assert os.path.splitext(path)[1] == '.npz', "file extension must be .npz"
        self.arena.save(path)

    def init_arena_map(self, **kwargs):
        """
        Initialize the arena map with a predefined shape
        """
        self.arena.init_arena_map(**kwargs)
        self.agent.set_arena(self.arena)

    def set_arena_map(self, arena_map):
        """
        Set arena map with custom map (numpy array; 0: free space, 1: wall)
        """
        self.arena.set_arena_map(arena_map=arena_map)
        self.agent.set_arena(self.arena)
    # ======================================================================================

    # Behavior methods
    # ======================================================================================
    def set_behavior_from_profile(self, profile: Dict):
        """
        Set behavior from a profile (Dict)
        """
        self.agent.set_behavior(behavior_profile=profile)
    # ======================================================================================
    
    # Sensory methods
    # ======================================================================================
    def set_sensory_from_profile(self, profile: Dict):
        """
        Set sensory profile from a json file
        """
        assert len(profile.keys()) == len(set(profile.keys())), "sensory keys must be unique"
        self.agent.set_sensory(sensory_profile=profile)

    def set_sensory_manually(self, sens_type, sensory):
        assert False, "TODO: temporary disabled"
        self.agent.set_sensory_manually(sens_type, sensory)
    # ======================================================================================

    # Trial generation and methods
    # ======================================================================================
    def new_trial(self, *args, **kwargs):
        """
        This will only generate a trajectory, the trajectory can always be reused 
        if the arena (not sensory) remain to be the same and thus saves generation time.

        Parameters:
            - duration: duration of the trial in seconds
            - batch_size: number of trials to generate
            - init_pos: initial position of the agent
        """
        return self.trial.new_trial(*args, **kwargs)

    def save_trial(self, path, file_name):
        """
        Save the generated trial to a file

        Parameters:
            - path: path to save the trial
        """
        self.trial.save_trial(path, file_name)

    def load_trial(self, path):
        """
        Load a trial from a file

        Parameters:
            - path: path to load the trial
        """
        self.trial.load_trial(path)

    def vis_traj(self, *args, **kwargs):
        return self.trial.vis_traj(*args, **kwargs)

    def vis_sensory(self, *args, **kwargs):
        return self.trial.vis_sensory(*args, **kwargs)
