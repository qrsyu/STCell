import numpy as np
from rtgym.agent.behavior import Behavior
from rtgym.agent.sensory import Sensory

class Agent():
    def __init__(
            self,
            temporal_resolution,
            spatial_resolution,
            ):
        self.temporal_resolution = temporal_resolution
        self.spatial_resolution = spatial_resolution
        self._arena = None
        self.behavior_profile = None
        self.sensory_profile = None
        self.sensory = Sensory(
            self.temporal_resolution,
            self.spatial_resolution,
        )
        self.behavior = Behavior(
            self.temporal_resolution, 
            self.spatial_resolution
        )

    @property
    def sensories(self):
        return self.sensory.sensories

    def set_arena(self, arena):
        self._arena = arena
        # if the arena is reset, the behavior and sensory should be reset as well
        if arena is not None:
            self.behavior.set_arena(arena)
            self.sensory.set_arena(arena)
            self._init_behavior_from_profile()
            self._init_sensory_from_profile()

    def set_behavior(self, behavior_profile):
        self.behavior_profile = behavior_profile
        self._init_behavior_from_profile()

    def _init_behavior_from_profile(self):
        """ 
        Initialize behavior from behavior profile 

        If called from set_arena, block if the self.behavior_profile is None
        If called from set_behavior, block if the self._arena is None
        """
        if self.behavior_profile is not None and self._arena is not None:
            self.behavior.init_from_profile(self.behavior_profile)

    def set_sensory(self, sensory_profile):
        self.sensory_profile = sensory_profile
        self._init_sensory_from_profile()

    def set_sensory_manually(self, sens_type, sens):
        assert False, "TODO: some error to be fixd"
        if self.sensory is not None:
            print('Warning: sensory is reset')
        self.sensory.set_sensory_manually(sens_type, sens)
        self.sensory = sensory

    @property
    def sensories(self):
        return self.sensory.sensories

    def _init_sensory_from_profile(self):
        """
        Initialize sensory from sensory profile

        If called from set_arena, block if the self.sensory_profile is None
        If called from set_sensory, block if the self._arena is None
        """
        if self.sensory_profile is not None and self._arena is not None:
            self.sensory.init_from_profile(self.sensory_profile)
