class BehaviorProfileParser:
    """
    This class parses behavior defined in .yaml files into a standardized format.
    The parsed behavior profile is then used to generate agent behavior.
    """
    def __init__(self):
        pass

    def parse(self, raw_profile):
        self.profile = {}
        self.preference = {}
        self.raw_profile = raw_profile

        self.parse_subject_movement()
        # self.parse_spatial_preference()

        return self.profile, None

    def parse_subject_movement(self):
        # movement behavior
        assert self.raw_profile['velocity_mean'] is not None, 'velocity_mean must be specified'
        assert self.raw_profile['velocity_mean'] > 0, 'velocity_mean must be positive'
        assert self.raw_profile['random_drift_magnitude'] is not None, 'random_drift_magnitude must be specified'
        assert self.raw_profile['random_drift_magnitude'] >= 0, 'random_drift_magnitude must be non-negative'
        assert self.raw_profile['switch_direction_prob'] is not None, 'switch_direction_prob must be specified'
        assert self.raw_profile['switch_direction_prob'] >= 0 and self.raw_profile['switch_direction_prob'] <= 1, 'switch_direction_prob must be in [0, 1]'
        assert self.raw_profile['switch_velocity_prob'] is not None, 'switch_velocity_prob must be specified'
        assert self.raw_profile['switch_velocity_prob'] >= 0 and self.raw_profile['switch_velocity_prob'] <= 1, 'switch_velocity_prob must be in [0, 1]'
        self.profile['velocity_mean'] = self.raw_profile['velocity_mean']
        self.profile['velocity_sd'] = max(self.raw_profile['velocity_sd'], 0)
        self.profile['random_drift_magnitude'] = self.raw_profile['random_drift_magnitude']
        self.profile['switch_direction_prob'] = self.raw_profile['switch_direction_prob']
        self.profile['switch_velocity_prob'] = self.raw_profile['switch_velocity_prob']
        self.profile['avoid_boundary_dist'] = self.raw_profile.get('avoid_boundary_dist', -1)

    def parse_spatial_preference(self):
        # assign blending factor
        blend_factor = self.raw_profile['blend_factor']
        self.preference['blend_factor'] = blend_factor if blend_factor is not None else 5
        assert self.preference['blend_factor'] > 0, 'blend_factor must be positive'

        # assign spatial preference
        if self.raw_profile['spatial_preferences'] is not None:
            spatial_preferences = self.raw_profile['spatial_preferences']
            for key in spatial_preferences:
                if key in ['corner', 'wall', 'bias']:
                    self.preference[f'{key}_pref'] = spatial_preferences[key]
                else:
                    assert False, 'Spatial preference type `{}` not supported'.format(key)

        # fill in missing spatial preferences, default to 0
        for key in ['corner', 'wall', 'bias']:
            if key not in spatial_preferences:
                self.preference[f'{key}_pref'] = 0
