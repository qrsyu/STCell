import numpy as np
from scipy.ndimage import gaussian_filter1d, distance_transform_edt
from .behavior_profile_parser import BehaviorProfileParser
from rtgym.trial import Trajectory


class Behavior():
    def __init__(
            self, 
            temporal_resolution, 
            spatial_resolution
        ):
        self.temporal_resolution = temporal_resolution
        self.spatial_resolution = spatial_resolution

    def set_arena(self, arena):
        self._arena = arena
        self._compute_distance_and_angle_maps(self._arena.arena_map)

    def init_from_profile(self, behavior_profile):
        """
        Initialize the behavior of the agent based on the provided behavior profile.
        """
        parser = BehaviorProfileParser()
        behavior_profile, preference_profile = parser.parse(behavior_profile)

        for k, v in behavior_profile.items(): setattr(self, k, v)
        self._compute_distance_and_angle_maps(self._arena.arena_map)

    def generate_trial(self, duration: float, batch_size: int, init_pos=None):
        """
            Generate a trial and return generated data.

        Parameters
            - duration: duration of the trial in seconds
            - batch_size: number of trials to generate
            - init_pos: initial position of the agent
            
        Returns:
            - coords: Generated coordinates over time
            - hds: Generated head directions over time
            - displacements: Generated displacements between coordinates
        """
        T = self.sec_to_frame(duration)
        coords = np.zeros((batch_size, T, 2))
        hds = np.zeros((batch_size, T, 1))
        directions = np.zeros((batch_size, T, 1))
        
        if init_pos is None:
            coords[:, 0, :] = self._arena.generate_random_pos(batch_size)
        else:
            coords[:, 0, :] = self.pos_to_coord(np.array(init_pos, dtype=np.float32))

        # Generate the trajectory
        self._generate_trajectory(coords, directions)
        displacements = np.diff(coords, axis=1)
        displacements = np.concatenate([displacements, np.zeros((batch_size, 1, 2))], axis=1)
        self._generate_hds(hds, displacements)

        return Trajectory(
            coords_float=coords,
            head_directions=hds,
            displacements=displacements
        )

    def pos_to_coord(self, pos):
        return np.array(pos/self.spatial_resolution)

    def sec_to_frame(self, s):
        return int(s/self.temporal_resolution*1e3)

    @staticmethod
    def _flip_coin(p, size):
        return np.random.choice([False, True], p=[1-p, p], size=size)

    def _random_velocity(self, size):
        # Solve for the mean and sigma of the underlying Gaussian
        # distribution with which we used to generate log-normal
        m = self.velocity_mean
        s = self.velocity_sd
        sigma = np.sqrt(np.log(1 + (s**2 / m**2)))
        mu = np.log(m**2 / np.sqrt(m**2 + s**2))
        dist = np.random.lognormal(mean=mu, sigma=sigma, size=size)
        return dist / 1e3 * self.temporal_resolution

    def _compute_distance_and_angle_maps(self, binary_map):
        if getattr(self, 'avoid_boundary_dist', -1) <= 0:
            return
        # Compute the distance map: Euclidean distance to the nearest boundary (obstacle)
        raw_distance_map = distance_transform_edt(1 - binary_map)

        # Apply Gaussian-like weighting based on the width
        distance_map = np.exp(- (raw_distance_map**2 / self.avoid_boundary_dist))  # Smoothly decays to zero beyond the width

        # Compute the gradient of the raw distance map
        gradient_y, gradient_x = np.gradient(raw_distance_map)

        # Compute the angle of the gradient at each point and normalize to [0, pi)
        perpend_angle_map = np.arctan2(-gradient_y, gradient_x)  # Flip the y-axis to match the coordinate system
        # Rotate left by 90 degrees to align with the direction of the gradient
        perpend_angle_map = np.mod(perpend_angle_map + np.pi/2, 2*np.pi) - np.pi

        self.distance_map = distance_map
        self.perpend_angle_map = perpend_angle_map

    @staticmethod
    def _random_direction(size):
        """
        Generate random 2D unit direction vectors.
        """
        return np.random.uniform(-np.pi, np.pi, size=size)[..., np.newaxis]

    @staticmethod
    def _compute_displacement(angle, velocity_norm):
        """
        Scale direction vectors by velocity norm.
        """
        # Scale by velocity norms
        direction = np.column_stack((np.cos(angle), np.sin(angle)))
        scaled_displacement = direction * velocity_norm[:, np.newaxis]
        return scaled_displacement

    def _generate_hds(self, hds, displacements):
        # TODO: Change scale
        hds[:] = np.random.normal(loc=0, scale=0.2, size=hds.shape)
        hds[:] = gaussian_filter1d(hds, sigma=5, axis=1)

        # Add the angle of the movement to the head direction
        angle_rad = np.arctan2(displacements[..., 0], displacements[..., 1])
        hds += angle_rad[..., np.newaxis]

    @staticmethod
    def _compute_direction(displacement):
        # Compute the angle of the displacement in radians
        direction = np.arctan2(displacement[..., 1], displacement[..., 0])
        return direction[..., np.newaxis]

    def _generate_trajectory(self, coords, directions):
        # Get the batch size and the number of time steps
        batch_size, T, _ = coords.shape

        # Generate velocity norms and random drifts
        velocity_norm = self._random_velocity(batch_size)
        random_drift = self._compute_displacement(
            self._random_direction(batch_size), 
            velocity_norm
        ) * self.random_drift_magnitude
        directions[:, 0] = self._random_direction(batch_size)  # Goal dir
        disp = self._compute_displacement(directions[:, 0], velocity_norm) + random_drift  # Actual disp
        directions[:, 0] = self._compute_direction(disp)  # Actual dir
        directions[:, 0] = self._avoid_boundary(directions[:, 0], coords[:, 0])  # Adjust to avoid boundary

        for i in range(1, T):
            # Update the location of the agent
            coords[:, i] = coords[:, i-1] + disp
            self._check_border(coords, directions, disp, i, velocity_norm)

            # Randomly switch the direction of the random drift
            switch_direction_idx = self._flip_coin(self.switch_direction_prob, batch_size)
            if np.sum(switch_direction_idx) > 0:
                random_drift[switch_direction_idx] = self._compute_displacement(
                    self._random_direction(np.sum(switch_direction_idx)),
                    velocity_norm[switch_direction_idx]
                ) * self.random_drift_magnitude
                
            # Randomly switch the velocity
            switch_velocity_idx = self._flip_coin(self.switch_velocity_prob, batch_size)
            if np.sum(switch_velocity_idx) > 0:
                velocity_norm[switch_velocity_idx] = self._random_velocity(np.sum(switch_velocity_idx))

            # Update the displacement if not the last time step
            if i < T - 1:
                disp = self._compute_displacement(directions[:, i-1], velocity_norm) + random_drift  # Actual disp
                directions[:, i] = self._compute_direction(disp)  # Goal dir for t+1 should be the actual dir for t
                directions[:, i] = self._avoid_boundary(directions[:, i], coords[:, i])  # Adjust to avoid boundary

    def _require_adjustment(self, direction, perpend_angle):
        # Compute angle difference
        angle_diff = perpend_angle[..., np.newaxis] - direction
        angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi  # Normalize to [-pi, pi)

        # Determine pointing boundary
        pointing_boundary = (angle_diff >= -np.pi / 2) & (angle_diff <= np.pi / 2)
        pointing_boundary = pointing_boundary.astype(int)

        # Compute complement angle
        complement_angle = -np.sign(angle_diff) * (np.pi / 2 - np.abs(angle_diff))

        return pointing_boundary, complement_angle

    def _avoid_boundary(self, direction, coords):
        if getattr(self, 'avoid_boundary_dist', -1) <= 0:
            return direction
        int_coords = coords.astype(int)
        avoid_coef = self.distance_map[int_coords[:, 0], int_coords[:, 1]]
        avoid_idx = np.where(avoid_coef > 1e-2)[0]  # Batches that are close to the boundary

        if len(avoid_idx) > 0:
            # If there are batches that are close to the boundary, adjust the direction
            perpend_angle = self.perpend_angle_map[int_coords[avoid_idx, 0], int_coords[avoid_idx, 1]]
            pointing_boundary, complement_angle = self._require_adjustment(direction[avoid_idx], perpend_angle)
            
            # Compute the adjustment coefficient
            avoid_coef = avoid_coef[avoid_idx][..., np.newaxis]
            avoid_coef = avoid_coef * pointing_boundary  # Not adjust if pointing away from the boundary

            # Adjust the direction
            direction[avoid_idx] = (1-avoid_coef) * direction[avoid_idx] + avoid_coef * (direction[avoid_idx] + complement_angle)

        return direction

    def _check_border(self, coords, directions, displacements, ts, velocity_norms):
        invalid_batches = np.logical_not(self._arena.validate_index(coords[:, ts].astype(int)))
        while np.sum(invalid_batches) > 0:
            # Generate new movement direction for those batches that hit the wall
            directions[invalid_batches, ts-1] = self._random_direction(np.sum(invalid_batches))
            disp = self._compute_displacement(
                directions[invalid_batches, ts-1], 
                velocity_norms[invalid_batches]
            )
            # Update the location of the invalid batches
            coords[invalid_batches, ts] = coords[invalid_batches, ts-1] + disp
            # Check if the new location is still invalid
            invalid_batches = np.logical_not(self._arena.validate_index(coords[:, ts].astype(int)))
