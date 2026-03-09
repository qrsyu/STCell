import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from functools import wraps


# DECORATORS
# ******************************************************************************
def check_init(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        if self._traj is None:
            raise ValueError("Trial not initialized. Please run new_trial() method first.")
        # If the trial is initialized, run the method
        return method(self, *args, **kwargs)
    return wrapper


# DATA CLASSES
# ******************************************************************************
class Coord:
    def __init__(self, coord_float: np.ndarray):
        self.coord_float = coord_float
        self.coord = coord_float.astype(int)
    
    def __repr__(self):
        return repr(self.coord)

    def __len__(self):
        # Return the length of the underlying coord array
        return len(self.coord)

    @property
    def shape(self):
        """
        Returns the shape of the underlying array.
        """
        return self.coord.shape

    @property
    def ndim(self):
        """
        Returns the number of dimensions of the underlying array.
        """
        return self.coord.ndim

    @property
    def dtype(self):
        """
        Returns the data type of the underlying array.
        """
        return self.coord.dtype

    @property
    def size(self):
        """
        Returns the size of the underlying array.
        """
        return self.coord.size

    @property
    def shape(self):
        """
        Returns the shape of the coordinates.
        """
        return self.coord.shape
    
    @property
    def i(self):
        """
        Returns the coordinates as integer values.
        """
        return self.coord

    @property
    def f(self):
        """
        Returns the coordinates as float values.
        """
        return self.coord_float
    
    def __getitem__(self, key):
        return self.coord[key]

    def __getattr__(self, attr):
        # Delegate any unknown attribute to the coords array
        return getattr(self.coord, attr)


# AGENT STATE DATA CLASS
# ******************************************************************************
class AgentState:
    """
    Class to hold the state data of the agent. This class is used to store the
    coordinates, coordinates in float, and head directions of the agent. It is just a fix-time
    snapshot of the agent's state.
    """
    def __init__(
        self,
        coord_float: np.ndarray = None,
        head_direction: np.ndarray = None,
        displacement: np.ndarray = None
    ):
        """
        Initialize the AgentState object with the given data. For all those values that are unknown,
        use np.nan as a placeholder. This data class helps standardize the data format for all different
        sensory modalities.

        Parameters:
            - coord_float: Coordinates of the agent as float values.
            - head_direction: Head direction of the agent.
            - displacement: Displacement of the agent between coordinates.
        """
        # Ensure that at least one input is not None
        if coord_float is None and head_direction is None and displacement is None:
            raise ValueError("At least one of coord_float, head_direction, or displacement must be provided.")

        # Determine batch size based on the first non-None input
        batch_size = next(
            x.shape[0] for x in (coord_float, head_direction, displacement) if x is not None
        )

        # Replace None inputs with np.nan arrays of appropriate shapes
        coord_float = coord_float if coord_float is not None else np.full((batch_size, 2), np.nan)
        head_direction = head_direction if head_direction is not None else np.full((batch_size, 1), np.nan)
        displacement = displacement if displacement is not None else np.full((batch_size, 2), np.nan)

        # Check if the input data are all numpy arrays
        assert isinstance(coord_float, np.ndarray), "coord_float must be a numpy array"
        assert isinstance(head_direction, np.ndarray), "head_direction must be a numpy array"
        assert isinstance(displacement, np.ndarray), "displacement must be a numpy array"

        # Check if the input data all have two dimensions, (n_batch, n_features)
        assert coord_float.ndim == 2, "coord_float must have two dimensions, (n_batch, 2)"
        assert head_direction.ndim == 2, "head_direction must have two dimensions, (n_batch, 2)"
        assert displacement.ndim == 2, "displacement must have two dimensions, (n_batch, 2)"

        self.coord = Coord(coord_float)
        self.head_direction = head_direction
        self.displacement = displacement


# TRIAL DATA CLASS
# ******************************************************************************
class Trajectory:
    """
    Class to hold trial data for the agent. This class is used to store the
    coordinates, coordinates in float, and head directions of the agent.
    """
    def __init__(
        self,
        coords_float: np.ndarray,
        head_directions: np.ndarray,
        displacements: np.ndarray
    ):
        """
        Initialize the Trajectory object with the given data.

        Parameters:
            - coords_float: Coordinates of the agent as float values.
            - head_directions: Head directions of the agent.
            - displacements: Displacements of the agent between coordinates.
                             The displacement at the last step is set to 0.
        """
        # Check if the input data are all numpy arrays
        assert isinstance(coords_float, np.ndarray), "coords_float must be a numpy array"
        assert isinstance(head_directions, np.ndarray), "head_directions must be a numpy array"
        assert isinstance(displacements, np.ndarray), "displacements must be a numpy array"

        # Check if the input data all have three dimensions, (n_batch, n_time, n_features)
        assert coords_float.ndim == 3, "coords_float must have three dimensions (n_batch, n_time, n_features)"
        assert head_directions.ndim == 3, "head_directions must have three dimensions (n_batch, n_time, n_features)"
        assert displacements.ndim == 3, "displacements must have three dimensions (n_batch, n_time, n_features)"

        # Check if the time dimension of the input data are the same
        assert coords_float.shape[1] == head_directions.shape[1], "coords_float and head_directions must have the same time dimension"
        assert coords_float.shape[1] == displacements.shape[1], "coords_float and displacements must have the same time dimension"

        # Store the input data
        self.coords = Coord(coords_float)
        self.hds = head_directions
        self.disps = displacements

    def copy(self):
        """
        Returns a copy of the Trajectory object.
        """
        return Trajectory(
            coords_float=self.coords.f.copy(),
            head_directions=self.hds.copy(),
            displacements=self.disps.copy()
        )

    def state(self, time_index):
        """
        Returns the agent state at the specified time index.
        
        Args:
            time_index (int): The time index for which to get the agent state.
        
        Returns:
            AgentState: An AgentState object containing the agent state at the specified time index.
        """
        return AgentState(
            coord_float=self.coords.f[:, time_index, :],
            head_direction=self.hds[:, time_index, :],
            displacement=self.disps[:, time_index, :]
        )
    
    def __len__(self):
        """
        Returns the number of time steps in the trajectory.
        """
        return self.coords.shape[1]

    def slice(self, start, end):
        return self.t_range((start, end))

    @property
    def n_steps(self):
        """
        Returns the number of time steps in the trajectory.
        """
        return self.coords.shape[1]

    def t_range(self, range_):
        """
        Returns a new Trajectory object with data trimmed to the specified range.
        
        Args:
            range_ (list or tuple): A 2-element list or tuple specifying the start
                                    and end indices for trimming.
        
        Returns:
            Trajectory: A new Trajectory object with trimmed data.
        """
        # Check if the range is valid
        assert len(range_) == 2, "range_ must be a tuple of length 2"
        assert range_[0] < range_[1], "range_[0] must be less than range_[1]"
        assert range_[1] <= self.coords.shape[1], "range_[1] must be less than the trial duration"
        assert range_[0] >= 0, "range_[0] must be greater than or equal to 0"

        # Trim the data
        return Trajectory(
            coords_float=self.coords.f[:, range_[0]:range_[1]],
            head_directions=self.hds[:, range_[0]:range_[1]],
            displacements=self.disps[:, range_[0]:range_[1]]
        )

    @staticmethod
    def load(path):
        """
        Load a Trajectory object from a file.

        Args:
            path (str): The file path to load the Trajectory object from.

        Returns:
            Trajectory: A Trajectory object loaded from the specified file.
        """
        assert os.path.splitext(path)[1] == '.npz', "File extension must be .npz"
        npz = np.load(path)
        return Trajectory(
            coords_float=npz['coords_float'],
            head_directions=npz['head_directions'],
            displacements=npz['displacements']
        )

    
# TRIAL CLASS
# ******************************************************************************
class Trial():
    def __init__(self, agent, arena):
        self._traj = None
        self.agent = agent
        self.arena = arena

    @property
    @check_init
    def shape(self):
        return self._traj.coords.shape

    @property
    def dur_ts(self):
        return self._traj.coords.shape[1]

    @property
    def dur_s(self):
        return self._traj.coords.shape[1] * self.agent.temporal_resolution / 1e3

    @property
    @check_init
    def trial_data(self):
        return self._traj

    @property
    @check_init
    def coords(self):
        return self._traj.coords

    @property
    @check_init
    def disps(self):
        return self._traj.disps

    @property
    @check_init
    def hds(self):
        return self._traj.hds
    
    def new_trial(self, duration: float, batch_size: int = 1, init_pos=None, external_traj=None):
        """
        This will only generate a trajectory, the trajectory can always be reused 
        if the arena (not sensory) remain to be the same and thus saves generation time.

        Parameters:
            - duration: duration of the trial in seconds
            - batch_size: number of trials to generate
            - init_pos: initial position of the agent
        """
        assert self.agent is not None, "agent not set"
        assert self.arena is not None, "arena not set"
        if external_traj is None:
            self._traj = self.agent.behavior.generate_trial(duration, batch_size, init_pos)
            print(type(self._traj))
        else:
            self._traj = Trajectory(
                                    coords_float=external_traj['coords'],
                                    head_directions=external_traj['hds'],
                                    displacements=external_traj['disps'],)
            print(type(self._traj))

    @check_init
    def get_responses(self, t_range=None, return_format='array', keys=None, str_filter=None, type_filter=None):
        """
        Get the sensory responses of the agent for the current trial.

        Parameters:
            t_range (tuple, optional): Start and end time of the trial. If None, use the entire trial.
            return_format (str, optional): Format of the output. Default is 'array' (numpy array 
                                            of shape (batch_size, duration, n_cells)).
            keys (str, optional): The sensory modality to get the response from.
                                If None or 'all', return responses for all modalities.
            str_filter (str, optional): Filter the sensory keys by the string.
            type_filter (str, optional): Filter the sensory keys by the type.
        Returns:
            dict or np.ndarray:
                - If `keys` is specified, returns the sensory response for the specified keys.
                - If `return_format` is 'array', returns a numpy array of shape 
                    (batch_size, duration, n_total_cells) containing all sensory responses.
                - If `return_format` is 'dict', returns a dictionary where keys are 
                    sensory names, and values are their responses. Each value
                    is a numpy array of shape (batch_size, duration, n_cells).
        """
        # Get the trial data for the specified time range
        traj = self._traj.t_range(t_range) if t_range else self._traj

        # If key is a string, make it a list
        keys = self.agent.sensory.filter_sensories(keys, str_filter, type_filter)

        # Get the sensory responses
        return self.agent.sensory.get_responses(
            traj=traj, keys=keys, return_format=return_format
        )

    @check_init
    def save_traj(self, path):
        assert os.path.splitext(path)[1] == '.npz', "File extension must be .npz"
        save_dict = {
            "coords_float":    self._traj.coords.f,
            "head_directions": self._traj.hds,
            "displacements":   self._traj.disps
        }
        np.savez(path, **save_dict)
        
    def get_traj(self):
        save_dict = {
            "coords_float":    self._traj.coords.f,
            "head_directions": self._traj.hds,
            "displacements":   self._traj.disps
        }
        return save_dict

    def load_traj(self, path):
        npz = np.load(path)
        self.load_state_dict(npz)

    def set_traj(self, traj):
        self._traj = traj
        locs = self.arena.arena_map[self._traj.coords[:, :, 0], self._traj.coords[:, :, 1]]
        assert np.all(locs == 0), "Invalid location in the arena map"
    
    def load_state_dict(self, state_dict):
        traj = Trajectory(**state_dict)
        self.set_traj(traj)

    # VISUALIZATION
    # ==============================================================================
    @check_init
    def vis_traj(self, return_format=None, height=3):
        """
        Visualize the trajectory of the agent.

        Parameters:
            - return_format: format of the returned visualization. Can be 'anim', 'html' or None.
                When 'anim'/'html', vis as an animation. Otherwise, vis as a static plot.
            - fig_height: height of the figure in inches.
        """

        plot_w, plot_h = self._compute_plot_dimensions(self.arena.arena_map, height)
        if return_format == 'anim' or return_format == 'html':
            return self.vis_gif(plot_w, plot_h, return_format)
        else:
            fig, ax = plt.subplots(figsize=(plot_w, plot_h))
            ax.imshow(self.arena.inv_arena_map)
            ax.plot(self.coords.f[:, :, 1].T, self.coords.f[:, :, 0].T, color='red')
            ax.set_axis_off()
            return fig, ax

    @staticmethod
    def _compute_plot_dimensions(arena_map, fig_height):
        aspect_ratio = arena_map.shape[1] / arena_map.shape[0]
        fig_width = fig_height * aspect_ratio
        return fig_width, fig_height

    @check_init
    def vis_gif(self, plot_w, plot_h, return_format='anim'):
        if self.coords.shape[1] == 0 or self.hds.shape[1] == 0:
            raise ValueError("self.coords or self.hds has no data along the expected axis.")

        plot_w, plot_h = self._compute_plot_dimensions(self.arena.arena_map)
        fig, ax = plt.subplots(figsize=(plot_w, plot_h))
        ax.imshow(self.arena.inv_arena_map)
        ax.axis('off')
        line, = ax.plot([], [], 'r')
        
        # Initialize the quiver arrow
        arrow = ax.quiver(0, 0, 0.05, 0.05, angles='xy', scale_units='xy', scale=0.2, color='blue')

        def init():
            line.set_data([], [])
            arrow.set_offsets([0, 0])  # Reset position
            arrow.set_UVC(0.05, 0.05)    # Initial direction
            return line, arrow

        def animate(i):
            max_index = min(i * 10, self.coords.shape[1])
            
            # Update line data within bounds
            line.set_data(self.coords.f[0, :max_index, 1], self.coords.f[0, :max_index, 0])

            if max_index > 0:
                # Get the tip position of the line
                tip_x, tip_y = self.coords.f[0, max_index - 1, 1], self.coords.f[0, max_index - 1, 0]
                
                # Calculate direction based on the angle
                angle = self.hds[0, max_index - 1, 0]
                direction_x = np.cos(angle)
                direction_y = np.sin(angle)
                
                # Update arrow position and direction without removing it
                arrow.set_offsets([tip_x, tip_y])
                arrow.set_UVC(direction_x, direction_y)
            
            return line, arrow

        frames = int(len(self.coords.f[0]) / 10)
        anim = animation.FuncAnimation(fig, animate, init_func=init, frames=frames, interval=50, blit=True)
        plt.tight_layout()

        return HTML(anim.to_jshtml()) if return_format == 'html' else anim

    @check_init
    def vis_sensory(self, N: int = 10, keys=None, str_filter=None, type_filter=None):
        """
        Visualize the sensory responses of the agent

        Parameters:
            - N: number of sensory cells to visualize for each sensory modality
            - duration: default duration of the trial in seconds
            - batch_size: default number of trials to generate
        """
        # Generate figures for each sensory modality
        plot_list = []
        keys = self.agent.sensory.filter_sensories(keys, str_filter, type_filter)
        for key in keys:
            assert key in self.agent.sensory.sensories, f"Unknown sensory name: {key}"
            plot_list.append(self.agent.sensory.sensories[key].vis(N=N, traj=self._traj))
        return plot_list