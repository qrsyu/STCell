import numpy as np

def generate_maze_0_arena(sr, **kwargs):
    """
    Generate a maze_0 arena with a border around it.
    @param sr: The spatial resolution of the arena.
    @param kwargs: The parameters of the arena.
    @return arena_map
    """
    room_width = int(kwargs.get('room_width', 100) / sr) # pixel
    room_distance = int(kwargs.get('room_distance', 50) / sr) # pixel
    tunnel_width = int(kwargs.get('tunnel_width', 20) / sr) # pixel
    width = room_width * 2 + room_distance
    height = room_width * 2 + room_distance
    border = 5

    arena_map = np.ones((height, width))  # Set all to walls
    # Add rooms
    arena_map[0:room_width, 0:room_width] = 0
    arena_map[0:room_width, -room_width:] = 0
    arena_map[-room_width:, 0:room_width] = 0
    arena_map[-room_width:, -room_width:] = 0
    # Add tunnel
    tunnel_start = room_width // 2 - tunnel_width // 2
    # Horizontal tunnels
    arena_map[tunnel_start:(tunnel_start+tunnel_width), room_width:(room_width+room_distance)] = 0
    arena_map[-(tunnel_start+tunnel_width):-(tunnel_start), room_width:(room_width+room_distance)] = 0
    # Vertical tunnels
    arena_map[room_width:(room_width+room_distance), tunnel_start:(tunnel_start+tunnel_width)] = 0
    arena_map[room_width:(room_width+room_distance), -(tunnel_start+tunnel_width):-(tunnel_start)] = 0

    # Add border around the arena
    arena_map = np.pad(arena_map, border, 'constant', constant_values=1)

    if kwargs.get('vertical'):
        arena_map = np.rot90(arena_map, 1)

    return arena_map
