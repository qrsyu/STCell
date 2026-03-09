import numpy as np

def generate_two_rooms_arena(sr, **kwargs):
    """
    Generate a two_rooms arena with a border around it.
    @param sr: The spatial resolution of the arena.
    @param kwargs: The parameters of the arena.
    @return arena_map
    """
    room_width = int(kwargs.get('room_width', 70) / sr) # pixel
    height = int(kwargs.get('room_height', 70) / sr) # pixel
    room_distance = int(kwargs.get('room_distance', 30) / sr) # pixel
    tunnel_width = int(kwargs.get('tunnel_width', 20) / sr) # pixel
    width = room_width*2 + room_distance
    border = 5

    arena_map = np.zeros((height, width))
    arena_map[0:(height-tunnel_width)//2, room_width:(room_width+room_distance)] = 1
    arena_map[-(height-tunnel_width)//2:, room_width:(room_width+room_distance)] = 1

    arena_map = np.pad(arena_map, border, 'constant', constant_values=1)

    if kwargs.get('vertical'):
        arena_map = np.rot90(arena_map, 1)

    return arena_map
