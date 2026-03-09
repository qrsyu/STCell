import numpy as np

def generate_maze_1_arena(sr, **kwargs):
    """
    Generate a maze_1 arena with a border around it.
    @param sr: The spatial resolution of the arena.
    @param kwargs: The parameters of the arena.
    @return arena_map
    """
    width, height = 400, 250
    border = 10

    arena_map = np.ones((height, width))  # Set all to walls

    # Room 1
    arena_map[0:120, 0:80] = 0

    # Room 2
    arena_map[0:60, 100:200] = 0

    # Room 3
    arena_map[0:30, 220:340] = 0

    # Room 4
    arena_map[0:100, 340:400] = 0

    # Room 5
    arena_map[140:250, 0:80] = 0
    arena_map[180:250, 80:220] = 0
    arena_map[160:250, 220:280] = 0

    # Room 6
    arena_map[80:160, 100:200] = 0

    # Room 7
    arena_map[50:140, 220:320] = 0
    arena_map[120:250, 300:400] = 0


    # Tunnels
    # Room 1; Room 2; Room 5
    arena_map[120:140, 40:100] = 0
    # Room 7; Room 4
    arena_map[100:120, 370:400] = 0
    # Room 2; Room 3; Room 6
    arena_map[60:80, 180:220] = 0
    # Room 2; Room 3
    arena_map[30:60, 220:240] = 0
    # Room 6; Room 7
    arena_map[230:250, 280:300] = 0

    # Wall
    arena_map[180:220, 140:160] = 1
    arena_map[220:250, 240:260] = 1
    
    
    # Add border around the arena
    arena_map = np.pad(arena_map, border, 'constant', constant_values=1)

    if kwargs.get('vertical'):
        arena_map = np.rot90(arena_map, 1)

    return arena_map
