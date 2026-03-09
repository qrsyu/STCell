import numpy as np

def generate_circle_arena(sr, **kwargs):
    """
    Generate a circle arena with a border around it.
    @param sr: The spatial resolution of the arena.
    @param kwargs: The parameters of the arena.
    @return arena_map
    """
    radius = int(kwargs.get('radius', 50) / sr) # pixel
    border = 5

    width = radius*2 + border*2
    arena_map = np.ones((width, width)) # Fill everything as wall

    # generate a circle
    for i in range(arena_map.shape[0]):
        for j in range(arena_map.shape[1]):
            if (i-radius-border)**2 + (j-radius-border)**2 <= radius**2:
                arena_map[i, j] = 0 # Cut out the circle

    return arena_map
