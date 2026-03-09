import numpy as np

def generate_loop_arena(sr, **kwargs):
    """
    Generate a loop arena with a border around it.
    @param sr: The spatial resolution of the arena.
    @param kwargs: The parameters of the arena.
    @return arena_map
    """
    outer_radius = int(kwargs.get('outer_radius', 50) / sr) # pixel
    inner_radius = int(kwargs.get('inner_radius', 35) / sr) # pixel
    border = int(kwargs.get('inner_radius', 5) / sr) # pixel
    print(outer_radius, inner_radius, border)
    width = outer_radius * 2 + border * 2  # Define the full arena size
    arena_map = np.ones((width, width))     # Initialize as walls
    
    # Generate the circular loop
    for i in range(arena_map.shape[0]):
        for j in range(arena_map.shape[1]):
            dist_sq = (i - outer_radius - border) ** 2 + (j - outer_radius - border) ** 2
            if inner_radius**2 <= dist_sq <= outer_radius**2:
                arena_map[i, j] = 0  # Cut out the loop

    return arena_map