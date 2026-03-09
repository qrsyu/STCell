import numpy as np

def generate_rectangle_arena(sr, **kwargs):
    """
    Generate a rectangle arena with a border around it.
    @param sr: The spatial resolution of the arena.
    @param kwargs: The parameters of the arena.
    @return arena_map
    """
    dimensions = kwargs.get('dimensions', [100, 100])
    assert len(dimensions) == 2, 'dimensions must be a list of 2 elements'

    # convert dimensions from cm to pixel
    dimensions = np.array([int(dimensions[i] / sr) for i in range (2)]) # pixel

    border = 5
    arena_map = np.zeros(dimensions)
    arena_map = np.pad(arena_map, border, 'constant', constant_values=1)

    return arena_map
