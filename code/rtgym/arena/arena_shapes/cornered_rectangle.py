import numpy as np

def generate_cornered_rectangle_arena(sr, **kwargs):
    """
    Generate a rectangle arena with a border around it.
    @param sr: The spatial resolution of the arena.
    @param kwargs: The parameters of the arena.
    @return arena_map
    """
    width = int(kwargs.get('width', 100) / sr) # pixel
    height = int(kwargs.get('height', 100) / sr) # pixel
    corner = int(kwargs.get('corner', 15) / sr) # pixel

    assert 2*corner < width and 2*corner < height, 'corner must be smaller than half of the width and height'

    border = 5
    arena_map = np.zeros((height, width))
    arena_map[:corner, :corner] = 1
    arena_map[:corner, -corner:] = 1
    arena_map[-corner:, :corner] = 1
    arena_map[-corner:, -corner:] = 1
    arena_map = np.pad(arena_map, border, 'constant', constant_values=1)

    return arena_map
