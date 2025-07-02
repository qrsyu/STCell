def plt_hs(hs, min_fr=0.1):
    
    from matplotlib import pyplot as plt
    import numpy as np

    time_points, num_neurons = hs.shape[0], hs.shape[1]
    # Create subplots
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))

    # Select neurons with mean firing rate > 0.1
    mean_fr = hs.mean(axis=0)
    # Get the index where mean_fr > min_fr
    mask = mean_fr > min_fr
    neuron_indices = np.where(mask)[0]
    select_hs = hs[:, neuron_indices]
    del hs

    # Normalize the hs along time points (Sure to be correct!)
    norm_hs = select_hs / np.linalg.norm(select_hs, axis=0, keepdims=True)
    del select_hs
    
    # Sort neurons from maximum firing time
    max_time = np.argmax(norm_hs, axis=0)
    sorted_neuron_indices = np.argsort(max_time)
    norm_hs = norm_hs[:, sorted_neuron_indices]

    # Plot the normalized hs
    x_ticks = np.arange(0, time_points)
    ax.imshow(norm_hs.T, aspect='auto', cmap='jet', # extent=[x_ticks[0], x_ticks[-1], 0, norm_hs.shape[1]]
              )
    ax.set_xlabel('Time Points')
    ax.set_ylabel('Neurons')
    plt.tight_layout()
    return fig, ax
