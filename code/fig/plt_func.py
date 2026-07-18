def plt_hs(hs, min_fr=0.1, fig=None, ax=None, time_start=None, time_end=None):
    import numpy as np
    from matplotlib import pyplot as plt

    # Select neurons with mean firing rate > 0.1
    mean_fr = hs.mean(axis=0)
    # Get the index where mean_fr > min_fr
    mask = mean_fr > min_fr
    neuron_indices = np.where(mask)[0]
    select_hs = hs[:, neuron_indices]
    del hs

    # Normalize the hs along time points (Sure to be correct!)
    # norm_hs = select_hs / np.linalg.norm(select_hs, axis=0, keepdims=True)
    # 0-1 Normalise method 2
    norm_hs = np.zeros_like(select_hs)
    for n in range(select_hs.shape[1]):
        norm_hs[:, n] = (select_hs[:, n]-np.min(select_hs[:, n]))/(np.max(select_hs[:, n])-np.min(select_hs[:, n]))
    del select_hs
    
    # Sort neurons from maximum firing time
    max_time = np.argmax(norm_hs, axis=0)
    sorted_neuron_indices = np.argsort(max_time)
    norm_hs = norm_hs[:, sorted_neuron_indices]
    
    time_pts, num_neurons = norm_hs.shape[0], norm_hs.shape[1]
    
    # Plot the normalized hs
    ax.imshow(norm_hs.T, aspect='auto', cmap='jet',  
              extent=[0, time_pts/10, num_neurons, 0]
              )
    
    # Plot a black dot at (max_time, neuron_index)
    if time_start is not None and time_end is not None:
        max_time_sorted = np.argmax(norm_hs, axis=0)
        # Plot those within the time interval
        time_mask = (max_time_sorted >= time_start) & (max_time_sorted <= time_end)
        y = np.arange(num_neurons)[time_mask]
        max_time_sorted = max_time_sorted[time_mask]
        ax.scatter(max_time_sorted/10, y, s=1, c='k')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Neurons')
    ax.set_xticks(np.linspace(0, norm_hs.shape[0]/10, 6))
    
    return norm_hs, fig, ax


def plt_corr_mask(hs, fig, ax, corr_inteval=[0, 100], corr_color=['skyblue', 'salmon']):
    import numpy as np
    from matplotlib import pyplot as plt
    import seaborn as sns
    from scipy.optimize import curve_fit
    def fit_func(x, a, b):
        return a * x + b

    max_times = np.argmax(hs, axis=0)
    max_times = max_times / 10  # Convert to seconds
    # Get the temporal firing width of each neuron, the threshold is 0.1 of the maximum firing rate
    firing_widths = np.zeros(hs.shape[1])
    for i in range(hs.shape[1]):
        firing_widths[i] = np.sum(hs[:, i] > 1E-1)
        # firing_widths[i] = np.sum(norm_hs[:, i] > 0.5 * np.max(norm_hs[:, i]))
    
    colors = [corr_color[0] if t < corr_inteval[0] or t > corr_inteval[1] else corr_color[1] for t in max_times]

    plt.scatter(max_times, firing_widths, c=colors, s=5)
    
    
    popt, pcov = curve_fit(fit_func, max_times, firing_widths)
    print(f'Fitted gradient: {popt[0]:.4f}')

    rval = np.corrcoef(max_times[(max_times >= corr_inteval[0]) & (max_times < corr_inteval[1])], 
                   firing_widths[(max_times >= corr_inteval[0]) & (max_times < corr_inteval[1])])[0,1]
    print(f'Correlation coefficient: {rval}')
    sns.regplot(x=max_times[(max_times >= corr_inteval[0]) & (max_times < corr_inteval[1])], 
            y=firing_widths[(max_times >= corr_inteval[0]) & (max_times < corr_inteval[1])],
            scatter=False, 
            color='black',
            line_kws={"linewidth":1, "linestyle":"-"},
            ci=95,
            label=f'r = {rval:.2f}, slope = {popt[0]:.2f}')

    # plt.xlabel('Maximum firing time (s)')
    plt.ylabel("Firing width (s)")
    plt.legend(loc='upper right')
    
    ax.set_xticks(np.linspace(0, hs.shape[0]/10, 6))
     
    return fig, ax