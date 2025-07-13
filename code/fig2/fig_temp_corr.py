import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns


load_data_type = '2WSMS_mask_vary'
load_dir = f'data/'

colours = ['#274753', '#297270', '#299D8F', '#8AB07C',  '#E7C66B', '#F3A361']
corr_starts = [5, 4.5, 4, 3.5, 3, 2.5]
corr_ends =   [5, 5.5, 6, 6.5, 7, 7.5]
# corr_starts = [0, 0.5, 1, 1.5, 2, 2.5]
# corr_ends =   [10, 9.5, 9, 8.5, 8, 7.5]

fig, axs = plt.subplots(figsize=(9, 10), constrained_layout=True)

gs = gridspec.GridSpec(6, 1, figure=fig, hspace=0)
for i in range(6):
    data = np.load(f'{load_dir}/{load_data_type}{i}.npy', allow_pickle=True).item()

    hs = data['hidden_states_512']
    avg_hs = np.mean(hs, axis=0)
    
    ax = fig.add_subplot(gs[i, 0])
    
    ax.set_yticks(np.linspace(0, 100, 5))
    ax.set_ylabel('Firing width (s)')
    if i < 5:
        ax.set_xticks([])
    else:
        ax.set_xlabel("Time (s)")
        
    # find each neuron's max firing time and width
    max_times = np.argmax(avg_hs, axis=0)
    # Sort neurons by the max firing time
    sort_indices = np.argsort(max_times)
    avg_hs = avg_hs[:, sort_indices]
    max_times = max_times[sort_indices]
    max_times = max_times / 10

    firing_widths = np.zeros(avg_hs.shape[1])
    for j in range(avg_hs.shape[1]):
        firing_widths[j] = np.sum(avg_hs[:, j] > 1E-1)
        # firing_widths[j] = np.sum(avg_hs[:, j] > 0.1 * np.max(avg_hs[:, j]))
    
    # Set the scatter points with corr starts and ends wiyth alpha=1, and others with alpha=0.5
    plt.scatter(max_times[max_times < corr_starts[i]], 
                firing_widths[max_times < corr_starts[i]],
                c=colours[i], s=10, alpha=0.3)
    plt.scatter(max_times[(max_times >= corr_starts[i]) & (max_times <= corr_ends[i])], 
                firing_widths[(max_times >= corr_starts[i]) & (max_times <= corr_ends[i])],
                c=colours[i], s=10, alpha=1)
    plt.scatter(max_times[max_times > corr_ends[i]], 
                firing_widths[max_times > corr_ends[i]],
                c=colours[i], s=10, alpha=0.3)
    # plt.scatter(max_times, firing_widths, c=colours[i], s=10)

    # use seaborn to plot the correlation of red dots with shaded area
    rval = np.corrcoef(max_times[(max_times >= corr_starts[i]) & (max_times < corr_ends[i])], 
                       firing_widths[(max_times >= corr_starts[i]) & (max_times < corr_ends[i])])[0,1]
    print(f'Correlation coefficient: {rval}')
    sns.regplot(x=max_times[(max_times >= corr_starts[i]) & (max_times < corr_ends[i])], 
                y=firing_widths[(max_times >= corr_starts[i]) & (max_times < corr_ends[i])],
                scatter=False, 
                color='black',
                line_kws={"linewidth":1, "linestyle":"-"},
                ci=95,
                label=f'r = {rval:.2f}')
    plt.legend()

fig.delaxes(fig.get_axes()[0])

# plt.tight_layout()
plt.savefig(f'output/fig_temp_corr_{load_data_type}')