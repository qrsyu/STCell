import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from func import plt_hs

load_data_type = '2WSMS_mask_vary'
load_dir = f'data/'


# def plt_hs(hs, min_fr=0.1, fig=None, ax=None, figsize=(4,3)):
    
#     from matplotlib import pyplot as plt
#     import numpy as np

#     time_points, num_neurons = hs.shape[0], hs.shape[1]

#     # Select neurons with max firing rate > 0.1
#     max_fr = hs.max(axis=0)
#     # Get the index where max_fr > min_fr
#     mask = max_fr > min_fr
#     neuron_indices = np.where(mask)[0]
#     select_hs = hs[:, neuron_indices]
#     del hs

#     # Normalize the hs along time points (Sure to be correct!)
#     norm_hs = select_hs / np.linalg.norm(select_hs, axis=0, keepdims=True)
#     del select_hs
    
#     # Sort neurons from maximum firing time
#     max_time = np.argmax(norm_hs, axis=0)
#     sorted_neuron_indices = np.argsort(max_time)
#     norm_hs = norm_hs[:, sorted_neuron_indices]

#     # Plot the normalized hs
#     # x_ticks = np.arange(0, time_points)
#     ax.imshow(norm_hs.T, aspect='auto', cmap='jet', extent=[0, 10, 0, 100]
#               )
#     # ax.set_xlabel('Time Points')
#     # ax.set_ylabel('Neurons')
#     # plt.tight_layout()
#     # Set y axis limit
#     # ax.set_ylim(300, 200)
#     return norm_hs, fig, ax


fig, axs = plt.subplots(figsize=(9, 10), constrained_layout=True)

gs = gridspec.GridSpec(6, 1, figure=fig, hspace=0.01)
for idx, i in enumerate(range(6)):
    print(f'{load_dir}/{load_data_type}{i}.npy')
    data = np.load(f'{load_dir}/{load_data_type}{i}.npy', allow_pickle=True).item()
    hs = data['hidden_states_512']
    avg_hs = np.mean(hs, axis=0)
    
    
    # # Remove the first 50 neurons
    # if i in [1, 2, 3, 4]:
    #     avg_hs = avg_hs[:, :-50]
    
    
    ax = fig.add_subplot(gs[idx, 0])
    norm_hs, fig, ax = plt_hs(avg_hs, min_fr=0.1, fig=fig, ax=ax,)
    # ax.set_yticks([])
    ax.set_ylabel('Neurons')
    # ax.set_xticks([])
    if i < 5:
        ax.set_xticks([])
        ax.set_xlabel("")
    else:
        ax.set_xlabel("Time (s)")
        ax.set_xticks(np.linspace(0, 10, 11))

fig.delaxes(fig.get_axes()[0])

# plt.tight_layout()
plt.savefig(f'output/fig_temp_fr_{load_data_type}')