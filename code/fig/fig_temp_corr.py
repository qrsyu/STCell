import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from code.func import plt_temp_corr


load_data_type = '2TS2WSMS_vary'
load_dir = f'data/'

colours = [# '#274753',
           '#297270', '#299D8F', '#8AB07C',  '#E7C66B', '#F3A361']


warmup = 10 # time points

# # Fig 3
# corr_starts = [5, 4.5, 4, 3.5, 3, 2.5] # Sec
# corr_ends   = [5, 5.5, 6, 6.5, 7, 7.5] # Sec

# Fig 4
corr_starts = [2.5] * 5 # Sec
corr_ends =   [6] * 5 # Sec

# Refine
corr_starts = [item - warmup/10 for item in corr_starts]
corr_ends   = [item - warmup/10 for item in corr_ends]


fig, axs = plt.subplots(figsize=(5, 5*2), constrained_layout=True)

gs = gridspec.GridSpec(5, 1, figure=fig, hspace=0)
for idx, item in enumerate([2, 10, 50, 90, 98]):
# for idx, i in enumerate(range(6)):
    data = np.load(f'{load_dir}/{load_data_type}{item}.npy', allow_pickle=True).item()

    hs = data['hidden_states_512']
    avg_hs = np.mean(hs, axis=0)
    avg_hs = avg_hs[warmup:, :]
    
    ax = fig.add_subplot(gs[idx, 0])
    
    _, ax = plt_temp_corr(avg_hs, fig=fig, ax=ax, corr_inteval=[corr_starts[idx], corr_ends[idx]],
                          corr_color=['silver', colours[idx],])
    
    if idx < len(colours)-1:
        ax.set_xticks([])
        # ax.set_xlabel("")
    else:
        ax.set_xlabel("Time (s)", fontsize=12)
    plt.legend()

fig.delaxes(fig.get_axes()[0])

# plt.tight_layout()
plt.savefig(f'code/fig/fig_temp_corr_{load_data_type}', dpi=500, transparent=True)