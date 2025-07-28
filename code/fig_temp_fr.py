import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from func import plt_hs


load_data_types = ['2TS2WSMS_vary2',# '2TS2WSMS_vary4', 
                   '2TS2WSMS_vary10', #'2TS2WSMS_vary40', 
                   '2TS2WSMS_vary50', #'2TS2WSMS_vary60', '2TS2WSMS_vary80', 
                   '2TS2WSMS_vary90', 
                   #'2TS2WSMS_vary96',
                   '2TS2WSMS_vary98']

# load_data_types = ['2TS_vary0', '2TS_vary1','2TS_vary2', 
#                    '2TS_vary3', '2TS_vary4','2TS_vary5',]

# load_data_types = ['2WSMS_mask_vary0', '2WSMS_mask_vary1','2WSMS_mask_vary2', 
#                    '2WSMS_mask_vary3', '2WSMS_mask_vary4','2WSMS_mask_vary5',]

load_dir = f'data/'


fig, axs = plt.subplots(figsize=(10, len(load_data_types)*2), constrained_layout=True)

gs = gridspec.GridSpec(len(load_data_types), 1, figure=fig, hspace=0.01)
for idx, item in enumerate(load_data_types):
    data = np.load(f'{load_dir}/{item}.npy', allow_pickle=True).item()
    hs = data['hidden_states_512']
    avg_hs = np.mean(hs, axis=0)
    warmup, end = 10, avg_hs.shape[0]
    
    
    ax = fig.add_subplot(gs[idx, 0])
    norm_hs, fig, ax = plt_hs(avg_hs[warmup:end], min_fr=0.1, fig=fig, ax=ax,)
    if idx < len(load_data_types)-1:
        ax.set_xticks([])
        # ax.set_xlabel("")
    else:
        ax.set_xlabel("Time (s)", fontsize=12)
        # ax.set_xticks(np.linspace(0, 10, 11))

fig.delaxes(fig.get_axes()[0])

# plt.tight_layout()
plt.savefig(f'output/fig_temp_fr_{load_data_types[0][:-1]}')