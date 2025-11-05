import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from code.func import plt_hs



### 2TS2WSMS_vary
# ----------------------------------------------------------------------------
load_data_types = ['2TS2WSMS_vary2',# '2TS2WSMS_vary4', 
                   '2TS2WSMS_vary10', #'2TS2WSMS_vary40', 
                   '2TS2WSMS_vary50', #'2TS2WSMS_vary60', '2TS2WSMS_vary80', 
                   '2TS2WSMS_vary90', 
                   #'2TS2WSMS_vary96',
                   '2TS2WSMS_vary98']
masks = None
# ----------------------------------------------------------------------------

# ### 2TS_vary
# ----------------------------------------------------------------------------
# load_data_types = ['2TS_vary0', '2TS_vary1','2TS_vary2', 
#                    '2TS_vary3', '2TS_vary4','2TS_vary5',]
# masks = [[[2.5, 3], [7.5, 8]], [[2, 3], [7, 8]], [[1.5, 3.5], [6.5, 8.5]],
#          [[1, 4], [6, 9]], [[0.5, 4.5], [5.5, 9.5]], [[0, 5], [5, 10]],]
# ----------------------------------------------------------------------------

### 2WSMS_vary
# ----------------------------------------------------------------------------
# load_data_types = ['2WSMS_mask_vary0', '2WSMS_mask_vary1','2WSMS_mask_vary2', 
#                    '2WSMS_mask_vary3', '2WSMS_mask_vary4','2WSMS_mask_vary5',]
# masks = [[[0, 5], [5, 10]], [[0.5, 4.5], [5.5, 9.5]], [[1, 4], [6, 9]],
#          [[1.5, 3.5], [6.5, 8.5]], [[2, 3], [7, 8]], [[2.5, 3], [7.5, 8]],]
# ----------------------------------------------------------------------------



warmup = 10
# Reduce all values in masks by warmup, if less than 0, set to 0
if masks is not None:
    for i in range(len(masks)):
        for j in range(len(masks[i])):
            masks[i][j][0] = max(0, masks[i][j][0]-warmup/10)
            masks[i][j][1] = max(0, masks[i][j][1]-warmup/10)



load_dir = f'data/'

fig, axs = plt.subplots(figsize=(5, len(load_data_types)*2), constrained_layout=True)

gs = gridspec.GridSpec(len(load_data_types), 1, figure=fig, hspace=0.01)
for idx, item in enumerate(load_data_types):
    data = np.load(f'{load_dir}/{item}.npy', allow_pickle=True).item()
    hs = data['hidden_states_512']
    avg_hs = np.mean(hs, axis=0)
    end = avg_hs.shape[0]
    
    
    ax = fig.add_subplot(gs[idx, 0])
    norm_hs, fig, ax = plt_hs(avg_hs[warmup:end], # masks=masks[idx],
                              min_fr=0.1, fig=fig, ax=ax,)
    if idx < len(load_data_types)-1:
        ax.set_xticks([])
        # ax.set_xlabel("")
    else:
        ax.set_xlabel("Time (s)", fontsize=12)
        # ax.set_xticks(np.linspace(0, 10, 11))

fig.delaxes(fig.get_axes()[0])

# plt.tight_layout()
plt.savefig(f'code/fig/fig_temp_fr_{load_data_types[0][:-1]}', transparent=True, dpi=500)