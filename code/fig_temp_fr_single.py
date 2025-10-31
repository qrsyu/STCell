import numpy as np
from matplotlib import pyplot as plt
from func import plt_hs, plt_temp_corr



load_data_type = 'STcorr'
num_neuron = 512


load_dir = f'data/'
data = np.load(f'{load_dir}/{load_data_type}.npy', allow_pickle=True).item()
hs = data[f'hidden_states_{num_neuron}']
# If place cell, STcorr, select batch
# -----------------------------
test_traj = data['test_traj']
select_batch = np.where(np.all(test_traj['coords'][:, int(20*10*0.75):int(20*10*0.8), 1] > 100, 
                               axis=1))[0]
print(select_batch)
avg_hs = hs[2]
# If time task, use average
# -----------------------------
# avg_hs = np.mean(hs, axis=0)
print(avg_hs.shape)

warmup, end = 10, avg_hs.shape[0]


# Plot the temporal representations
fig, ax = plt.subplots(figsize=(4, 3), constrained_layout=True)
norm_hs, fig, ax = plt_hs(avg_hs[warmup:end], min_fr=0.1, fig=fig, ax=ax)
plt.savefig(f'output/fig_temp_fr_{load_data_type}_{num_neuron}')


# # Plot the correlation
# print(norm_hs.shape)
# fig, ax = plt.subplots(figsize=(4, 3), constrained_layout=True)
# fig, ax = plt_temp_corr(norm_hs, fig, ax, corr_inteval=[0, 11])
# plt.savefig(f'output/fig_temp_corr_{load_data_type}')