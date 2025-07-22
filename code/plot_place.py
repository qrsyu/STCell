from matplotlib import pyplot as plt
from func import *
import numpy as np
from rtgym.utils.data_processing import RatemapAggregator
from rtgym.utils.visualization import plot_ratemaps
from tqdm import tqdm
import argparse
import os 

# Input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--load_data_type', default=None, type=str)
parser.add_argument('--num_neuron',     default=512,  type=int)
args = parser.parse_args()
    
data = np.load(f'data/{args.load_data_type}.npy', allow_pickle=True).item()
hidden_states = data[f'hidden_states_{args.num_neuron}']

# ===========================================================================================
# Calculate the occupancy map
# ===========================================================================================

occupancy = compute_coarse_occupancy(data['test_traj']['coords'], 
                                     old_bins=data['arena_map'].shape,
                                     new_bins=(36, 36))
print('Occupancy has the shape of the arena: ', occupancy.shape)

# # Plot the occupancy (有用)
# --------------------------------------------------------------------------------------------
# fig, ax = plt.subplots(1, 1, figsize=(4.9, 4))
# cbar = plt.colorbar(ax.imshow(occupancy, cmap='jet', aspect='auto'), 
#                     ax=ax, fraction=0.046, pad=0.04, orientation='vertical', location='left')
# cbar.set_label('Occupancy', rotation=270, labelpad=20)
# cbar.ax.tick_params(labelsize=12)  # Set colorbar tick label size
# # Move colorbar ticks to the left
# cbar.ax.yaxis.set_ticks_position('left')
# cbar.ax.yaxis.set_label_position('left')
# ax.imshow(occupancy, cmap='jet', aspect='auto')
# ax.axis('off')
# plt.tight_layout()
# plt.savefig(f'{args.load_data_type}_{args.num_neuron}_occupancy.png', transparent=True)
# plt.close(fig)

# ===========================================================================================
# Compute the ratemap
# ===========================================================================================

aggregator = RatemapAggregator(data['arena_map'], device='cuda')

aggregator.update(hidden_states, data['test_traj']['coords'])
ratemap = aggregator.get_ratemap().cpu().numpy()
print(ratemap.shape)

# Resize ratemap
coa_ratemap = coarse_ratemap(ratemap, new_bins=(36, 36))
print(coa_ratemap.shape)

# ===========================================================================================
# Calculate the SIC
# ===========================================================================================

spatial_infos, decisions = SIC_analysis(coa_ratemap, occupancy)
print('The ratio of place cells is: ', np.sum(decisions) / len(decisions))
print('The number of place cells is: ', np.sum(decisions))

# ===========================================================================================
# Plot the ratemap
# ===========================================================================================

save_dir = f'result/{args.load_data_type}_{args.num_neuron}_ratemap/'
os.makedirs(save_dir, exist_ok=True)

# # Convert all zero to nan
# ratemap[ratemap == 0] = np.nan

# # Only plot those with decisions = True
# # select_indices = [i for i in range(ratemap.shape[0]) if decisions[i]]
select_indices = range(args.num_neuron)
# # Plot the ratemap
# for imap in tqdm(select_indices):
#     fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=300)  
#     ax.imshow(ratemap[imap], cmap='jet', aspect='auto')
#     ax.axis('off')
#     plt.tight_layout()
#     plt.savefig(f'{save_dir}/ratemap_neuron_{imap}', transparent=True)
#     plt.close(fig)

# ===========================================================================================
# Plot a summation of place cells angular contributions
# ===========================================================================================

ratemap_sum = np.nansum(ratemap[select_indices], axis=0)
ratemap_sum[ratemap_sum == 0] = np.nan

fig, ax = plt.subplots(1, 1, figsize=(4.9, 4))  
cbar = plt.colorbar(ax.imshow(ratemap_sum, cmap='jet', aspect='auto'), 
                    ax=ax, fraction=0.046, pad=0.04, orientation='vertical', location='left')
cbar.set_label('Firing Rate (Hz)', rotation=270, labelpad=20)
cbar.ax.tick_params(labelsize=12)  # Set colorbar tick label size
# Move colorbar ticks to the left
cbar.ax.yaxis.set_ticks_position('left')
cbar.ax.yaxis.set_label_position('left')
ax.imshow(ratemap_sum, cmap='jet', aspect='auto')
ax.axis('off')
plt.tight_layout()
plt.savefig(f'{save_dir}/ratemap_sum', transparent=True)
plt.close(fig)

# ===========================================================================================
# Plot temporal firing rates
# ===========================================================================================

# save_dir = f'result/{args.load_data_type}_{args.num_neuron}_temporal/'
# os.makedirs(save_dir, exist_ok=True)

# Average across the batch
avg_hs = np.mean(hidden_states, axis=0)
print(avg_hs.shape)

norm_hs, fig, ax = plt_hs(avg_hs, min_fr=0.1)
plt.savefig(f'{save_dir}/temporal_batch_avg.png', transparent=True)
plt.close(fig)