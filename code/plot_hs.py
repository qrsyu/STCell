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
# Plot place cells
# ===========================================================================================

aggregator = RatemapAggregator(data['arena_map'], device='cuda')

aggregator.update(hidden_states, data['test_traj']['coords'])
ratemap = aggregator.get_ratemap().cpu().numpy()
print(ratemap.shape)

# Convert all zero to nan
ratemap[ratemap == 0] = np.nan

save_dir = f'result/{args.load_data_type}_{args.num_neuron}_ratemap/'
os.makedirs(save_dir, exist_ok=True)

# Only plot those with np.count_nonzero(np.nan_to_num(ratemap[x])) >= 100
select_indices = [i for i in range(ratemap.shape[0]) if np.count_nonzero(np.nan_to_num(ratemap[i])) >= 120]
# Plot the ratemap
for imap in tqdm(select_indices):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=300)  
    ax.imshow(ratemap[imap], cmap='jet', aspect='auto')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/ratemap_neuron_{imap}', transparent=True)
    plt.close(fig)
    
# Print the ratio of place cells
print(f'number of place cells: {len(select_indices)} / {args.num_neuron}')
print(f"percentage of place cells: {len(select_indices) / ratemap.shape[0] * 100:.2f}%")

# Plot a summation of place cells angular contributions
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

save_dir = f'result/{args.load_data_type}_{args.num_neuron}_temporal/'
os.makedirs(save_dir, exist_ok=True)

print(hidden_states.shape)

for ihs in tqdm(range(hidden_states.shape[0])):
    _, fig, ax = plt_hs(hidden_states[ihs], min_fr=0.01)
    plt.savefig(f'{save_dir}/temporal_batch_{ihs}.png', transparent=True)
    plt.close(fig)