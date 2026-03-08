from matplotlib import pyplot as plt
from func import compute_occupancy, ratemap_to_angle_profile, SIC_analysis, plt_hs
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
parser.add_argument('--theory',         default='',   type=str)
# Optional: select half of the time points
parser.add_argument('--time_start',     default=0,    type=int)
parser.add_argument('--time_end',       default=-1,   type=int)
args = parser.parse_args()


data = np.load(f'data/{args.load_data_type}.npy', allow_pickle=True).item()
hidden_states = data[f'{args.theory}hidden_states_{args.num_neuron}']

select_hs = hidden_states[:, args.time_start:args.time_end, :]
select_coords = data['test_traj']['coords'][:, args.time_start:args.time_end, :]

# ===========================================================================================
# Calculate the occupancy map
# ===========================================================================================

occupancy = compute_occupancy(select_coords, bins=data['arena_map'].shape)
print('Occupancy has the shape of the arena: ', occupancy.shape)

# Plot the occupancy (有用)
# --------------------------------------------------------------------------------------------
fig, ax = plt.subplots(1, 1, figsize=(4.9, 4))
cbar = plt.colorbar(ax.imshow(occupancy, cmap='jet', aspect='auto'), 
                    ax=ax, fraction=0.046, pad=0.04, orientation='vertical', location='left')
cbar.set_label('Occupancy', rotation=270, labelpad=20)
cbar.ax.tick_params(labelsize=12)  # Set colorbar tick label size
# Move colorbar ticks to the left
cbar.ax.yaxis.set_ticks_position('left')
cbar.ax.yaxis.set_label_position('left')
ax.imshow(occupancy, cmap='jet', aspect='auto')
ax.axis('off')
plt.tight_layout()
save_dir = f'fig-place-cells/{args.load_data_type}_{args.num_neuron}/{args.theory}ratemap_time{args.time_start}-{args.time_end}/'
os.makedirs(save_dir, exist_ok=True)
plt.savefig(f'{save_dir}/occupancy.png', transparent=True)
plt.close(fig)

# ===========================================================================================
# Compute the ratemap
# ===========================================================================================

aggregator = RatemapAggregator(data['arena_map'], device='cuda')

aggregator.update(select_hs, select_coords)
ratemap = aggregator.get_ratemap().cpu().numpy()
print(ratemap.shape)

# ===========================================================================================
# Calculate the SIC
# ===========================================================================================

ratemap_angles, angles, radius = ratemap_to_angle_profile(ratemap)
occupancy_angles, _, _ = ratemap_to_angle_profile(occupancy[None, :, :])
SIC, place_cells = SIC_analysis(ratemap_angles, occupancy_angles, threshold=8)
print(f'Number of place cells: {np.sum(place_cells)} / {args.num_neuron}')
select_indices = np.where(place_cells)[0]

# ===========================================================================================
# Plot the ratemap
# ===========================================================================================

# Convert all zero to nan
ratemap[ratemap == 0] = np.nan

# Plot the ratemap
# select_indices = range(args.num_neuron)
for imap in tqdm(select_indices):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=300)  
    ax.imshow(ratemap[imap], cmap='jet', aspect='auto')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/ratemap_neuron_{imap}', transparent=True)
    plt.close(fig)

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

# # ===========================================================================================
# # Plot temporal firing rates
# # ===========================================================================================

# # Average across the batch
# avg_hs = np.mean(hidden_states, axis=0)
# print(avg_hs.shape)

# fig, ax = plt.subplots(figsize=(10, 6))
# norm_hs, fig, ax = plt_hs(avg_hs, min_fr=0.0, fig=fig, ax=ax)
# plt.savefig(f'{save_dir}/temporal_batch_avg.png', transparent=True)
# plt.close(fig),