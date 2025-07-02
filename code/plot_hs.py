from matplotlib import pyplot as plt
from func import *
import numpy as np
from rtgym.utils.data_processing import RatemapAggregator
from tqdm import tqdm
import argparse

# Input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--load_data_type', default=None, type=str)
args = parser.parse_args()
    
data = np.load(f'data/{args.load_data_type}.npy', allow_pickle=True).item()
hidden_states = data['hidden_states']
print(hidden_states[0])
batch_idx = 0
fig, ax = plt_hs(hidden_states[batch_idx], min_fr=0)
plt.savefig(f'data/{args.load_data_type}_hs_{batch_idx}.png')

# # ===========================================================================================
# # Plot place cells
# # ===========================================================================================

# aggregator = RatemapAggregator(data['arena_map'], device='cuda')

# aggregator.update(hidden_states, data['test_traj']['coords'])
# ratemap = aggregator.get_ratemap().cpu().numpy()
# print(ratemap.shape)

# # Plot the ratemap of all 512 neurons in a large subplots
# fig, ax = plt.subplots(32, 16, figsize=(64, 70), dpi=500, sharex=True, sharey=True)
# for row in tqdm(range(32)):
#     for col in range(16):
#         neuron_idx = row * 16 + col
#         if neuron_idx < ratemap.shape[0]:
#             ax[row, col].imshow(ratemap[neuron_idx], cmap='jet', aspect='auto')
#             ax[row, col].axis('off')
#         # Plot text in the center of each subplot
#         ax[row, col].text(0.5, 0.5, str(neuron_idx), 
#                           horizontalalignment='center', verticalalignment='center', fontsize=5)
# plt.tight_layout()
# plt.savefig(f'data/{load_data_type}_ratemap.png')

# # Plot place cells
# fig, ax = plt.subplots(2, 6, figsize=(8, 2.5), dpi=300, sharex=True, sharey=True)
# place_cell_idx = [170, 204, 233, 234, 259, 284, 309, 313, 324, 377, 423, 438]
# for i, idx in enumerate(place_cell_idx):
#     ax[i // 6, i % 6].imshow(ratemap[idx], cmap='jet', aspect='auto')
#     ax[i // 6, i % 6].text(0.8, 0.8, str(idx), 
#                           # horizontalalignment='center', verticalalignment='center',
#                            fontsize=10)
#     # ax[i // 6, i % 6].set_title(f'Neuron {idx}')
#     ax[i // 6, i % 6].axis('off')
# plt.tight_layout()
# plt.savefig(f'data/{load_data_type}_place_cells.png')