import numpy as np
from matplotlib import pyplot as plt
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from func import plt_hs, plt_corr, time_analysis

load_data_type = '2WSMS_mask'
data = np.load(f'data/{load_data_type}.npy', allow_pickle=True).item()
num_neuron = 512
hidden_states = data[f'hidden_states_{num_neuron}']  

# Averge accross the batch
avg_hs = np.mean(hidden_states, axis=0)
print(avg_hs.shape)

fig, ax = plt.subplots(figsize=(5, 4))
norm_hs, fig, ax, select_idx = plt_hs(avg_hs, min_fr=0.1, fig=fig, ax=ax, return_idx=True)

is_time_cell = time_analysis(hidden_states[:, :, select_idx])
time_cell_global_idx = select_idx[is_time_cell]

# Sort the norm_hs with maximum firing time
max_time_pts = np.argmax(norm_hs, axis=0)

threshold = 0.7
delta = 0.1

# For regions left to the max time pts, find the index where the firing rate is closest to 0.7 and between 0.65 and 0.75
firing_starts = np.zeros(norm_hs.shape[1])
for i in range(norm_hs.shape[1]):
    left_half = norm_hs[:max_time_pts[i], i]
    if np.any((left_half > threshold-delta) & (left_half < threshold+delta)):
        firing_starts[i] = np.where((left_half > threshold-delta) & (left_half < threshold+delta))[0][-1]
        
    else:
        firing_starts[i] = np.nan

# For regions right to the max time pts, find the index where the firing rate is closest to 0.7 and between 0.65 and 0.75
firing_ends = np.zeros(norm_hs.shape[1])
for i in range(norm_hs.shape[1]):
    right_half = norm_hs[max_time_pts[i]:, i]
    if np.any((right_half > threshold-delta) & (right_half < threshold+delta)):
        firing_ends[i] = max_time_pts[i] + np.where((right_half > threshold-delta) & (right_half < threshold+delta))[0][0]
    else:
        firing_ends[i] = np.nan

firing_widths = firing_ends - firing_starts

# Change the unit to time (s)
max_time_pts = max_time_pts / 10
firing_starts = firing_starts / 10
firing_ends = firing_ends / 10
firing_widths = firing_widths / 10


# ============================================================================
# Plot the firing width vs time for the first half
# ============================================================================

fig, axs = plt.subplots(2, 1, figsize=(3.5, 5), sharex=True)  
_, fig, axs[0], select_idx_1 = plt_hs(norm_hs[:, (firing_starts >= 0) & (firing_ends <= 5)], 
                    min_fr=0.1, fig=fig, ax=axs[0], return_idx=True)


norm_hs_idx_1 = np.where((firing_starts >= 0) & (firing_ends <= 5))[0]
global_idx_1 = select_idx[norm_hs_idx_1[select_idx_1]]
is_tc_1 = time_analysis(hidden_states[:, :50, global_idx_1])
time_cell_global_1 = global_idx_1[is_tc_1]

data[f'time_cell_{num_neuron}_0-50'] = time_cell_global_1

# Draw dashed lines at firing starts and ends
dashed_starts, dashed_ends, y = [], [], []
for idx, i in enumerate(norm_hs_idx_1):
    if (firing_starts[i] >= 0) and (firing_ends[i] <= 5):
        dashed_starts.append(firing_starts[i])
        dashed_ends.append(firing_ends[i])
        y.append(idx)
# axs[0].plot(dashed_starts[::-1], y, color='black', linestyle='--', linewidth=0.5)
# axs[0].plot(dashed_ends[::-1], y, color='black', linestyle='--', linewidth=0.5)
axs[0].set_xlim(0, 5)

axs[1].scatter(max_time_pts[(firing_starts >= 0) & (firing_ends <= 5)], 
            firing_widths[(firing_starts >= 0) & (firing_ends <= 5)], 
            c='salmon', s=10)
plt_corr(max_time_pts[(firing_starts >= 0) & (firing_ends <= 5)], 
         firing_widths[(firing_starts >= 0) & (firing_ends <= 5)], 
         fig=fig, ax=axs[1])
plt.legend()
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel("Firing width (s)")
plt.tight_layout()
plt.savefig(f'code/spacetime_exp/firing_width_vs_time(RealModel-{load_data_type})_part1.png', dpi=500, transparent=False)



# ============================================================================
# Plot the firing width vs time for the second half
# ============================================================================

fig, axs = plt.subplots(2, 1, figsize=(3.5, 5), sharex=True)
_, fig, axs[0], select_idx_2 = plt_hs(norm_hs[:, (firing_starts >= 5) & (firing_ends <= 10)], 
                    min_fr=0.1, fig=fig, ax=axs[0], return_idx=True)

norm_hs_idx_2 = np.where((firing_starts >= 5) & (firing_ends <= 10))[0]
global_idx_2 = select_idx[norm_hs_idx_2[select_idx_2]]
is_tc_2 = time_analysis(hidden_states[:, 50:, global_idx_2])
time_cell_global_2 = global_idx_2[is_tc_2]

data[f'time_cell_{num_neuron}_50_-10'] = time_cell_global_2
np.save(f'data/{load_data_type}.npy', data)

# Draw dashed lines at firing starts and ends
dashed_starts, dashed_ends, y = [], [], []
for idx, i in enumerate(norm_hs_idx_2):
    if (firing_starts[i] >= 5) and (firing_ends[i] <= 10):
        dashed_starts.append(firing_starts[i])
        dashed_ends.append(firing_ends[i])
        y.append(idx)
# axs[0].plot(dashed_starts[::-1], y, color='black', linestyle='--', linewidth=0.5)
# axs[0].plot(dashed_ends[::-1], y, color='black', linestyle='--', linewidth=0.5)
axs[0].set_xlim(5, 10)

axs[1].scatter(max_time_pts[(firing_starts >= 5) & (firing_ends <= 10)], 
            firing_widths[(firing_starts >= 5) & (firing_ends <= 10)], 
            c='salmon', s=10)
plt_corr(max_time_pts[(firing_starts >= 5) & (firing_ends <= 10)], 
         firing_widths[(firing_starts >= 5) & (firing_ends <= 10)], 
         fig=fig, ax=axs[1])
plt.legend()
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel("Firing width (s)")
plt.tight_layout()
plt.savefig(f'code/spacetime_exp/firing_width_vs_time(RealModel-{load_data_type})_part2.png', dpi=500, transparent=False)

# Find the common index
place_cell_glob1 = data['place_cells_512_0_50']
place_cell_glob2 = data['place_cells_512_50_-1']
time_cell_glob1 = data['time_cell_512_0-50']
time_cell_glob2 = data['time_cell_512_50_-10']
# Find how many time cells are also place cells
common_idx1 = np.intersect1d(place_cell_glob1, time_cell_glob1)
common_idx2 = np.intersect1d(place_cell_glob2, time_cell_glob2)
print(f'Number of place cells, time cells, common cells:')
print(f'First lap:  {len(place_cell_glob1)}, {len(time_cell_glob1)}, {len(common_idx1)}')
print(f'Second lap: {len(place_cell_glob2)}, {len(time_cell_glob2)}, {len(common_idx2)}')

# The intersect of two place fields
common_place = np.intersect1d(place_cell_glob1, place_cell_glob2)
# The intersect of two time fields
common_time = np.intersect1d(time_cell_glob1, time_cell_glob2)
print(f'Number of common place cells: {len(common_place)}') 
print(f'Number of common time cells: {len(common_time)}')

P1 = set(place_cell_glob1)
T1 = set(time_cell_glob1)
P2 = set(place_cell_glob2)
T2 = set(time_cell_glob2)

print('P1 ∩ T2:', len(P1 & T2))
print('T1 ∩ P2:', len(T1 & P2))
print('P1 ∩ T1 ∩ P2:', len(P1 & T1 & P2))
print('P1 ∩ T1 ∩ T2:', len(P1 & T1 & T2))
print('P1 ∩ P2 ∩ T2:', len(P1 & P2 & T2))
print('T1 ∩ P2 ∩ T2:', len(T1 & P2 & T2))
print('P1 ∩ T1 ∩ P2 ∩ T2:', len(P1 & T1 & P2 & T2))