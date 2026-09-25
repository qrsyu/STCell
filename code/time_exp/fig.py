import numpy as np
from matplotlib import pyplot as plt
import sys, os
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from func import plt_hs, plt_corr, time_analysis

# Only change these
name = '_b'
time_start, time_end = 3, 12.5



data = np.load(f'data/2TS{name}.npy', allow_pickle=True).item()
hidden_states = data['test_hidden_states']
if isinstance(hidden_states, torch.Tensor):
    hidden_states = hidden_states.detach().cpu().numpy()
print('The shape of hidden_states:', hidden_states.shape)
    

    
is_time_cell = time_analysis(hidden_states)
print('The number of time cells:', np.sum(is_time_cell), 'out of', len(is_time_cell), 'cells')



# Averge accross the batch
avg_hs = np.mean(hidden_states, axis=0)
print(avg_hs.shape)

# Plot the sorted hidden states
fig, ax = plt.subplots(figsize=(4, 3))
norm_hs, fig, ax = plt_hs(avg_hs, min_fr=0.1, fig=fig, ax=ax)
ax.axvspan(2.5, 3,   alpha=0.5, color='white', zorder=0)
ax.axvspan(17.5, 18, alpha=0.5, color='white', zorder=0)
# ax.set_xlim(2, 20)
ax.set_xlabel('Time (s)')
plt.savefig(f'code/time_exp/time_exp_fr{name}.png', dpi=500, transparent=False, bbox_inches='tight')



# Sort the norm_hs with maximum firing time
max_time_pts = np.argmax(norm_hs, axis=0)

threshold = 0.5
delta = 0.1

# For regions left to the max time pts, find the index where the firing rate is closest to 0.5 and between 0.6 and 0.4
firing_starts = np.zeros(norm_hs.shape[1])
for i in range(norm_hs.shape[1]):
    left_half = norm_hs[:max_time_pts[i], i]
    if np.any((left_half > threshold-delta) & (left_half < threshold+delta)):
        firing_starts[i] = np.where((left_half > threshold-delta) & (left_half < threshold+delta))[0][-1]
    else:
        firing_starts[i] = np.nan

# For regions right to the max time pts, find the index where the firing rate is closest to 0.5 and between 0.6 and 0.4
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

# Plot the firing widths vs the maximum firing times
fig2, ax2 = plt.subplots(figsize=(4,3))

ax2.scatter(max_time_pts[(time_start <= max_time_pts) & (max_time_pts < time_end)], 
            firing_widths[(time_start <= max_time_pts) & (max_time_pts < time_end)], c='salmon', s=10)
ax2.scatter(max_time_pts[(time_start <= max_time_pts) & (max_time_pts > time_end)], 
            firing_widths[(time_start <= max_time_pts) & (max_time_pts > time_end)], c='skyblue', s=10)

# Plot where not nan firing widths
not_nan_mask = ~np.isnan(firing_widths)
max_time_pts = max_time_pts[not_nan_mask]
firing_widths = firing_widths[not_nan_mask]
firing_starts = firing_starts[not_nan_mask]
firing_ends = firing_ends[not_nan_mask]
norm_hs = norm_hs[:, not_nan_mask]



fig2, ax2 = plt_corr(max_time_pts[(time_start <= max_time_pts) & (max_time_pts < time_end)], 
            firing_widths[(time_start <= max_time_pts) & (max_time_pts < time_end)], fig=fig2, ax=ax2)
ax2.legend()

ax2.axvspan(2.5, 3,   alpha=0.5, color='grey', zorder=0)
ax2.axvspan(17.5, 18, alpha=0.5, color='grey', zorder=0)

# ax2.set_xlim(2.5, 20)
ax2.set_xlabel('Peak firing time (s)')
ax2.set_ylabel("Firing width (s)")
fig2.tight_layout()
fig2.savefig(f'code/time_exp/time_exp_temp_corr{name}.png', dpi=500, transparent=False, bbox_inches='tight')