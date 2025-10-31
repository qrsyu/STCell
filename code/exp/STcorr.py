"""The spatial and temporal signal correlation experiment."""

import os
import torch
from rtgym import RatatouGym
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

# ===========================================================================================
# Set up the arena and SPATIAL input
# ===========================================================================================

load_data_type = 'STcorr'

temp_reso, spat_reso = 100, 1  # Temp reso: 500ms; Spatial reso: 1cm
gym = RatatouGym(temporal_resolution=temp_reso, spatial_resolution=spat_reso)

gym.init_arena_map(shape='two_rooms')

vel_mean, vel_std = 10, 2
behavior_profile = {
                    "name":                   "random_explore",
                    "type":                   "predefined",
                    "velocity_mean":          vel_mean,
                    "velocity_sd":            vel_std,
                    "random_drift_magnitude": 0.05,
                    "switch_direction_prob":  0.05,
                    "switch_velocity_prob":   0.1,
                    'avoid_boundary_dist':    60
                    }

n_space_cells = 50

sensory_profile = {
                    "wsm": {
                            "type":      "weak_sm_cell",
                            "n_cells":   n_space_cells,
                            "sigma":     2,
                            "magnitude": 4,
                            "normalize": True
                            },
                    }

# Set the sensory and behavior profiles
gym.set_sensory_from_profile(sensory_profile)
gym.set_behavior_from_profile(behavior_profile)

arena_map = gym.arena_map

duration = 20 # sec
gym.trial.new_trial(duration=duration, batch_size=1000, init_pos=40)

# Get some specific responses within a time range, the key should be the same as the sensory profile
space_res = gym.trial.get_responses(keys='wsm')
print('space responses:', space_res.shape)

traj = gym.trial.get_traj()
traj['coords'] = traj['coords_float'] 
traj['hds'] = traj['head_directions']
traj['disps'] = traj['displacements']
del traj['coords_float']
del traj['head_directions']
del traj['displacements']
print(traj.keys())

# ===========================================================================================
# TEMPORAL input
# ===========================================================================================

n_time_cells = 50
np.random.seed(42)  # For reproducibility
temp_event_1 = np.random.normal(loc=4.0, scale=0.5, size=(n_time_cells,))
temp_event_2 = np.random.normal(loc=5.0, scale=1.0, size=(n_time_cells,))

sensory_profile = {
                    "time": {
                            "type":        "time_cell",
                            "n_cells":     n_time_cells,
                            "event_onset": [0.25,],
                            "event_onset_sigma": [0.01,],
                            "event_width": [0.05,],
                            "temp_events": [temp_event_1,],
                            "sigma":        0.5,    # sigma of Gaussian noise
                            "ssigma":       0.2,    # sigma of Gaussian noise smoothing (in sec)
                            "bias":         0.0   
                            },
                    }

# Set the sensory and behavior profiles
gym.set_sensory_from_profile(sensory_profile)
gym.set_behavior_from_profile(behavior_profile)

# Generate (Batch size) trial
gym.trial.new_trial(duration=duration, batch_size=1000)

# Get some specific responses within a time range, the key should be the same as the sensory profile
time_res = gym.trial.get_responses(keys='time')
print('time responses:', time_res.shape)

_ = gym.trial.vis_sensory()

# Get the batch idx where traj has x coords > 100 during t in [375:400]
select_batch = np.where(np.all(traj['coords'][:, int(duration*10*0.75):int(duration*10*0.8), 1] > 100, 
                               axis=1))[0]
print(len(select_batch))

# Give those time res a second pulse at t in [375:400]
sensory_profile = {
                    "time": {
                            "type":        "time_cell",
                            "n_cells":     n_time_cells,
                            "event_onset": [0.75],
                            "event_onset_sigma": [0.01],
                            "event_width": [0.05],
                            "temp_events": [temp_event_2],
                            "sigma":       0.5,    # sigma of Gaussian noise
                            "ssigma":      0.2,    # sigma of Gaussian noise smoothing (in sec)
                            "bias":        0.0
                           },
                   }

# Set the sensory and behavior profiles
gym.set_sensory_from_profile(sensory_profile)
gym.set_behavior_from_profile(behavior_profile)

# Generate (Batch size) trial
gym.trial.new_trial(duration=duration, batch_size=len(select_batch))

# Get some specific responses within a time range, the key should be the same as the sensory profile
time_res2 = gym.trial.get_responses(keys='time')
print('time responses:', time_res2.shape)

# Add time_res2 to time_res with select_batch idx
time_res[select_batch] = time_res[select_batch] + time_res2
print(time_res[select_batch].shape, time_res2.shape)
# # Plot the time_res
# idx = select_batch[0]
# fig, axs = plt.subplots(1, 1, figsize=(6, 4))
# for i in range(time_res.shape[-1]):
#   axs.plot(range(time_res.shape[1]), time_res[idx,:,i], # marker='o',
#               linestyle='-', linewidth=1)
# axs.set_title('Time Response')
# plt.show()

# ===========================================================================================
# Make input and label
# ===========================================================================================

space_labels = space_res.copy()
space_inputs = space_res.copy()

from rtgym.utils.masking import Masking
mask = Masking(
                m_max=0.2,  # Maximum masking ratio
                m_min=0.1,   # Minimum masking ratio
                sigma_t=1.0,  # Temporal smoothing
                sigma_d=1.0,  # Spatial smoothing
                t_warmup=10,   # Number of initial time steps to remain unmasked
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
            )

# Randomly mask the data
space_inputs = mask.mask(space_inputs).cpu().numpy()

time_labels = time_res.copy()
time_inputs = time_res.copy()

time_inputs[:, int(duration*10*0.35):, :] = 0 # 30% of the entire time series

st_inputs = np.concatenate([space_inputs, time_inputs], axis=2)
st_labels = np.concatenate([space_labels, time_labels], axis=2)

print('spacetime responses:', st_inputs.shape, st_labels.shape)

# Split the data to training and test set along axis=1
indices = np.arange(st_inputs.shape[0])
train_inputs, test_inputs, train_labels, test_labels, train_indices, \
test_indices = train_test_split(st_inputs, st_labels, indices, test_size=0.05, random_state=42)

print('train_inputs:', train_inputs.shape)
print('train_labels:', train_labels.shape)
print('test_inputs:', test_inputs.shape)
print('test_labels:', test_labels.shape)

train_traj, test_traj = {}, {}
for key, val in traj.items():
    train_traj[key] = val[train_indices]
    test_traj[key] = val[test_indices]

# ===========================================================================================
# Plot the sensory
# ===========================================================================================

# Plot the entire heat map, left input and right label
plot_batch_idx = 0  
fig, axs = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
# axs[0].imshow(st_inputs[plot_batch_idx].T, aspect='auto', cmap='hot')
# axs[0].set_title('Input (Masked)')
# axs[0].set_xlabel('Time (ms)')
# axs[0].set_ylabel('Cell Index')
# axs[1].imshow(st_labels[plot_batch_idx].T, aspect='auto', cmap='hot')
# axs[1].set_title('Label (Unmasked)')
# axs[1].set_xlabel('Time (ms)')

# Plot single channel
# -------------------------------------------------------------------------------------------
axs[0].plot(st_inputs[plot_batch_idx, :, 0], label='Input Channel 1')
axs[0].plot(st_inputs[plot_batch_idx, :, 1], label='Input Channel 2')
axs[0].plot(st_inputs[plot_batch_idx, :, 2], label='Input Channel 3')
axs[1].plot(st_labels[plot_batch_idx, :, 0], label='Label Channel 1')
axs[1].plot(st_labels[plot_batch_idx, :, 1], label='Label Channel 2')
axs[1].plot(st_labels[plot_batch_idx, :, 2], label='Label Channel 3')
axs[0].plot(st_inputs[plot_batch_idx, :, -1], label='Input Channel 1')
axs[0].plot(st_inputs[plot_batch_idx, :, -2], label='Input Channel 2')
axs[0].plot(st_inputs[plot_batch_idx, :, -3], label='Input Channel 3')
axs[1].plot(st_labels[plot_batch_idx, :, -1], label='Label Channel 1')
axs[1].plot(st_labels[plot_batch_idx, :, -2], label='Label Channel 2')
axs[1].plot(st_labels[plot_batch_idx, :, -3], label='Label Channel 3')
plt.legend()

save_dir = f'data/'
os.makedirs(save_dir, exist_ok=True)
plt.savefig(f'{save_dir}/fig/{load_data_type}_sensory_{plot_batch_idx}.png', dpi=300, bbox_inches='tight')

# ===========================================================================================
# Save the sensory
# ===========================================================================================

save_dict = {
    'train_inputs': train_inputs,
    'train_labels': train_labels,
    'test_inputs': test_inputs,
    'test_labels': test_labels,
    'arena_map':    arena_map,
    'train_traj':   train_traj,
    'test_traj':    test_traj,
}
np.save(f'{save_dir}/{load_data_type}', save_dict)
print('Saved!')