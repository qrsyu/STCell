"""The time task + the space task:
"""

import os
from rtgym import RatatouGym
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from func import generate_circular_trajectories

load_data_type = '2TS2WSMS_5'

# ===========================================================================================
# Set up the arena and sensory input
# ===========================================================================================

temp_reso, spat_reso = 100, 1 # Temp reso: 100ms; Spatial reso: 1cm
gym = RatatouGym(temporal_resolution=temp_reso, spatial_resolution=spat_reso)

R_out, R_in = 17, 10
gym.init_arena_map(shape="loop", outer_radius=R_out, inner_radius=R_in)

vel_mean, vel_std = 2, 2
behavior_profile = {
                    "name":                   "random_explore",
                    "type":                   "predefined",
                    "velocity_mean":          vel_mean,
                    "velocity_sd":            vel_std,
                    "random_drift_magnitude": 0.05,
                    "switch_direction_prob":  0.05,
                    "switch_velocity_prob":   0.1,
                    'avoid_boundary_dist': 60
                    }

sensory_profile = {
                   "wsm": {
                          "type":     "weak_sm_cell",
                          "n_cells":   96,
                          "sigma":     15,
                          "magnitude": 4,
                          "normalize": True
                          },
                   "time": {
                            "type":        "time_cell",
                            "n_cells":     10,
                            "mag":         0.5,
                            "mag_sigma":   0.5,
                            # 'mag_func': lambda x: (x-1)**2 + 2,
                            # 'mag_func': lambda x: 4 *np.sin(x)/x+1,
                            "event_onset": [0.25, 0.75],
                            "event_width": [0.05, 0.05],
                            "sigma":       0.2,    # sigma of Gaussian noise
                            "ssigma":      0.1,    # sigma of Gaussian noise smoothing (in sec)
                            "bias":        0.
                            },
                    }

# Set the sensory and behavior profiles
gym.set_sensory_from_profile(sensory_profile)
gym.set_behavior_from_profile(behavior_profile)

arena_map = gym.arena_map

time_pts = 100
traj = generate_circular_trajectories(arena_map, R_out, R_in, vel_mean, vel_std,
                    time_points=time_pts, batch_size=1000, visualize=False)

# Generate (Batch size) trial
gym.trial.new_trial(duration=0, external_traj=traj)

# Get some specific responses within a time range, the key should be the same as the sensory profile
space_res = gym.trial.get_responses(keys='wsm')
print('space responses:', space_res.shape)

# Get some specific responses within a time range, the key should be the same as the sensory profile
time_res = gym.trial.get_responses(keys='time')
print('time responses:', time_res.shape)

traj = gym.trial.get_traj()
traj['coords'] = traj['coords_float'] 
traj['hds'] = traj['head_directions']
traj['disps'] = traj['displacements']
del traj['coords_float']
del traj['head_directions']
del traj['displacements']
print(traj.keys())

# ===========================================================================================
# Make input and label
# ===========================================================================================

time_labels = time_res.copy()
time_inputs = time_res.copy()
time_inputs[:, int(time_inputs.shape[1]*(sensory_profile['time']['event_onset'][0]+sensory_profile['time']['event_width'][0])):, :] = 0

space_labels = space_res.copy()
space_inputs = space_res.copy()
space_inputs[:, 50:, :] = 0

labels = np.concatenate([space_labels, time_labels], axis=-1)
inputs = np.concatenate([space_inputs, time_inputs], axis=-1)

# labels = time_labels
# inputs = time_inputs

# Mask the inputs
from rtgym.utils.masking import Masking
mask = Masking(
                m_max=0.2,    # Maximum masking ratio
                m_min=0.1,    # Minimum masking ratio
                sigma_t=2.0,  # Temporal smoothing
                sigma_d=1.0,  # Spatial smoothing
                t_warmup=10,  # Number of initial time steps to remain unmasked
                # device=torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
                )
inputs = mask.mask(inputs).numpy()
# inputs[:, int(inputs.shape[1]*(sensory_profile['time']['event_onset'][0]+sensory_profile['time']['event_width'][0])):, :] = 0

# Split the data to training and test set along axis=1
indices = np.arange(inputs.shape[0])
train_inputs, test_inputs, train_labels, test_labels, train_indices, \
test_indices = train_test_split(inputs, labels, indices, test_size=0.05, random_state=42)

print('train_inputs:',  train_inputs.shape)
print('train_labels:',  train_labels.shape)
print('test_inputs:',   test_inputs.shape)
print('test_labels:',   test_labels.shape)
print('train_indices:', train_indices.shape)
print('test_indices:',  test_indices.shape)

train_traj, test_traj = {}, {}
for key, val in traj.items():
    train_traj[key] = val[train_indices]
    test_traj[key] = val[test_indices]

# ===========================================================================================
# Plot the sensory
# ===========================================================================================

plot_batch_idx = 0  
fig, axs = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
# Plot the entire heat map, left input and right label
# -------------------------------------------------------------------------------------------
axs[0].imshow(inputs[plot_batch_idx].T, aspect='auto', cmap='hot')
axs[0].set_title('Input (Masked)')
axs[0].set_xlabel('Time (ms)')
axs[0].set_ylabel('Cell Index')
axs[1].imshow(labels[plot_batch_idx].T, aspect='auto', cmap='hot')
axs[1].set_title('Label (Unmasked)')
axs[1].set_xlabel('Time (ms)')

# # Plot single channel
# # -------------------------------------------------------------------------------------------
# axs[0].plot(inputs[plot_batch_idx, :, 0], label='Input Channel 1')
# axs[0].plot(inputs[plot_batch_idx, :, 1], label='Input Channel 2')
# axs[0].plot(inputs[plot_batch_idx, :, 2], label='Input Channel 3')
# axs[0].plot(inputs[plot_batch_idx, :, -3], label='Input Channel 98')
# axs[0].plot(inputs[plot_batch_idx, :, -2], label='Input Channel 99')
# axs[0].plot(inputs[plot_batch_idx, :, -1], label='Input Channel 100')

# axs[1].plot(labels[plot_batch_idx, :, 0], label='Label Channel 1')
# axs[1].plot(labels[plot_batch_idx, :, 1], label='Label Channel 2')
# axs[1].plot(labels[plot_batch_idx, :, 2], label='Label Channel 3')
# axs[1].plot(labels[plot_batch_idx, :, -3], label='Label Channel 98')
# axs[1].plot(labels[plot_batch_idx, :, -2], label='Label Channel 99')
# axs[1].plot(labels[plot_batch_idx, :, -1], label='Label Channel 100')
# plt.legend()

save_dir = f'data/'
os.makedirs(save_dir, exist_ok=True)
plt.savefig(f'{save_dir}/{load_data_type}_sensory_{plot_batch_idx}.png', dpi=300, bbox_inches='tight')

# ===========================================================================================
# Save the sensory
# ===========================================================================================

save_dict = {
            'train_inputs': train_inputs,
            'train_labels': train_labels,
            'test_inputs':  test_inputs,
            'test_labels':  test_labels,
            'arena_map':    arena_map,
            'train_traj':   train_traj,
            'test_traj':    test_traj,
        }
np.save(f'{save_dir}/{load_data_type}', save_dict)
print('Saved!')