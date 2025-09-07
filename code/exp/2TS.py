"""The pure time task:
The experiment synthesize two temporal events. The RNN is trained to predict the second event
given the first event.
"""

import os
from rtgym import RatatouGym
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

# ===========================================================================================
# Set up the arena and sensory input
# ===========================================================================================

temp_reso, spat_reso = 100, 1 # Temp reso: 100ms; Spatial reso: 1cm
gym = RatatouGym(temporal_resolution=temp_reso, spatial_resolution=spat_reso)

gym.init_arena_map(shape="rectangle")

behavior_profile = {
                    "name":                   "random_explore",
                    "type":                   "predefined",
                    "velocity_mean":          5.,
                    "velocity_sd":            1.,
                    "random_drift_magnitude": 0.05,
                    "switch_direction_prob":  0.05,
                    "switch_velocity_prob":   0.1
                    }

# Generate the first temporal event with shape n_cells, Select fixed random number from a normal distribution
n_cells = 100
np.random.seed(42)  # For reproducibility
temp_event_1 = np.random.normal(loc=4.0, scale=0.5, size=(n_cells,))
temp_event_2 = np.random.normal(loc=5.0, scale=1.0, size=(n_cells,))

sensory_profile = {
                    "time": {
                            "type":        "time_cell",
                            "n_cells":     n_cells,
                            "event_onset": [0.125, 0.875],
                            "event_onset_sigma": [0.01, 0.01],
                            "event_width": [0.025, 0.025],
                            "temp_events": [temp_event_1, temp_event_2],  # List of temporal events
                            "sigma":       0.5,    # sigma of Gaussian noise
                            "ssigma":      0.2,    # sigma of Gaussian noise smoothing (in sec)
                            "bias":        0.
                            },
                    }

# Set the sensory and behavior profiles
gym.set_sensory_from_profile(sensory_profile)
gym.set_behavior_from_profile(behavior_profile)

arena_map = gym.arena_map

# Generate (Batch size) trial
gym.trial.new_trial(duration=20, batch_size=1000)

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

labels = time_res.copy()

inputs = time_res.copy()
# Mask the inputs
from rtgym.utils.masking import Masking
mask = Masking(
                m_max=0.3,    # Maximum masking ratio
                m_min=0.1,    # Minimum masking ratio
                sigma_t=2.0,  # Temporal smoothing
                sigma_d=1.0,  # Spatial smoothing
                t_warmup=10,  # Number of initial time steps to remain unmasked
                # device=torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
                )
inputs = mask.mask(inputs).numpy()
mask_start_idx = int(inputs.shape[1]*(sensory_profile['time']['event_onset'][0]+\
                                      sensory_profile['time']['event_width'][0]+\
                                      sensory_profile['time']['event_onset_sigma'][0]))
inputs[:, mask_start_idx:, :] = 0

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
# Create and share the same colorbar
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
# axs[1].plot(labels[plot_batch_idx, :, 0], label='Label Channel 1')
# axs[1].plot(labels[plot_batch_idx, :, 1], label='Label Channel 2')
# axs[1].plot(labels[plot_batch_idx, :, 2], label='Label Channel 3')
# plt.legend()


save_dir = f'data/'
os.makedirs(save_dir, exist_ok=True)
plt.savefig(f'{save_dir}/2TS_sensory_{plot_batch_idx}.png', dpi=300, bbox_inches='tight')

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
np.save(f'{save_dir}/2TS_trial', save_dict)
print('Saved!')