import numpy as np
from matplotlib import pyplot as plt
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

parent_dir = 'spacetime_exp'
name = '2WSMS_mask_vel-2-3'



data = np.load(f'data/{name}.npy', allow_pickle=True).item()
train_traj = data['train_traj']
print(train_traj['coords'].shape)
dt = 0.1 # 0.1s
speed = np.linalg.norm(train_traj['disps'], axis=-1) / dt  
time_pts = speed.shape[1]
time_axis = np.linspace(0, 10, time_pts)

fig, axes = plt.subplots(3, 1, figsize=(6, 4), sharex=True, sharey=True)

for i, trial in enumerate([0, 1, 2]):
    coords = train_traj['coords'][trial]
    center = coords.mean(axis=0)
    angles = np.arctan2(coords[:, 1] - center[1], coords[:, 0] - center[0])
    cumulative = np.unwrap(angles)
    total_rotation = cumulative - cumulative[0]
    lap_boundary = np.argmax(np.abs(total_rotation) >= 2 * np.pi)

    axes[i].plot(time_axis[:lap_boundary], speed[trial, :lap_boundary], color='black', label='Lap 1')
    axes[i].plot(time_axis[lap_boundary:], speed[trial, lap_boundary:], color='red', label='Lap 2')
    axes[i].set_ylabel('Speed (cm/s)')
    if i == 0:
        axes[i].legend()
    axes[i].text(0.02, 0.9, f'Trial {trial + 1}', transform=axes[i].transAxes, verticalalignment='top')

axes[0].set_title(f'{parent_dir}: agent speed')
axes[-1].set_xlabel('Time (s)')
fig.tight_layout()
fig.savefig(f'code/{parent_dir}/{name}_speed.png', dpi=300, bbox_inches='tight')