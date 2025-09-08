import numpy as np
from .RNN import ExperienceCANN
from .jacobian import *
from ..func import plt_hs
from matplotlib import pyplot as plt



### Load the experimental data ###
load_data_type = '2WSMS'
num_neuron = 512
data = np.load(f'data/{load_data_type}.npy', allow_pickle=True).item()
exp_vectors = data['test_labels']
print(exp_vectors.shape)



### The coordinates of neurons and input channels in abstract 2D space
# ---- neurons: 512 = 8 x 8 x 8 ----
Ny, Nx, Nt = 8, 8, 8               
y = np.linspace(0, 1.0, Ny, endpoint=False)
x = np.linspace(0, 1.0, Nx, endpoint=False)
t = np.linspace(0, 1.0, Nt, endpoint=False)
Y, X, T = np.meshgrid(y, x, t, indexing='ij')
neuron_coords = np.stack([X.ravel(), Y.ravel(), T.ravel()], axis=-1)   # (512, 3)

# ---- channels: 100 = 5 x 5 x 4 ----
Cy, Cx, Ct = 5, 5, 4
yc = np.linspace(0, 1.0, Cy, endpoint=False)
xc = np.linspace(0, 1.0, Cx, endpoint=False)
tc = np.linspace(0, 1.0, Ct, endpoint=False)
Yc, Xc, Tc = np.meshgrid(yc, xc, tc, indexing='ij')
channel_coords = np.stack([Xc.ravel(), Yc.ravel(), Tc.ravel()], axis=-1)  # (100, 3)

# 2) 建网（不训练，权重由高斯核直接给出）
RNNnet = ExperienceCANN(
    neuron_coords=neuron_coords,
    channel_coords=channel_coords,
    alpha=0.3,
    sigma_rec=0.02, #sigma_rec,
    sigma_in=0.1, #sigma_in,
    W0_rc=1, W0_in=1,  # Amplitude
    gain=1,  nonlinearity="relu",
    periodic=(True, True, False), 
    box_size=(1.0,1.0,1.0),   # 打开环面距离（可关）
    divisive_norm=True        # 分式归一化确保 bump 稳定
)



ref_fr = RNNnet.run(exp_vectors)
print(ref_fr.shape)



# Patially mask the input 
mask_indices = [30, 40, 50, 60, 70, 80, 90, 100]

# Set up the figure
fig, ax = plt.subplots(len(mask_indices), 1, figsize=(6, len(mask_indices)), sharex=True)
for idx, mask_idx in enumerate(mask_indices):
    
    exp_vectors[:, 20:mask_idx, :] = 0

    fr = RNNnet.run(exp_vectors)   # -> shape (B,T,N)
    
    # Compute the higher dimensional distance
    dist = np.linalg.norm(fr - ref_fr, axis=-1)  # (B,T)
    
    # Get the mean and std over batch
    mean_dist = np.mean(dist, axis=0)  # (T,)
    std_dist = np.std(dist, axis=0)    # (T,)
    
    # Plot the distance with error bars
    ax[idx].plot(mean_dist, color=f'C{idx}')
    ax[idx].fill_between(np.arange(mean_dist.shape[0]), 
                    mean_dist - std_dist,
                    mean_dist + std_dist,
                    color=f'C{idx}', alpha=0.3)
    # Plot the vertical dashed line at the mask point
    ax[idx].axvline(x=mask_idx, color=f'C{idx}', linestyle='--')
    ax[idx].ylabel('N=512 Distance')
    
plt.xlabel('Time steps')
plt.tight_layout()
plt.savefig('higher_dim_distance.png')