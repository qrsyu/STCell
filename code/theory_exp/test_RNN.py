import numpy as np
from .RNN import ExperienceCANN
from ..func import plt_hs
from matplotlib import pyplot as plt



### Load the experimental data ###
load_data_type = '2TS_1_data'
num_neuron = 512
data = np.load(f'code/time_exp/{load_data_type}.npy', allow_pickle=True).item()
exp_vectors = data['test_inputs'].cpu().numpy()  
print(exp_vectors.shape)



### The coordinates of neurons and input channels in abstract 2D space
# ---- neurons: 512 = 32 x 16 ----
# # One dim ver
# Nt = 512
# neuron_coords = np.linspace(0, 1.0, Nt, endpoint=False).reshape(-1,1)   # (512, 1)
# Two dims ver
Nx, Ny = 32, 16
y = np.linspace(0, 1.0, Ny, endpoint=False)
x = np.linspace(0, 1.0, Nx, endpoint=False)
Y, X = np.meshgrid(y, x, indexing='ij')
neuron_coords = np.stack([X.ravel(), Y.ravel()], axis=-1)   # (512, 2)
# Three dims ver
# Nx, Ny, Nt = 8, 8, 8              
# y = np.linspace(0, 1.0, Ny, endpoint=False)
# x = np.linspace(0, 1.0, Nx, endpoint=False)
# t = np.linspace(0, 1.0, Nt, endpoint=False)
# Y, X, T = np.meshgrid(y, x, t, indexing='ij')
# neuron_coords = np.stack([X.ravel(), Y.ravel(), T.ravel()], axis=-1)   # (512, 3)

# ---- channels: 100 = 5 x 5 x 4 ----
# # One dim ver
# Ct = 100
# channel_coords = np.linspace(0, 1.0, Ct, endpoint=False).reshape(-1,1)  # (100, 1)
# Three dims ver
Cx, Cy, Ct = 5, 5, 4
xc = np.linspace(0, 1.0, Cx, endpoint=False)
yc = np.linspace(0, 1.0, Cy, endpoint=False)
tc = np.linspace(0, 1.0, Ct, endpoint=False)
Yc, Xc, Tc = np.meshgrid(yc, xc, tc, indexing='ij')
channel_coords = np.stack([Xc.ravel(), Yc.ravel(), Tc.ravel()], axis=-1)  # (100, 3)

# ---- Parameters of the RNN ----
RNNnet = ExperienceCANN(
    alpha = 0.01,
    neuron_coords=neuron_coords,
    channel_coords=channel_coords,
)

fr = RNNnet.run(exp_vectors)   # -> shape (B,T,N)
# avg_fr = np.mean(fr, axis=0)   # (T,N)
avg_fr = fr[0]

# Plot the hidden states
fig, ax = plt.subplots(figsize=(4,3))
norm_hs, fig, ax = plt_hs(avg_fr, ax=ax, fig=fig, min_fr=0.1, 
                        #   mask_start=20, mask_end=150
                          )
ax.set_xlabel('Time (s)')
ax.set_xlim(2,16)
plt.tight_layout()
plt.savefig(f'code/theory_exp/theory_rnn_{load_data_type}_{num_neuron}_hs.png', transparent=True
            )














# # Save the data
# data[f'theory_hidden_states_{num_neuron}'] = fr



# ### Compute the Jacobian stability over time ###
# max_real_c, _, eigen_c, _ = compute_jacobian_over_time(RNNnet, exp_vectors[0], return_full_spectrum=True) # (B, T,) (B, T,)

# plt.figure(); 
# # Plot the mean over trials
# plt.plot(max_real_c.mean(axis=0), label="max Re λ(Jc)")
# plt.ylim(-1, 3)
# # Plot the 1 std shaded area
# plt.fill_between(np.arange(max_real_c.shape[1]), 
#                  max_real_c.mean(axis=0)-max_real_c.std(axis=0),
#                  max_real_c.mean(axis=0)+max_real_c.std(axis=0),
#                  alpha=0.3)
# plt.axhline(0, ls='--', lw=1); plt.legend(loc='best'); plt.title("Continuous-time local stability")
# plt.savefig(f'output/theory_rnn_{load_data_type}_{num_neuron}_max_real_c.png')

# eigen_c = np.array(eigen_c)  # (B, T, N)
# print(eigen_c.shape)
# plt_fig = plot_realpart_heatmap(eigen_c, rmax=2) # Default of using the first batch
# plt_fig.savefig(f'output/theory_rnn_{load_data_type}_{num_neuron}_eigen_heatmap.png')

# data['max_real_c'] = max_real_c
# np.save(f'data/{load_data_type}.npy', data, allow_pickle=True)



# ### Lyapunov exponent estimation ###
# avg_lpn, lpn = max_lyapunov_exponent(RNNnet, exp_vectors[0], return_logs=True)
# print(avg_lpn)
# plt.figure(); plt.plot(lpn[0]); 
# plt.axhline(avg_lpn, ls='--', lw=1, label='average'); plt.legend(loc='best'); plt.title("Max Lyapunov exponent")
# plt.savefig(f'output/theory_rnn_{load_data_type}_{num_neuron}_lyapunov.png')

# data['lyapunov'] = lpn
# np.save(f'data/{load_data_type}.npy', data, allow_pickle=True)