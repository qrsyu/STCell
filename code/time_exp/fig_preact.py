import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import sys, os
code_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, code_dir)
sys.path.insert(0, os.path.join(code_dir, 'nn4n'))
import nn4n.nn

# Only change these
name = '_symmetry_numpy'
model_name = '2TS_symmetry'
events = [(2.5, 3.0), (17.5, 18.0)]   # temporal events (s)
min_max_fr = 0.1                      # keep neurons whose trial-averaged max fr > this



# Rebuild the RNN (same config as 2TS_symmetry.py) and load the trained weights
D, H = 100, 512
rnn = nn4n.nn.RNN(
      recurrent_layers=[nn4n.nn.RecurrentLayer(
                        projection_layer=nn4n.nn.LinearLayer(input_dim=D, output_dim=H),
                        leaky_layer=nn4n.nn.LeakyLinearLayer(
                                    linear_layer=nn4n.nn.LinearLayer(input_dim=H, output_dim=H),
                                    activation=torch.nn.ReLU(), alpha=0.01,
                                    preact_noise=0.1, postact_noise=0.1))],
      readout_layer=nn4n.nn.LinearLayer(input_dim=H, output_dim=D))
rnn.load_state_dict(torch.load(f'model/{model_name}.pth', map_location='cpu'))
rnn.eval()

data = np.load(f'data/2TS{name}.npy', allow_pickle=True).item()
test_inputs = torch.tensor(data['test_inputs']).float()



# Step the leaky layer by hand to record the pre-activation v (fr = ReLU(v))
rec = rnn.recurrent_layers[0]
B, T, _ = test_inputs.shape
v, fr = torch.zeros(B, H), torch.zeros(B, H)
vs = []
with torch.no_grad():
    for t in range(T):
        u = rec.projection_layer(test_inputs[:, t])
        fr, v = rec.leaky_layer(fr, v, u)
        vs.append(v)
vs = torch.stack(vs, dim=1).numpy()          # (B, T, H)
frs = np.maximum(vs, 0)
assert np.allclose(frs, data['test_hidden_states'], atol=1e-4), 'fr does not match saved hidden states'

# Average across trials (event times are fixed in this dataset)
avg_v, avg_fr = vs.mean(axis=0), frs.mean(axis=0)

# Keep active neurons, sort by peak firing time
active = np.where(avg_fr.max(axis=0) > min_max_fr)[0]
order = active[np.argsort(np.argmax(avg_fr[:, active], axis=0))]
print('Active neurons:', len(order), 'out of', H)



fig, axes = plt.subplots(1, 3, figsize=(12, 3.2), gridspec_kw={'width_ratios': [1, 1, 0.9]})
extent = [0, T/10, 0, len(order)]
time = np.arange(T) / 10

from func import plt_hs
_, fig, axes[0] = plt_hs(avg_fr[:, order], min_fr=0, fig=fig, ax=axes[0])
axes[0].set_title('Normalised firing rate', fontsize=10)
cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='jet'), ax=axes[0], extend='max')

# Clip the colour scale so a few very negative neurons do not hide the slow ramps
vmin = np.percentile(avg_v[:, order], 1)
vmax = avg_v[:, order].max()
im1 = axes[1].imshow(avg_v[:, order].T, aspect='auto', cmap='RdBu_r', extent=extent,
                     norm=TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax),
                     interpolation='nearest')
axes[1].set_title(r'Pre-activation $v$ (0 = firing threshold)', fontsize=10)
cb = fig.colorbar(im1, ax=axes[1], extend='min')
cb.set_ticks([np.ceil(vmin), np.ceil(vmin)/2, 0, np.floor(vmax*10)/10])
axes[1].sharey(axes[0])
# axes[1].tick_params(labelleft=False)

# Late neurons: silent between the two events, firing after the second event starts
late = order[(avg_fr[40:140, order].max(axis=0) < 0.01) & (avg_fr[150:, order].max(axis=0) > min_max_fr)]
for n in late:
    axes[2].plot(time, avg_v[:, n], color='#2a6fb0', lw=0.8, alpha=0.35)
axes[2].plot(time, avg_v[:, late].mean(axis=1), color='#0b2e59', lw=2, label=f'average $v$')
axes[2].axhline(0, color='black', lw=1, ls='--')
# axes[2].text(0.3, 0.4, 'threshold (v = 0)', fontsize=8, va='bottom')
axes[2].set_title(r'$v$ of neurons that fire only after ~16 s', fontsize=10)
axes[2].set_ylabel(r'Pre-activation $v$')
axes[2].legend(fontsize=8, loc='lower right', frameon=False)
axes[2].spines[['top', 'right']].set_visible(False)

for ax in axes:
    for on, off in events:
        ax.axvspan(on, off, color='grey', alpha=0.25, lw=0)
    ax.set_xlim(0, T/10)
    ax.set_xticks(np.linspace(0, T/10, 6))
    ax.set_xlabel('Time (s)')
axes[0].set_ylabel('Neurons')
axes[1].set_ylabel('Neurons')

fig.tight_layout()
os.makedirs('code/time_exp/suppfig', exist_ok=True)
fig.savefig(f'code/time_exp/suppfig/time_exp_preact{name}.png', dpi=500, bbox_inches='tight')
print('Saved.')
