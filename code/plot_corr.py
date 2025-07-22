from matplotlib import pyplot as plt
from func import *
import numpy as np
from rtgym.utils.data_processing import RatemapAggregator
from rtgym.utils.visualization import plot_ratemaps
from tqdm import tqdm
import argparse
import os 

# Input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--load_data_type', default=None, type=str)
parser.add_argument('--num_neuron',     default=512,  type=int)
args = parser.parse_args()
    
data = np.load(f'data/{args.load_data_type}.npy', allow_pickle=True).item()
hidden_states = data[f'hidden_states_{args.num_neuron}']

# ===========================================================================================
# Plot temporal firing rates correlations
# ===========================================================================================

save_dir = f'result/{args.load_data_type}_{args.num_neuron}_temporal/'
os.makedirs(save_dir, exist_ok=True)

print(hidden_states.shape)

avg_hs = np.mean(hidden_states, axis=0)
norm_hs, fig, ax = plt_hs(avg_hs, min_fr=0.1)
plt.savefig(f'{save_dir}/temporal_batch_avg.png', transparent=True)
plt.close(fig)
    
# for ihs in tqdm(range(hidden_states.shape[0])):
#     fig, ax = plt_hs(hidden_states[ihs], min_fr=0.1)
#     plt.savefig(f'{save_dir}/temporal_batch_{ihs}.png', transparent=True)
#     plt.close(fig)

# Spatial task
print(norm_hs.shape)
# Sort the norm_hs with maximum firing time
max_times = np.argmax(norm_hs, axis=0)
# Get the temporal firing width of each neuron, the threshold is 0.1 of the maximum firing rate
firing_widths = np.zeros(norm_hs.shape[1])
for i in range(norm_hs.shape[1]):
    firing_widths[i] = np.sum(norm_hs[:, i] > 1E-1)
    # firing_widths[i] = np.sum(norm_hs[:, i] > 0.5 * np.max(norm_hs[:, i]))
start = 25
end = 125
# Plot the firing widths vs the maximum firing times
plt.figure(figsize=(4,3))
# Max time < 25: skyblue
# Max time > 25: salmon
colors = ['skyblue' if t < start or t > end else 'salmon' for t in max_times]
plt.scatter(max_times, firing_widths, c=colors, 
            s=10)

# use seaborn to plot the correlation of red dots with shaded area
import seaborn as sns
rval = np.corrcoef(max_times[(max_times >= start) & (max_times < end)], 
                   firing_widths[(max_times >= start) & (max_times < end)])[0,1]
print(f'Correlation coefficient: {rval}')
sns.regplot(x=max_times[(max_times >= start) & (max_times < end)], 
            y=firing_widths[(max_times >= start) & (max_times < end)],
            scatter=False, 
            color='black',
            line_kws={"linewidth":1, "linestyle":"-"},
            ci=95,
            label=f'r = {rval:.2f}')

# calculate the fitted gradient
from scipy.optimize import curve_fit
def fit_func(x, a, b):
    return a * x + b
popt, pcov = curve_fit(fit_func, 
                       max_times[(max_times >= start) & (max_times < end)], 
                       firing_widths[(max_times >= start) & (max_times < end)])
print(f'Fitted gradient: {popt[0]:.4f}')

plt.xlabel('Maximum firing time points')
plt.ylabel("Firing width")
plt.legend()
plt.tight_layout()
plt.savefig(f'{save_dir}/temporal_corr.png', transparent=True)
plt.close(fig)