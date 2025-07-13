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

for ihs in tqdm(range(hidden_states.shape[0])):
    fig, ax = plt_hs(hidden_states[ihs], min_fr=0.01)
    plt.savefig(f'{save_dir}/temporal_batch_{ihs}.png', transparent=True)
    plt.close(fig)