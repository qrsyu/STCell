from matplotlib import pyplot as plt
from func import *
import numpy as np
from rtgym.utils.data_processing import RatemapAggregator
from rtgym.utils.visualization import plot_ratemaps
from tqdm import tqdm
import os 

load_data_type = '2WSMS_mask' 
data = np.load(f'data/{load_data_type}.npy', allow_pickle=True).item()

groups = [64, 128, 256, 512]
total_mean, total_errs = [], []
for group in groups:
    hidden_states = data[f'hidden_states_{group}']
    print(hidden_states.shape)
    
    # First turn data
    hs1 = hidden_states[:, :int(hidden_states.shape[1] / 2), :]
    hs2 = hidden_states[:, int(hidden_states.shape[1] / 2):, :]
    
    min_fr = 0.01  # Minimum firing rate threshold
    
    num_neurons1, num_neurons2 = [], []
    for i in range(hidden_states.shape[0]):
        
        max_fr1 = hs1[i].max(axis=0)
        mask1 = max_fr1 > min_fr
        idx1 = np.where(mask1)[0]
        select_hs1 = hs1[i][:, mask1]
        
        max_fr2 = hs2[i].max(axis=0)
        mask2 = max_fr2 > min_fr
        idx2 = np.where(mask2)[0]
        select_hs2 = hs2[i][:, mask2]

        num_neurons1.append(len(idx1))
        num_neurons2.append(len(idx2))
    num_neurons1, num_neurons2 = np.array(num_neurons1), np.array(num_neurons2)
    
    # Get the mean and standard error 
    means = np.array([num_neurons1.mean(), num_neurons2.mean()])
    errors = np.array([num_neurons1.std() / np.sqrt(len(num_neurons1)), 
                       num_neurons2.std() / np.sqrt(len(num_neurons2))])
    
    total_mean.append(means)
    total_errs.append(errors)
total_mean, total_errs = np.array(total_mean), np.array(total_errs)
print(total_mean.shape, total_errs.shape)

def plt_hist(means, errs, groups=groups):
    
    conditions = ['1st turn', '2nd turn']
    colors = ['lightskyblue', 'salmon']

    x = np.arange(means.shape[0])  
    bar_width = 0.25

    fig, ax = plt.subplots()
    
    for i in range(means.shape[1]):
        ax.bar(x + i * bar_width - bar_width,  
                means[:,i],
                yerr=errs[:,i],
                width=bar_width,
                label=conditions[i],
                color=colors[i],
                capsize=5)

    ax.set_ylabel('Number of Neurons', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=12)
    ax.legend(title='', fontsize=11)
    # ax.set_title('Ac', loc='left', fontsize=14, fontweight='bold')

    # ax.set_facecolor('#fef6e9')  # 淡黄色背景
    fig.tight_layout()
    plt.show()
    plt.savefig('test_hist.png', dpi=300, transparent=True)
    
plt_hist(total_mean, total_errs)