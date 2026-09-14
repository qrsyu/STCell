def time_analysis(hidden, n_shuffles=500, alpha=0.01):
    """Identify time cells via split-half reliability.

    Parameters
    ----------
    hidden : array, shape (trials, time, neurons)
    n_shuffles : int
        Number of shuffles for significance threshold.
    alpha : float
        Significance level.

    Returns
    -------
    is_time_cell : bool array, shape (neurons,)
    info : dict with tuning curves and correlations.
    """
    import numpy as np
    n_trials, T, N = hidden.shape

    # split trials into odd/even
    odd = hidden[0::2]   # (trials/2, T, N)
    even = hidden[1::2]

    # temporal tuning curve = trial-averaged firing rate over time
    tuning_odd = odd.mean(axis=0)    # (T, N)
    tuning_even = even.mean(axis=0)

    # real split-half correlation per neuron
    real_corr = np.zeros(N)
    for n in range(N):
        if tuning_odd[:, n].std() == 0 or tuning_even[:, n].std() == 0:
            real_corr[n] = 0.0  # silent neuron, not a time cell
        else:
            real_corr[n] = np.corrcoef(tuning_odd[:, n], tuning_even[:, n])[0, 1]

    # shuffle null: circularly shift each trial's time axis
    rng = np.random.default_rng(42)
    shuffle_corrs = np.zeros((n_shuffles, N))

    for s in range(n_shuffles):
        shifted = hidden.copy()
        for trial in range(n_trials):
            shift = rng.integers(1, T)
            shifted[trial] = np.roll(shifted[trial], shift, axis=0)

        s_odd = shifted[0::2].mean(axis=0)
        s_even = shifted[1::2].mean(axis=0)
        for n in range(N):
            if s_odd[:, n].std() == 0 or s_even[:, n].std() == 0:
                shuffle_corrs[s, n] = 0.0
            else:
                shuffle_corrs[s, n] = np.corrcoef(s_odd[:, n], s_even[:, n])[0, 1]

    # significance: real corr > (1 - alpha) percentile of shuffle
    threshold = np.percentile(shuffle_corrs, 100 * (1 - alpha), axis=0)
    is_time_cell = real_corr > threshold

    return is_time_cell


import numpy as np

parent_dir = 'spacetime_exp'
name = '2WSMS_mask'
data = np.load(f'data/{name}.npy', allow_pickle=True).item()
hs = data['hidden_states_512']  # (trials, time, neurons)
print(hs.shape)
hs_early = hs[:, :50, :]  # (trials, time, neurons)
hs_later = hs[:, 50:, :]  # (trials, time, neurons)



# Only select those with mean firing rate > 0.1
hs_later = hs_later[:, :, hs_later.mean(axis=(0, 1)) > 0.1]
print('active neurons:', hs_later.shape[2])

is_time_cell = time_analysis(hs_later)

# Select time cells only
hs_later_time = hs_later[:, :, is_time_cell]  # (trials, time, time_cells)
print('time cells:', hs_later_time.shape[2])



import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from func import plt_hs
from matplotlib import pyplot as plt

fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.5))
_, fig, ax = plt_hs(np.mean(hs_later_time, axis=0), min_fr=0, fig=fig, ax=ax)
ax.set_xlabel('Time (s)')

fig.savefig(f'code/spacetime_exp/time_cells_2nd_lap.png', dpi=300, bbox_inches='tight')