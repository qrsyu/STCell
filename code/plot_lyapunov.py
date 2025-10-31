import numpy as np
data_type = 'max_real_c'

TS = np.load(f'data/2TS_trial.npy', allow_pickle=True).item()
TS_lpn = TS[data_type]
WSMS = np.load(f'data/2WSMS.npy', allow_pickle=True).item()
WSMS_lpn = WSMS[data_type]

from matplotlib import pyplot as plt
plt.figure(); plt.plot(np.mean(TS_lpn, axis=0), label='2TS'); plt.plot(np.mean(WSMS_lpn, axis=0), label='2WSMS')
plt.fill_between(np.arange(TS_lpn.shape[1]), 
                 np.mean(TS_lpn, axis=0)-np.std(TS_lpn, axis=0), 
                 np.mean(TS_lpn, axis=0)+np.std(TS_lpn, axis=0), alpha=0.3)
plt.fill_between(np.arange(WSMS_lpn.shape[1]), 
                 np.mean(WSMS_lpn, axis=0)-np.std(WSMS_lpn, axis=0), 
                 np.mean(WSMS_lpn, axis=0)+np.std(WSMS_lpn, axis=0), alpha=0.3)
plt.xlabel('Time step'); plt.ylabel(data_type)
plt.ylim(-1, 2)
plt.legend()
plt.savefig(f'output/compare_2TS_2WSMS_{data_type}.png')