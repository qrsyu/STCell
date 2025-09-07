import numpy as np

TS = np.load(f'data/2TS_trial.npy', allow_pickle=True).item()
TS_lpn = TS['lyapunov']
WSMS = np.load(f'data/2WSMS.npy', allow_pickle=True).item()
WSMS_lpn = WSMS['lyapunov']

from matplotlib import pyplot as plt
plt.figure(); plt.plot(TS_lpn[0], label='2TS'); plt.plot(WSMS_lpn[0], label='2WSMS')
plt.legend(); plt.title('Max Lyapunov exponent over time')
plt.savefig('output/compare_2TS_2WSMS_lyapunov.png')