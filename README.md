# STCell
Hippocampal neurons encode both spatial location (place cells) and elapsed time (time cells), to support episodic memory and spatial cognition. However, existing models explain these two phenomena using fundamentally different mechanisms: place cells emerge from continuous attractor dynamics, while time cells are often modeled as leaky integrators. This separation leaves unresolved how both representations arise within the same recurrent circuit, particularly in hippocampal CA3. We propose that place cells and time cells are two dynamical regimes of a single recurrent network. Both representations arise from hippocampal reconstruction of sensory experience, but different sensory structures give rise to distinct representational regimes.

## Install

In order to run the simulations, download the following two external repositories:

* Install [nn4n](https://github.com/NN4Neurosim/nn4n) to `../code/nn4n`:

```bash
git clone https://github.com/NN4Neurosim/nn4n.git 
cd ../code/nn4n
git checkout v1.2.1
```

* Install RatatouGym to `../code/RatatouGym`: