# STCell
Hippocampal neurons encode both spatial location (place cells) and elapsed time (time cells), to support episodic memory and spatial cognition. However, existing models explain these two phenomena using fundamentally different mechanisms: place cells emerge from continuous attractor dynamics, while time cells are often modeled as leaky integrators. This separation leaves unresolved how both representations arise within the same recurrent circuit, particularly in hippocampal CA3. We propose that place cells and time cells are two dynamical regimes of a single recurrent network. Both representations arise from hippocampal reconstruction of sensory experience, but different sensory structures give rise to distinct representational regimes.

## Install

In order to run the simulations, download the following external repository:

* Go to `../code/`, and install [nn4n](https://github.com/NN4Neurosim/nn4n):

```bash
git clone --single-branch --branch v1.2.1 https://github.com/NN4Neurosim/nn4n.git 
cd nn4n
pip install .
```

## Acknowledgement

Parts of the code in `../code/rtgym/` are adapted from the original 
[RatatouGym](https://github.com/zhaozewang/rtgym) repository. 
We thank the authors for open-sourcing their implementation.

If you use this repository, please also cite the original work:

```bibtex
@article{wang2024time,
  title={Time makes space: Emergence of place fields in networks encoding temporally continuous sensory experiences},
  author={Wang, Zhaoze and Di Tullio, Ronald W and Rooke, Spencer and Balasubramanian, Vijay},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={37836--37864},
  year={2024}
}
```