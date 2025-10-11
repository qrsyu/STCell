import numpy as np
from dataclasses import dataclass
import torch
import matplotlib.pyplot as plt

# Load model weights 
model_dict = torch.load(f'code/time_exp/2TS_1_rnn.pth', map_location=torch.device('cuda'))

# Get the weights and biases
trained_Wrc = model_dict['recurrent_layers.0.leaky_layer.linear_layer.weight'].cpu().numpy()
trained_brc = model_dict['recurrent_layers.0.leaky_layer.linear_layer.bias'].cpu().numpy()
print('Wrc shape:', trained_Wrc.shape)  # (N,N)
# print('brc shape:', trained_brc.shape)  # (N,)
trained_Win = model_dict['recurrent_layers.0.projection_layer.weight'].cpu().numpy()
trained_bin = model_dict['recurrent_layers.0.projection_layer.bias'].cpu().numpy()
print('Win shape:', trained_Win.shape)  # (N,C)
# print('bin shape:', trained_bin.shape)  # (N,)
# trained_Wout = model_dict['readout_layer.weight'].cpu().numpy()
# trained_bout = model_dict['readout_layer.bias'].cpu().numpy()
# print('Wout shape:', trained_Wout.shape)  # (C,N)
# print('bout shape:', trained_bout.shape)  # (C,)
          



@dataclass
class ExperienceCANN:
    neuron_coords: np.ndarray
    channel_coords: np.ndarray

    # ------------------------
    # Wout
    W_out_type: str = None
    b_out_type: str = None
    # ------------------------
    
    alpha: float = 0.01
    nonlinearity: str = "relu"
    
    def __post_init__(self):
        N = self.neuron_coords.shape[0]   # hidden size
        C = self.channel_coords.shape[0]  # input size

        self.W_rc = np.load('code/theory_exp/Wrc_theory.npy')
        # self.W_rc = trained_Wrc
        self.b_rc = np.zeros((N,), dtype=float)
        
        # Plot Wrc
        fig, ax = plt.subplots(figsize=(4,3))
        im = ax.imshow(self.W_rc.T, cmap='bwr')
        plt.colorbar(im, ax=ax)
        ax.set_title('Wrc')
        ax.set_xlabel('Neurons')
        ax.set_ylabel('Neurons')
        plt.tight_layout()
        plt.savefig('code/theory_exp/Wrc_theory.png')
        plt.close()
        
        # Either uniform or normal initialization works
        # self.W_in = np.random.uniform(-np.sqrt(1/C), np.sqrt(1/C), size=(N, C)).astype(np.float32)
        # self.W_in = np.random.normal(0, 1/np.sqrt(C), size=(N, C)).astype(np.float32)
        self.W_in = trained_Win
        self.b_in = np.zeros((N,), dtype=float)
        
        # Plot Wrc
        fig, ax = plt.subplots(figsize=(4,2))
        im = ax.imshow(self.W_in.T, cmap='bwr')
        plt.colorbar(im, ax=ax)
        ax.set_title('Win')
        ax.set_xlabel('Channels')
        ax.set_ylabel('Neurons')
        plt.tight_layout()
        plt.savefig('code/theory_exp/Win_theory.png')
        plt.close()
        
        
        if self.W_out_type is None:
            # self.W_out = trained_Wout
            self.W_out = np.random.uniform(-np.sqrt(1/C), np.sqrt(1/C), size=(N, C)).astype(np.float32)
        else:
            raise NotImplementedError("Only None W_out_type is implemented now.")
        if self.b_out_type is None:
            # self.b_out = trained_bout
            self.b_out = np.random.uniform(-np.sqrt(1/C), np.sqrt(1/C), size=(C,)).astype(np.float32)
        else:
            raise NotImplementedError("Only None b_out_type is implemented now.")


    def phi(self, V):
        if self.nonlinearity == "relu":
            return np.maximum(0.0, V)
        else:
            raise ValueError("Unknown nonlinearity")


    def step(self, V, e_t):
        r = self.phi(V)           # (B,N)
        
        # W^in @ e_t + b^in
        inn = e_t @ self.W_in.T + self.b_in[None, :]
        # W^rc @ r + b^rc
        rec = r   @ self.W_rc.T + self.b_rc[None, :]
        
        target = rec + inn 

        V_new = (1.0 - self.alpha) * V + self.alpha * target
        r_new = self.phi(V_new)   # (B,N)
        return V_new, r_new

    def run(self, e):
        B, T, C = e.shape
        N = self.neuron_coords.shape[0]

        V = np.zeros((B, N), dtype=float)        # Initialise V to zero
        r_all = np.zeros((B, T, N), dtype=float) # Record outputs for all time steps
        for t in range(T):
            V, r = self.step(V, e[:, t, :])
            r_all[:, t, :] = r
        return r_all
