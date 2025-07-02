import nn4n
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import time
import numpy as np
import argparse

# Input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--load_data_type', default=None, type=str)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ===========================================================================================
# Load the input and the label
# ===========================================================================================

load_dir = f'data/'
data = np.load(f'{load_dir}/{args.load_data_type}.npy', allow_pickle=True).item()

train_inputs = torch.tensor(data['train_inputs'], dtype=torch.float32).to(device)
train_labels = torch.tensor(data['train_labels'], dtype=torch.float32).to(device)

# Create DataLoader for training and testing
train_dataset = TensorDataset(train_inputs, train_labels)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# ===========================================================================================
# Initialise the RNN
# ===========================================================================================

model_cfg = {
            "input_dim": train_inputs.shape[2],
            "hidden_dim": 512,
            'output_dim': train_inputs.shape[2],
            "alpha": 0.1,
            "learn_alpha": False,
            "preact_noise": False,
            "postact_noise": False
            }


rnn = nn4n.nn.RNN(
      # Input laters (multiple recurrent layers)
      recurrent_layers=[# 1st recurrent layer
                        nn4n.nn.RecurrentLayer(
                        projection_layer=nn4n.nn.LinearLayer(
                                         input_dim=model_cfg["input_dim"],
                                         output_dim=model_cfg["hidden_dim"]
                                         ),
                        leaky_layer=nn4n.nn.LeakyLinearLayer(
                                    # input_dim=model_cfg["hidden_dim"],
                                    # output_dim=model_cfg["hidden_dim"],
                                    # ----------------------------------------
                                    linear_layer=nn4n.nn.LinearLayer(
                                        input_dim=model_cfg["hidden_dim"],
                                        output_dim=model_cfg["hidden_dim"]),
                                    # ----------------------------------------
                                    activation=torch.nn.ReLU(),
                                    alpha=model_cfg["alpha"],
                                    learn_alpha=model_cfg["learn_alpha"],
                                    preact_noise=model_cfg["preact_noise"],
                                    postact_noise=model_cfg["postact_noise"]
                                    )
                        )
                        # 2nd recurrent layer (None)
                        ],
      # Output layer
      readout_layer=nn4n.nn.LinearLayer(
                    input_dim=model_cfg["hidden_dim"],
                    output_dim=model_cfg["output_dim"]
                    )
      )
rnn.to(device)


def custom_loss(recon, target, firing_rates, λ_mse=1.0, λ_r=1.0):
    """
    Args:
        recon: [batch, time, dim] — reconstructed experience vectors (ŷ)
        target: [batch, time, dim] — ground truth experience vectors (y)
        firing_rates: [batch, time, N] — firing rates of all hidden neurons
    """

    # Shape values
    B, T, D = target.shape
    N = firing_rates[0].shape[2]

    # 1. Reconstruction MSE Loss
    mse = torch.sum((recon - target) ** 2) / (D * T * B)

    # 2. Firing rate regularization term
    reg = torch.sum(torch.sum(firing_rates[0], dim=(0,1)) ** 2) / N

    total_loss = λ_mse * mse + λ_r * reg
    return total_loss


optimizer = torch.optim.Adam(rnn.parameters(), lr=0.0005) # Zhaoze paper
# criterion = torch.nn.MSELoss()

# ===========================================================================================
# Train the RNN
# ===========================================================================================

rnn.train()

losses = []

for epoch in tqdm(range(5000)):

    for batch_inputs, batch_labels in train_loader:

        optimizer.zero_grad()
        batch_outputs, batch_hidden = rnn(batch_inputs)
        
        # loss = criterion(batch_outputs, batch_labels)
        loss = custom_loss(batch_outputs, batch_labels, batch_hidden)
        loss.backward()
        
        optimizer.step()
        losses.append(loss.item())
        
    if epoch % 50 == 0:
        print(f'Epoch {epoch} Loss {loss.item()}')
    if len(losses) > 50 and abs(losses[-1] - losses[-50]) < 1e-3 and losses[-1] < 1:
        print("Early stopping due to convergence.")
        break
    
print("Training complete.")

# ===========================================================================================
# Test the RNN
# ===========================================================================================

test_inputs = torch.tensor(data['test_inputs'], dtype=torch.float32).to(device)

rnn.eval()
with torch.no_grad():
    test_outputs_from_RNN, hidden_states_from_RNN = rnn(test_inputs)

test_outputs = test_outputs_from_RNN.cpu().numpy()
hidden_states = hidden_states_from_RNN[0].cpu().numpy()
print('test outputs:',  type(test_outputs),  test_outputs.shape)
print('hidden states:', type(hidden_states), hidden_states.shape)

# ===========================================================================================
# Save the output and hidden states
# ===========================================================================================

print(data.keys())
data['test_outputs'] = test_outputs
data['hidden_states'] = hidden_states
print(data.keys())
np.save(f'{load_dir}/{args.load_data_type}.npy', data, allow_pickle=True)