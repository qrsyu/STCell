import nn4n
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import time

# ===========================================================================================
# Load the input and the label
# ===========================================================================================



# ===========================================================================================
# Initialise the RNN
# ===========================================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model_cfg = {
            "input_dim": time_res.shape[2],
            "hidden_dim": 512,
            'output_dim': time_res.shape[2],
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

optimizer = torch.optim.Adam(rnn.parameters(), lr=0.001) # Zhaoze paper
criterion = torch.nn.MSELoss()

# ===========================================================================================
# Train the RNN
# ===========================================================================================

rnn.train()

losses = []

for epoch in tqdm(range(2000)):

    for batch_inputs, batch_labels in dataloader:

        optimizer.zero_grad()
        # print(batch_inputs.shape)

        # Ts = time.time()
        batch_outputs, batch_hidden = rnn(batch_inputs)
        # Te = time.time()
        # print(Te-Ts)
        mse_loss = criterion(batch_outputs, batch_labels)
        loss = mse_loss

        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    if epoch % 50 == 0:
      print(f'Epoch {epoch} Loss {loss.item()}')

    if len(losses) > 50 and abs(losses[-1] - losses[-50]) < 1e-4 and losses[-1] < 0.5:
      print("Early stopping due to convergence.")
      break
print("Training complete.")

# ===========================================================================================
# Test the RNN
# ===========================================================================================

rnn.eval()
with torch.no_grad():
    test_outputs_from_RNN, hidden_states_from_RNN = rnn(torch.tensor(test_inputs, dtype=torch.float32).to(device))

test_outputs = test_outputs_from_RNN.cpu().numpy()
hidden_states = hidden_states_from_RNN[0].cpu().numpy()
print('test output:', type(test_outputs), test_outputs.shape)
print('hidden state:', type(hidden_states), hidden_states.shape)

# ===========================================================================================
# Save the output and hidden states
# ===========================================================================================