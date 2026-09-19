import torch
import nn4n.nn
import sys, os
import numpy as np
from tqdm import tqdm
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from func import custom_loss
from torch.utils.data import DataLoader, TensorDataset
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

fname = '2TS_mask_loss'

data = np.load(f'data/{fname}.npy', allow_pickle=True).item()
train_inputs = torch.tensor(data['train_inputs'], dtype=torch.float32).to(device)
train_labels = torch.tensor(data['train_labels'], dtype=torch.float32).to(device)
test_inputs = torch.tensor(data['test_inputs'], dtype=torch.float32).to(device)

train_dataset = TensorDataset(train_inputs, train_labels)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)



model_cfg = {
            "input_dim":    100,
            "hidden_dim":   512,
            'output_dim':   100,
            "alpha":        0.01,
            "learn_alpha":  False,
            "preact_noise": 0.1,
            "postact_noise":0.1
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
optimizer = torch.optim.Adam(rnn.parameters(), lr=0.0005) 



rnn.train()
losses = []
for epoch in tqdm(range(5000)):

    for batch_inputs, batch_labels in train_loader:
        optimizer.zero_grad()
        batch_outputs, batch_hidden = rnn(batch_inputs)
        loss, loss1, loss2 = custom_loss(
                                batch_outputs, batch_labels, batch_hidden,
                                lambda_mse=1, lambda_r=0.0001)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
    if epoch % 500 == 0:
        print(f'Epoch {epoch} Loss {loss.item()}')
        print(loss1, loss2)
    if  losses[-1] < 0.05 and abs(losses[-1] - losses[-50]) < 1e-4: 
        print("Early stopping due to convergence.")
        break
print("Training complete.")



rnn.eval()
with torch.no_grad():
    test_outputs_from_RNN, hidden_states_from_RNN = rnn(test_inputs)

test_outputs = test_outputs_from_RNN.cpu().numpy()
hidden_states = hidden_states_from_RNN[0].cpu().numpy()
print('test outputs:',  type(test_outputs),  test_outputs.shape)
print('hidden states:', type(hidden_states), hidden_states.shape)



data['test_outputs'] = test_outputs
data['test_hidden_states'] = hidden_states
data['train_losses'] = losses
np.save(f'data/{fname}.npy', data)

os.makedirs('../../model', exist_ok=True)
torch.save(rnn.state_dict(), f'../../model/{fname}.pth')