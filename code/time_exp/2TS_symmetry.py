import torch
import sys, os
import numpy as np
from tqdm import tqdm
code_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, code_dir)
sys.path.insert(0, os.path.join(code_dir, 'nn4n'))
from func import custom_loss
import nn4n.nn
from torch.utils.data import DataLoader, TensorDataset
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

fname = '2TS_symmetry_M2'

data = np.load(f'data/{fname}.npy', allow_pickle=True).item()
train_inputs = data['train_inputs'].to(device)
train_labels = data['train_labels'].to(device)
test_inputs = data['test_inputs'].to(device)
test_labels = data['test_labels'].to(device)

train_dataset = TensorDataset(train_inputs, train_labels)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)


D = 100
model_cfg = {
            "input_dim":    D,
            "hidden_dim":   512,
            'output_dim':   D,
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



losses = []
for epoch in tqdm(range(1000)):
    rnn.train()
    for bx, by in train_loader:
        optimizer.zero_grad()
        out, hid = rnn(bx)
        loss, _, _ = custom_loss(out, by, hid,
                                lambda_mse=1, lambda_r=0.0001)
        loss.backward(); optimizer.step()
        losses.append(loss.item())
    
    # # Add testing evaluation every 500 epochs
    # if epoch % 500 == 0:
    #     rnn.eval()
    #     with torch.no_grad():
    #         test_out, _ = rnn(test_inputs)
    #         ev2_mse = ((test_out - test_labels)**2)[w2].mean().item()
    #         base    = ((4.0 - test_labels)**2)[w2].mean().item()
    #     if ev2_mse < best - 1e-4:
    #         best, wait = ev2_mse, 0
    #         torch.save(rnn.state_dict(), f'model/{fname}_best.pth')
    #     else:
    #         wait += 10
    #         if wait > 1000:
    #             print(f"Early stopping at epoch {epoch} due to no improvement in ev2_mse.")
    #             break
    
    if  losses[-1] < 0.05 and abs(losses[-1] - losses[-50]) < 1e-3: 
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
data['train_losses'] = np.asarray(losses)
np.save(f'data/{fname}.npy', data)

os.makedirs('model', exist_ok=True)
torch.save(rnn.state_dict(), f'model/{fname}.pth')
print('Training output saved.')