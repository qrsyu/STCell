import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from func import plt_hs
from sklearn.decomposition import PCA



load_data_type = '2TS_trial'
num_neuron = 512



# Load the hidden state data
data = np.load(f'data/{load_data_type}.npy', allow_pickle=True).item()
hs = data[f'hidden_states_{num_neuron}']
avg_hs = np.mean(hs, axis=0)
print(avg_hs.shape)

# Plot the hidden states
fig, ax = plt.subplots(figsize=(10, 6))
norm_hs, fig, ax = plt_hs(avg_hs, ax=ax, fig=fig, 
                        #   mask_start=20, mask_end=150
                          )
plt.savefig(f'output/rnn_{load_data_type}_{num_neuron}_hs.png')



# Load the model
model_dir = f'data/rnn_model/'
model = torch.load(f'{model_dir}/{load_data_type}_{num_neuron}.pth', map_location=torch.device('cuda'))

# Get the weights and biases
print(model['recurrent_layers.0.leaky_layer.linear_layer.weight'].shape)
# print(model['recurrent_layers.0.leaky_layer.linear_layer.bias'].shape)
# print(model['recurrent_layers.0.projection_layer.weight'].shape)
# print(model['recurrent_layers.0.projection_layer.bias'].shape)
# print(model['readout_layer.weight'].shape)
# print(model['readout_layer.bias'].shape)

recurrent_weights = model['recurrent_layers.0.leaky_layer.linear_layer.weight']
vmin = np.percentile(recurrent_weights.cpu().numpy(), 1)
vmax = np.percentile(recurrent_weights.cpu().numpy(), 99)

# Plot the weights
plt.figure(figsize=(8, 6))
plt.imshow(recurrent_weights.cpu().numpy(), aspect='auto', cmap='plasma', vmin=vmin, vmax=vmax)
plt.colorbar(label='Weight Value')
plt.title('Recurrent Layer Weights')
plt.xlabel('Neuron Index')
plt.ylabel('Neuron Index')
plt.tight_layout()
plt.savefig(f'output/rnn_{load_data_type}_{num_neuron}_weights.png')



# # Re-organize the weights for better visualization
# pca = PCA(n_components=1)
# neuron_embedding = pca.fit_transform(recurrent_weights.cpu().numpy())[:, 0]

# sorted_idx = np.argsort(neuron_embedding)
# reorg_weights = recurrent_weights[sorted_idx, :][:, sorted_idx]

# # Plot the weights
# plt.figure(figsize=(8, 6))
# plt.imshow(reorg_weights.cpu().numpy(), aspect='auto', cmap='plasma', vmin=vmin, vmax=vmax)
# plt.colorbar(label='Weight Value')
# plt.title('Recurrent Layer Weights (Reorganized)')
# plt.xlabel('Neuron Index')
# plt.ylabel('Neuron Index')
# plt.tight_layout()
# plt.savefig(f'output/rnn_{load_data_type}_{num_neuron}_reorg_weights.png')



from sklearn.manifold import TSNE

# 1D embeddings
embedding_1d = TSNE(n_components=1).fit_transform(recurrent_weights.cpu().numpy())
x = embedding_1d[:, 0]
# Sort with increasing x values
x = x[np.argsort(x)]
# Plot the 1D embedding
fig = plt.figure(figsize=(8, 6))
plt.plot(x, c='blue', linewidth=2)
plt.fill_between(np.arange(len(x)), x, color='blue', alpha=0.1)
plt.title('1D Embedding of Recurrent Weights')
plt.xlabel('Neuron Index')
plt.tight_layout()
plt.savefig(f'output/rnn_{load_data_type}_{num_neuron}_weights_embedding1.png')

# 3D embeddings
embedding_3d = TSNE(n_components=3).fit_transform(recurrent_weights.cpu().numpy())
x3d, y3d, z3d = embedding_3d[:, 0], embedding_3d[:, 1], embedding_3d[:, 2]
# Sort with increasing x values
y3d = y3d[np.argsort(x3d)]
z3d = z3d[np.argsort(x3d)]
x3d = x3d[np.argsort(x3d)]
# Calculate the geometric center of the embedding
center_x = np.mean(x3d)
center_y = np.mean(y3d)
center_z = np.mean(z3d)
# Plot the 3D embedding
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
# Plot with a change of color
sc = ax.scatter(x, y3d, z3d, c=np.arange(len(x3d)), cmap='plasma', s=10)
# # Plot the geometric center point in black
# ax.scatter(center_x, center_y, center_z, c='black', s=50, label='Center Point')
# plt.legend()
# plt.title('3D Embedding of Recurrent Weights')
ax.set_xlabel('W1')
ax.set_ylabel('W2')
ax.set_zlabel('W3')
plt.tight_layout()
plt.savefig(f'output/rnn_{load_data_type}_{num_neuron}_weights_embedding3.png',
            transparent=True)

# 2D embeddings
embedding_2d = TSNE(n_components=2).fit_transform(recurrent_weights.cpu().numpy())
x2d, y2d = embedding_2d[:, 0], embedding_2d[:, 1]
# Sort with increasing x values
y2d = y2d[np.argsort(x2d)]
x2d = x2d[np.argsort(x2d)]
# Find the geometric center of the embedding
center_x = np.mean(x2d)
center_y = np.mean(y2d)
# Calculate the angle of each points wrt the center point
angles_2d = np.arctan2(y2d - center_y, x2d - center_x)
# Sort the points by angle
sorted_indices = np.argsort(angles_2d)
x2d = x2d[sorted_indices]
y2d = y2d[sorted_indices]
# Plot the 2D embedding
fig = plt.figure(figsize=(8, 6))
plt.scatter(x2d, y2d, c=np.arange(len(x2d)), cmap='plasma', s=10)
# Plot the center point in black
plt.scatter(center_x, center_y, c='black', s=50, label='Center Point')
# Plot arrows from the center point to each point
for i in range(len(x2d)):
    plt.arrow(center_x, center_y, x2d[i] - center_x, y2d[i] - center_y,
              head_width=0.1, head_length=0.1, fc='gray', ec='gray', alpha=0.5)
plt.legend()
plt.title('2D Embedding of Recurrent Weights')
plt.xlabel('W1')
plt.ylabel('W2')
plt.tight_layout()
plt.savefig(f'output/rnn_{load_data_type}_{num_neuron}_weights_embedding2.png')


# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import r2_score

# def compute_features(weights, embedding):
#     N = embedding.shape[0]
#     X = []
#     y = []
#     for i in range(N):
#         for j in range(N):
#             if i != j:
#                 d = np.linalg.norm(embedding[i] - embedding[j])
#                 X.append([d])
#                 y.append(weights[i, j])
#     return np.array(X), np.array(y)

# X, y = compute_features(recurrent_weights.cpu().numpy(), embedding_2d)
# # Plot X and y
# plt.figure(figsize=(8, 6))
# plt.scatter(X, y, alpha=0.5)
# plt.xlabel('Distance between Neurons')
# plt.ylabel('Weight Value')
# plt.title('Weight Value vs Distance between Neurons')
# plt.tight_layout()
# plt.savefig(f'output/rnn_{load_data_type}_{num_neuron}_weights_distance.png')
# model = LinearRegression().fit(X, y)
# print("R^2:", r2_score(y, model.predict(X)))


print(embedding_1d.shape, embedding_2d.shape, embedding_3d.shape)

# Plot the weights sorted by 1d embeddings
sorted_idx = np.argsort(embedding_1d[:, 0])  
reorg_weights = recurrent_weights[sorted_idx, :][:, sorted_idx]
plt.figure(figsize=(8, 6))
plt.imshow(reorg_weights.cpu().numpy(), aspect='auto', cmap='plasma', vmin=vmin, vmax=vmax)
plt.colorbar(label='Weight Value')
plt.title('Recurrent Layer Weights (Reorganized 2)')
plt.xlabel('Neuron Index')
plt.ylabel('Neuron Index')
plt.tight_layout()
plt.savefig(f'output/rnn_{load_data_type}_{num_neuron}_reorg_weights_1.png')

# Plot the weights sorted by 2d embeddings angle
sorted_idx = np.argsort(angles_2d)
reorg_weights = recurrent_weights[sorted_idx, :][:, sorted_idx] 
plt.figure(figsize=(8, 6))
plt.imshow(reorg_weights.cpu().numpy(), aspect='auto', cmap='plasma', vmin=vmin, vmax=vmax)
plt.colorbar(label='Weight Value')
plt.title('Recurrent Layer Weights (Reorganized 2)')
plt.xlabel('Neuron Index')
plt.ylabel('Neuron Index')
plt.tight_layout()
plt.savefig(f'output/rnn_{load_data_type}_{num_neuron}_reorg_weights_2.png')


# Pick the first neuron, and find the distance to all other neurons
next_idx = 0
update_idx = 0
sorted_idx = [next_idx]
while update_idx < len(embedding_3d)-1:
    pick_location = embedding_3d[next_idx, :]
    distances = np.linalg.norm(embedding_3d - pick_location, axis=1)
    # The next index is the one with the smallest distance, excluding the current index
    distances[sorted_idx] = np.inf  # Exclude the current index
    next_idx = np.argsort(distances)[0]
    sorted_idx.append(next_idx)
    print(x3d[next_idx], y3d[next_idx])
    update_idx += 1
print(len(sorted_idx))
reorg_weights = recurrent_weights[sorted_idx, :][:, sorted_idx]
plt.figure(figsize=(8, 6))
plt.imshow(reorg_weights.cpu().numpy(), aspect='auto', cmap='plasma', vmin=vmin, vmax=vmax)
plt.colorbar(label='Weight Value')
plt.title('Recurrent Layer Weights (Reorganized 3)')
plt.xlabel('Neuron Index')
plt.ylabel('Neuron Index')
plt.tight_layout()
plt.savefig(f'output/rnn_{load_data_type}_{num_neuron}_reorg_weights_3z.png',
            transparent=True)

# Create an animation of the 3D embedding in the order of the sorted indices
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
# Create the Gif
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter([], [], [], c=[], cmap='plasma', s=10)
ax.set_xlabel('W1')
ax.set_ylabel('W2')
ax.set_zlabel('W3') 

def update(frame):
    ax.clear()
    ax.set_xlabel('W1')
    ax.set_ylabel('W2')
    ax.set_zlabel('W3')
    # Plot the points in the order of the sorted indices
    ax.scatter(x3d[sorted_idx[:frame+1]], y3d[sorted_idx[:frame+1]], z3d[sorted_idx[:frame+1]], 
               c=np.arange(frame+1), cmap='plasma', s=10)
    return sc

ani = FuncAnimation(fig, update, frames=len(sorted_idx), interval=100, repeat=False)
ani.save(f'output/rnn_{load_data_type}_{num_neuron}_weights_embedding3_animation.gif', writer='pillow', fps=10)