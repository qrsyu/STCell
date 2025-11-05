from matplotlib import pyplot as plt

colours = [ '#274753', '#297270', '#299D8F',"#4E975E", '#8AB07C',# '#D3D7AF',
           '#E7C66B', '#F3A361', '#E66D50', "#A3432B"]

num_place_cells = [129, 109, 85, 68, 60, 53, 53, 20, 2]
num_place_cells.reverse()

num_spatial_channels = [4, 10, 20, 40, 50, 60, 80, 90, 96]
num_temporal_channels = [100 - n for n in num_spatial_channels]

# Generate a histogram
fig, ax = plt.subplots(figsize=(4, 3))
ax.bar(num_temporal_channels, num_place_cells, color=colours, width=3, edgecolor='black')
# Plot a dashed line from (4,129) to (96,2)
ax.plot([4, 96], [129, 2], linestyle='--', color='black', alpha=0.5)
ax.set_xlabel('Number of temporal channels')
ax.set_ylabel('Number of place cells')
plt.tight_layout()
plt.savefig('code/fig/fig4_hist.png', transparent=True,dpi=500)