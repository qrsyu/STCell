from matplotlib import pyplot as plt

colours = ['#274753', '#297270', '#299D8F', '#8AB07C',  '#E7C66B', '#F3A361']

num_place_cells = [98, 36, 5, 11, 16, 9]

num_spatial_channels = [0,1,2,3,4,5]

# Generate a histogram
fig, ax = plt.subplots(figsize=(4, 3))
ax.plot(num_spatial_channels, num_place_cells, color='black', marker='o', markersize=8, linewidth=2)
# ax.bar(num_spatial_channels, num_place_cells, color=colours, width=3, edgecolor='black')
ax.set_xlabel('Trial')
ax.set_ylabel('Number of place cells')
plt.tight_layout()
plt.savefig('output/fig3_place_cells.png', transparent=True)