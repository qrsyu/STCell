import matplotlib.pyplot as plt

rows = [
    {'name': 'Trial 1', 'edge': 4},
    {'name': 'Trial 2', 'edge': 10},
    {'name': 'Trial 3', 'edge': 50},
    {'name': 'Trial 4', 'edge': 90},
    {'name': 'Trial 5', 'edge': 96},
]

# each trial has one colours
colours = [# '#274753', 
           '#297270', '#299D8F', '#8AB07C',  '#E7C66B', '#F3A361']

fig, ax = plt.subplots(figsize=(4, 3))

yticks = []
yticklabels = []
for i, row in enumerate(rows):
    y = len(rows) - i 
    yticks.append(y)
    yticklabels.append(row['name'])
    # The first bar surrounded by a box full color filled, the second bar surrounded by a box partially filled
    
    # Plot the left bar
    ax.barh(y, row['edge'], left=0, height=0.4, color=colours[i], edgecolor='black')
    # Plot the right bar
    ax.barh(y, 100-row['edge'], left=row['edge'], height=0.4, color=colours[i], alpha=0.5, edgecolor='black')

    # # Label solid bar as "X" and transparent bar as "Y"
    # ax.text(row['bars'][0][0] + (row['bars'][0][1] - row['bars'][0][0]) / 2, y, 'X',
    #         ha='center', va='center', color='white', fontsize=12)
        

ax.set_yticks(yticks)
ax.set_yticklabels(yticklabels)

ax.set_xlim(0, 100)

plt.xlabel('Time (s)')


plt.tight_layout()
plt.savefig('output/fig4_b.png')
