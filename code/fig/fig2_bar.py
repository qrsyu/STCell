import matplotlib.pyplot as plt

rows = [
    {'name': 'Trial 0', 'bars': [(10*0.25, 10*0.3),
                                 (10*0.75, 10*0.8)]},
    {'name': 'Trial 1', 'bars': [(10*0.2, 10*0.3),
                                 (10*0.7, 10*0.8)]},
    {'name': 'Trial 2', 'bars': [(10*0.15, 10*0.35),
                                 (10*0.65, 10*0.85)]},
    {'name': 'Trial 3', 'bars': [(10*0.1, 10*0.4),
                                 (10*0.6, 10*0.9)]},
    {'name': 'Trial 4', 'bars': [(10*0.05, 10*0.45),
                                 (10*0.55, 10*0.95)]},
    {'name': 'Trial 5', 'bars': [(10*0,   10*0.5),
                                 (10*0.5, 10*1)]},
        ]

# each trial has one colours
colours = ['#274753', '#297270', '#299D8F', '#8AB07C',  '#E7C66B', '#F3A361']

fig, ax = plt.subplots(figsize=(5, 4))
# # Plot the grid below the bars
# ax.grid(axis='x', linestyle='--', color='gray', alpha=0.7)
# ax.grid(axis='y', linestyle='--', color='gray', alpha=0.7)
# ax.set_axisbelow(True)

yticks = []
yticklabels = []
for i, row in enumerate(rows):
    y = len(rows) - i 
    yticks.append(y)
    yticklabels.append(row['name'])
    # The first bar surrounded by a box, the second bar surrounded by a dashed box
    for j, bar in enumerate(row['bars']):
        start = bar[0]
        end = bar[1]
        if j == 0:
            ax.barh(y, end - start, left=start, height=0.4, color=colours[i], edgecolor='black')
        else:
            # colored edge and filled transparent
            ax.barh(y, end - start, left=start, height=0.4, color='none', edgecolor=colours[i])

    # Label solid bar as "X" and transparent bar as "Y"
    ax.text(row['bars'][0][0] + (row['bars'][0][1] - row['bars'][0][0]) / 2, y, 'X',
            ha='center', va='center', color='white', fontsize=12)
    ax.text(row['bars'][1][0] + (row['bars'][1][1] - row['bars'][1][0]) / 2, y, 'Y',
            ha='center', va='center', color=colours[i], fontsize=12)
    
    # Draw a dashed line connecting the two bars (horizontally)
    ax.plot([row['bars'][0][1], row['bars'][1][0]], [y, y], color=colours[i], linestyle='--', linewidth=1)
    # Label the dashed line as 'Z'
    if i != 5:
        ax.text(5, y, 'Z', ha='center', va='bottom', color=colours[i], fontsize=12)    

ax.set_yticks(yticks)
ax.set_yticklabels(yticklabels)

ax.set_xlim(0, 10)

plt.xlabel('Time (s)')


plt.tight_layout()
plt.savefig('code/fig/fig2_bar.png')
