import matplotlib.pyplot as plt

rows = [
    {'name': 'Trial 0', 'bars': [(10*0,   10*0.5),
                                 (10*0.5, 10*1)]},
    {'name': 'Trial 1', 'bars': [(10*0.05, 10*0.45),
                                 (10*0.55, 10*0.95)]},
    {'name': 'Trial 2', 'bars': [(10*0.1, 10*0.4),
                                 (10*0.6, 10*0.9)]},
    {'name': 'Trial 3', 'bars': [(10*0.15, 10*0.35),
                                 (10*0.65, 10*0.85)]},
    {'name': 'Trial 4', 'bars': [(10*0.2, 10*0.3),
                                 (10*0.7, 10*0.8)]},
    {'name': 'Trial 5', 'bars': [(10*0.25, 10*0.3),
                                 (10*0.75, 10*0.8)]},
]

# each trial has one colours
colours = ['#274753', '#297270', '#299D8F', '#8AB07C',  '#E7C66B', '#F3A361']

fig, ax = plt.subplots(figsize=(5, 4))

yticks = []
yticklabels = []
for i, row in enumerate(rows):
    y = len(rows) - i 
    yticks.append(y)
    yticklabels.append(row['name'])
    # The first bar surrounded by a box full color filled, the second bar surrounded by a box partially filled
    for j, bar in enumerate(row['bars']):
        start = bar[0]
        end = bar[1]
        if j == 0:
            ax.barh(y, end - start, left=start, height=0.4, color=colours[i], edgecolor='black')
        else:
            # colored edge and filled transparent
            ax.barh(y, end - start, left=start, height=0.4, color=colours[i], alpha=0.5, edgecolor='black')
    # Fill the other region in the row with black bars
    ax.barh(y, row['bars'][0][0], left=0, height=0.4, color='black', edgecolor='black',)
    ax.barh(y, 10 - row['bars'][1][1], left=row['bars'][1][1], height=0.4, color='black', edgecolor='black',)
    # Fill the migion region in the row with black bars
    ax.barh(y, row['bars'][1][0] - row['bars'][0][1], left=row['bars'][0][1], height=0.4, color='black', edgecolor='black',)

    # Label solid bar as "X" and transparent bar as "Y"
    ax.text(row['bars'][0][0] + (row['bars'][0][1] - row['bars'][0][0]) / 2, y, 'X',
            ha='center', va='center', color='white', fontsize=12)
    # ax.text(row['bars'][1][0] + (row['bars'][1][1] - row['bars'][1][0]) / 2, y, 'Y',
    #         ha='center', va='center', color=colours[i], fontsize=12)
        

ax.set_yticks(yticks)
ax.set_yticklabels(yticklabels)

ax.set_xlim(0, 10)

plt.xlabel('Time (s)')


plt.tight_layout()
plt.savefig('output/fig3_b.png', transparent=True)
