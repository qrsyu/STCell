from matplotlib_venn import venn2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, axes = plt.subplots(1, 2, figsize=(9, 4))

data = [
    ((282 - 47, 80 - 47, 47), '1st lap'),
    ((208 - 6, 54 - 6, 6), '2nd lap'),
]

# First pass: draw venns and collect bounds
venns = []
for ax, (subsets, title) in zip(axes, data):
    v = venn2(subsets=subsets, set_labels=('Place cells', 'Time cells'), ax=ax)
    v.get_patch_by_id('10').set_color('#85B7EB')
    v.get_patch_by_id('01').set_color('#F09595')
    v.get_patch_by_id('11').set_color('#B09ACD')
    for pid in ['10', '01', '11']:
        v.get_patch_by_id(pid).set_alpha(0.5)
        v.get_patch_by_id(pid).set_zorder(2)
    for label in v.set_labels + v.subset_labels:
        if label:
            label.set_zorder(3)
    ax.set_title(title)
    venns.append(v)

# Second pass: add same-sized gray box to both
# Use the larger subplot's span so the box fits both
all_xlims = [ax.get_xlim() for ax in axes]
all_ylims = [ax.get_ylim() for ax in axes]
max_span_x = max(xl[1] - xl[0] for xl in all_xlims)
max_span_y = max(yl[1] - yl[0] for yl in all_ylims)
max_total = max(sum(d[0]) for d in data)
box_w = max_span_x * (512 / max_total) ** 0.5
box_h = max_span_y * (512 / max_total) ** 0.5

for ax in axes:
    cx = sum(ax.get_xlim()) / 2
    cy = sum(ax.get_ylim()) / 2
    rect = mpatches.FancyBboxPatch(
        (cx - box_w/2, cy - box_h/2), box_w, box_h,
        boxstyle="round,pad=0.02",
        facecolor='#E8E8E8', edgecolor='#999999',
        alpha=0.3, zorder=0
    )
    ax.add_patch(rect)
    ax.text(cx - box_w/2 + 0.02, cy + box_h/2 - 0.02, '512 neurons',
            fontsize=9, color='gray', ha='left', va='top')
    ax.set_xlim(cx - box_w/2 - 0.05, cx + box_w/2 + 0.05)
    ax.set_ylim(cy - box_h/2 - 0.05, cy + box_h/2 + 0.05)

fig.tight_layout()
fig.savefig('code/spacetime_exp/place_time_venn.png', dpi=300, bbox_inches='tight')