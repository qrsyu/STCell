from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm


def plt_hs(hs, min_fr=0.1, masks=None, fig=None, ax=None):

    time_points = hs.shape[0]

    # Select neurons with mean firing rate > 0.1
    mean_fr = hs.mean(axis=0)
    # Get the index where mean_fr > min_fr
    mask = mean_fr > min_fr
    neuron_indices = np.where(mask)[0]
    select_hs = hs[:, neuron_indices]
    del hs

    # Normalize the hs along time points (Sure to be correct!)
    # norm_hs = select_hs / np.linalg.norm(select_hs, axis=0, keepdims=True)
    # 0-1 Normalise method 2
    norm_hs = np.zeros_like(select_hs)
    for n in range(select_hs.shape[1]):
        norm_hs[:, n] = (select_hs[:, n]-np.min(select_hs[:, n]))/(np.max(select_hs[:, n])-np.min(select_hs[:, n]))
    del select_hs
    
    # Sort neurons from maximum firing time
    max_time = np.argmax(norm_hs, axis=0)
    sorted_neuron_indices = np.argsort(max_time)
    norm_hs = norm_hs[:, sorted_neuron_indices]
    num_neurons = norm_hs.shape[1]
    
    # Plot the normalized hs
    ax.imshow(norm_hs.T, aspect='auto', cmap='jet',  
              extent=[0, time_points/10, 0, num_neurons]
              )
    # ax.set_xlabel('Time (s)')
    ax.set_ylabel('Neurons')
    ax.set_xticks(np.linspace(0, norm_hs.shape[0]/10, 6))

    if masks is not None:
        for m in masks:
            ax.axvline(x=m[0], color='white', linestyle='--', linewidth=1)
            ax.axvline(x=m[1], color='white', linestyle='-', linewidth=1)
            # Plot a semi-transparent rectangle to cover the masked region
            ax.add_patch(plt.Rectangle((m[0], 0), m[1]-m[0], norm_hs.shape[1], 
                                       color='white', alpha=0.3))
    
    # # Plot the colorbar
    # cbar = fig.colorbar(ax.images[0], ax=ax)
    # cbar.set_label('Normalized firing rate')
    return norm_hs, fig, ax


def plt_temp_corr(hs, fig, ax, corr_inteval=[0, 100], corr_color=['skyblue', 'salmon']):
    import seaborn as sns
    from scipy.optimize import curve_fit

    max_times = np.argmax(hs, axis=0)
    max_times = max_times / 10  # Convert to seconds
    # Get the temporal firing width of each neuron, the threshold is 0.1 of the maximum firing rate
    firing_widths = np.zeros(hs.shape[1])
    for i in range(hs.shape[1]):
        firing_widths[i] = np.sum(hs[:, i] > 1E-1)
        # firing_widths[i] = np.sum(norm_hs[:, i] > 0.5 * np.max(norm_hs[:, i]))
    
    colors = [corr_color[0] if t < corr_inteval[0] or t > corr_inteval[1] else corr_color[1] for t in max_times]

    plt.scatter(max_times, firing_widths, c=colors, s=10)


    rval = np.corrcoef(max_times[(max_times >= corr_inteval[0]) & (max_times < corr_inteval[1])], 
                   firing_widths[(max_times >= corr_inteval[0]) & (max_times < corr_inteval[1])])[0,1]
    print(f'Correlation coefficient: {rval}')
    sns.regplot(x=max_times[(max_times >= corr_inteval[0]) & (max_times < corr_inteval[1])], 
            y=firing_widths[(max_times >= corr_inteval[0]) & (max_times < corr_inteval[1])],
            scatter=False, 
            color='black',
            line_kws={"linewidth":1, "linestyle":"-"},
            ci=95,
            label=f'r = {rval:.2f}')

    def fit_func(x, a, b):
        return a * x + b
    popt, pcov = curve_fit(fit_func, max_times, firing_widths)
    print(f'Fitted gradient: {popt[0]:.4f}')

    # plt.xlabel('Maximum firing time (s)')
    plt.ylabel("Firing width (s)")
    plt.legend()
    
    ax.set_xticks(np.linspace(0, hs.shape[0]/10, 6))
     
    return fig, ax


def compute_occupancy(pos, bins):
    H, W = bins
    occ_map = np.zeros((H, W))

    all_pos = pos.reshape(-1, 2)

    for p in all_pos:
        x, y = p[0], p[1]
        # Find the nearest integer grid point
        x = int(np.clip(np.round(x), 0, W-1))
        y = int(np.clip(np.round(y), 0, H-1))
        occ_map[x, y] += 1

    # # smoothing
    # from scipy.ndimage import gaussian_filter
    # occ_map = gaussian_filter(occ_map, sigma=sigma)

    # normalize
    occ_map /= np.sum(occ_map)
    return occ_map


def _compute_SIC(rate_map, occupancy_map, eps=1e-12):
    """
    rate_map: (H, W)
    occupancy_map: (H, W), must sum to 1
    """
    
    mean_rate = np.nansum(rate_map * occupancy_map)
    if mean_rate < eps:
        # print('False')
        return 0.0

    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = rate_map / mean_rate
        log_term = np.log2(ratio + eps)
        info = np.nansum(occupancy_map * (rate_map / mean_rate) * log_term)
        return info
    
    
def SIC_analysis(ratemaps, occupancy_map, threshold=3):
    """
    ratemaps: (N, H, W)
    occupancy_map: (H, W), should sum to 1
    """

    N = ratemaps.shape[0]
    spatial_infos = np.zeros(N)
    is_place_cell = np.zeros(N, dtype=bool)

    # Compute real spatial info
    for i in tqdm(range(N), desc='Computing SIC for neurons'):
        spatial_infos[i] = _compute_SIC(ratemaps[i], occupancy_map)
    
    is_place_cell = spatial_infos > threshold

    return spatial_infos, is_place_cell


def ratemap_to_angle_profile(ratemaps, nbins=18, radius=None):

    N, X, _ = ratemaps.shape

    yy, xx = np.mgrid[0:X, 0:X]

    cy = X / 2.0
    cx = X / 2.0

    dy = yy - cy
    dx = xx - cx
    angles = np.arctan2(dy, dx)  # [-pi, pi)
    dists  = np.hypot(dy, dx)

    if radius is None:
        radius = np.min([cx, cy, X-1-cx, X-1-cy])
    # print(radius)
    
    circle_mask = dists <= radius
    
    edges = np.linspace(-np.pi, np.pi, nbins + 1, endpoint=True)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])

    ang_flat = angles[circle_mask].ravel()
    R_flat = ratemaps[:, circle_mask] 

    ratemap_angle = np.zeros((N, nbins), dtype=float)
    counts = np.histogram(ang_flat, bins=edges)[0]
    for i in range(N):
        
        Ri = R_flat[i]                  
        valid = np.isfinite(Ri)             
        ang_i  = ang_flat[valid]
        R_i    = Ri[valid]
        
        sums = np.histogram(ang_i, bins=edges, weights=R_i)[0]
        ratemap_angle[i] = np.divide(
            sums, counts,
            out=np.zeros_like(sums, dtype=float),
            where=(counts > 0)
        )
    return ratemap_angle, bin_centers, radius