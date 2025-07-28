from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm


def plt_hs(hs, min_fr=0.1, fig=None, ax=None):

    time_points, num_neurons = hs.shape[0], hs.shape[1]

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
    
    # Plot the normalized hs
    ax.imshow(norm_hs.T, aspect='auto', cmap='jet',  
              extent=[0, time_points/10, 0, num_neurons]
              )
    # ax.set_xlabel('Time (s)')
    ax.set_ylabel('Neurons')
    ax.set_xticks(np.linspace(0, norm_hs.shape[0]/10, 6))
    
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

# def compute_occupancy_map(positions, arena_shape, smoothing_sigma=1.0):
#     """
#     positions: (batch_size, time, 2), 每个时间点的位置 (x, y)
#     arena_shape: (H, W), 你的 ratemap 的空间分辨率
#     """
#     H, W = arena_shape
#     occupancy = np.zeros((H, W))

#     # 合并 batch
#     all_positions = positions.reshape(-1, 2)

#     # 把位置坐标离散化到 0~H-1, 0~W-1 网格上
#     x = np.clip(all_positions[:, 0].astype(int), 0, W - 1)
#     y = np.clip(all_positions[:, 1].astype(int), 0, H - 1)

#     # 计数
#     for i in range(len(x)):
#         occupancy[y[i], x[i]] += 1

#     # 可选平滑
#     from scipy.ndimage import gaussian_filter
#     occupancy = gaussian_filter(occupancy, sigma=smoothing_sigma)

#     # 归一化
#     occupancy = occupancy / np.sum(occupancy)

#     return occupancy


def compute_coarse_occupancy(pos, old_bins, new_bins=(10, 10), sigma=1):
    coarse_pos = coarse_grain_positions(pos, old_bins, new_bins)
    H, W = new_bins
    occ_map = np.zeros((H, W))

    all_pos = coarse_pos.reshape(-1, 2)
    for p in all_pos:
        y, x = p[1], p[0]
        occ_map[y, x] += 1

    # smoothing
    from scipy.ndimage import gaussian_filter
    occ_map = gaussian_filter(occ_map, sigma=sigma)

    # normalize
    occ_map /= np.sum(occ_map)
    return occ_map


def compute_SIC(rate_map, occupancy_map, eps=1e-12):
    """
    rate_map: (H, W)
    occupancy_map: (H, W), must sum to 1
    """
    
    mean_rate = np.nansum(rate_map * occupancy_map)
    if mean_rate < eps:
        print('False')
        return 0.0

    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = rate_map / mean_rate
        log_term = np.log2(ratio + eps)
        info = np.nansum(occupancy_map * rate_map / mean_rate * log_term)
        return info
    
    
def SIC_analysis(ratemaps, occupancy_map, threshold=3):
    """
    ratemaps: (N, H, W)
    occupancy_map: (H, W), should sum to 1
    """
    
    N, H, W = ratemaps.shape
    spatial_infos = np.zeros(N)
    p_values = np.zeros(N)
    is_place_cell = np.zeros(N, dtype=bool)

    # Compute real spatial info
    for i in tqdm(range(N), desc='Computing SIC for neurons'):
        spatial_infos[i] = compute_SIC(ratemaps[i], occupancy_map)
    
    is_place_cell = spatial_infos > threshold

    return spatial_infos, is_place_cell


def coarse_grain_positions(positions, old_bins = (54, 54), new_bins=(10, 10)):
    """
    positions: (B, T, 2), 每个点是 (x, y)
    返回 coarse 之后的整数 index 格子位置
    """
    
    scale_x = new_bins[1] / old_bins[1]
    scale_y = new_bins[0] / old_bins[0]

    scaled = positions.copy()
    scaled[..., 0] = positions[..., 0] * scale_x  # x
    scaled[..., 1] = positions[..., 1] * scale_y  # y

    # 转成整数 index
    scaled = np.floor(scaled).astype(int)
    scaled[..., 0] = np.clip(scaled[..., 0], 0, new_bins[1] - 1)
    scaled[..., 1] = np.clip(scaled[..., 1], 0, new_bins[0] - 1)
    return scaled


def coarse_ratemap(ratemaps, new_bins=(20, 20)):
    """
    ratemaps: shape (N, H, W)
    target_shape: new shape (h, w), coarse bins
    Returns: shape (N, h, w)
    """
    N, H, W = ratemaps.shape
    h, w = new_bins

    # 计算 coarse bin 的真实大小（可能是浮点）
    bin_height = H / h
    bin_width = W / w

    coarse_maps = np.zeros((N, h, w))

    for i in range(h):
        for j in range(w):
            # 对应原图的范围：从什么到什么
            y_start = int(np.floor(i * bin_height))
            y_end   = min(int(np.floor((i + 1) * bin_height)), H)
            x_start = int(np.floor(j * bin_width))
            x_end   = min(int(np.floor((j + 1) * bin_width)), W)

            # 平均这个块内的值
            if (y_end > y_start) and (x_end > x_start):
                block = ratemaps[:, y_start:y_end, x_start:x_end]
                # Should take care of nan
                block = np.where(np.isnan(block), 0, block)
                coarse_maps[:, i, j] = block.mean(axis=(1, 2))

    return coarse_maps