import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt



def _phi(V, gain, kind):
    if kind=="relu": return np.maximum(0, gain*V)
    if kind=="softplus": return np.log1p(np.exp(gain*V))/gain

def _dphi(V, gain, kind):
    if kind=="relu": return (V>0).astype(float)*gain
    if kind=="softplus":
        z = gain*V; return 1.0/(1.0+np.exp(-z))

# ---- 2) Jacobian（连续 & 离散）----

def _jacobian_mats(Wrc, Vt, gain, nonlin, divisive_norm, alpha): 
    """ 连续: Vdot = -V + Wrc*r -> J_c = -I + Wrc @ D 
    离散: V+ = V + alpha*(-V + Wrc*r) -> J_d = I + alpha * J_c 
    其中 D = ∂r/∂V；若开启分式归一化 r = φ(V)/sum φ(V)， 则 D = diag(φ')/s - (φ ⊗ φ')/s^2 (秩-1修正) """ 
    u = _phi(Vt, gain, nonlin) 
    du = _dphi(Vt, gain, nonlin) 
    N = Vt.size 
    if not divisive_norm: 
        D = np.diag(du) # (N,N) 
    else: 
        s = max(1e-8, u.sum()) # 标量 
        D = np.diag(du)/s - np.outer(u, du)/(s*s) # (N,N) 
    Jc = -np.eye(N) + Wrc @ D 
    Jd = np.eye(N) + alpha * Jc 
    return Jc, Jd

# ---- 3) 主函数：沿时间计算 Jacobian 稳定性 ----
def compute_jacobian_over_time(net, E, return_full_spectrum=False):
    """
    net : ExperienceCANN 实例（已构造好 W_rc / W_in 等）
    E   : 形状 (1, T, C) 或 (T, C) 的输入序列（仅支持 batch=1）
    return_full_spectrum : 若为 True，还返回每个 t 的特征值数组（可能较占内存）

    返回:
      max_real_cont      : (T,) 连续系统最大特征值实部  max Re(λ(J_c(t)))
      spectral_radius_disc: (T,) 离散步进谱半径       ρ(J_d(t))
      (可选) eigs_c, eigs_d: list 长度 T，每个元素是一维复数组（该时刻的全谱）
    """
    # 统一成 (1,T,C)
    if E.ndim == 2:     # (T,C)
        E = E[None, :, :]

    B, T, C = E.shape
    N = net.neuron_coords.shape[0]

    # 先正向一次，收集每步的 V_t
    V = np.zeros((B, N), dtype=float)
    Vs = np.empty((B, T, N), dtype=float)
    for t in range(T):
        V, _ = net.step(V, E[:, t, :])  # your step already supports batch
        Vs[:, t, :] = V

    # 逐时刻计算 Jacobian 与稳定性指标
    max_real_cont      = np.empty((B, T), dtype=float)
    spectral_radius_disc = np.empty((B, T), dtype=float)

    eigs_c_list = np.empty((B, T, N), dtype=complex) if return_full_spectrum else None
    eigs_d_list = np.empty((B, T, N), dtype=complex) if return_full_spectrum else None

    for b in tqdm(range(B)):
        for t in range(T):
            Jc, Jd = _jacobian_mats(
                Wrc=net.W_rc,
                Vt=Vs[b, t, :],
                gain=net.gain,
                nonlin=net.nonlinearity,
                divisive_norm=net.divisive_norm,
                alpha=net.alpha,   # dt/tau
            )
            eig_c = np.linalg.eigvals(Jc)
            eig_d = np.linalg.eigvals(Jd)
            max_real_cont[b, t]       = np.max(np.real(eig_c))
            spectral_radius_disc[b, t] = np.max(np.abs(eig_d))

            if return_full_spectrum:
                eigs_c_list[b, t, :] = eig_c
                eigs_d_list[b, t, :] = eig_d

    if return_full_spectrum:
        return max_real_cont, spectral_radius_disc, eigs_c_list, eigs_d_list
    else:
        return max_real_cont, spectral_radius_disc
    
    
def plot_realpart_heatmap(eigs, bins=120, rmin=None, rmax=None, title=None):
    """
    用 2D 直方图 (time × Re(λ)) 可视化实部的分布随时间的变化。
    """
    arr = np.asarray(eigs)
    if arr.ndim == 3: arr = arr[0]
    T, N = arr.shape
    re = np.real(arr)

    if rmin is None: rmin = float(np.min(re))
    if rmax is None: rmax = float(np.max(re))
    # 构造 2D 直方图：时间轴离散到 T 个 bin，实部分到 bins 个 bin
    H, xedges, yedges = np.histogram2d(
        np.repeat(np.arange(T), N),
        re.ravel(),
        bins=[T, bins],
        range=[[0, T], [rmin, rmax]],
    )

    plt.figure(figsize=(6.8, 4.6))
    plt.imshow(H.T, origin='lower', aspect='auto',
               extent=[0, T, rmin, rmax])
    # plt.axhline(0.0, linestyle='--', linewidth=1)
    plt.colorbar(label='count')
    plt.xlabel('time step')
    plt.ylabel('Re(λ)')
    plt.title(title or 'Eigenspectrum (Re parts) over time')
    plt.tight_layout()
    return plt


def max_lyapunov_exponent(net, E, return_logs=False):
    """
    计算最大 Lyapunov 指数（batch=1）。
    - net: ExperienceCANN
    - E: (T,C) 或 (1,T,C) 输入序列
    - burn_in: 前若干步不计入平均（让轨道先“就位”）
    - use_continuous: False→用离散 Jd，True→用连续 Jc * alpha 近似一步
    - return_ftle: 返回每步的即时有限时间 Lyapunov 指数（可作时序曲线）
    返回:
      mle, (可选) ftle_series
    """
    # 统一形状
    if E.ndim == 2:     # (T,C)
        E = E[None, :, :]
    B, T, C = E.shape
    N = net.neuron_coords.shape[0]

    # 先跑真实轨道，保存 V_t
    V = np.zeros((B, N), dtype=float)
    Vs = np.zeros((B, T, N), dtype=float)
    for t in range(T):
        V, _ = net.step(V, E[:, t, :])
        Vs[:, t, :] = V

    # 初始化微扰向量（单位长度）
    dv = np.random.randn(N)
    dv /= (np.linalg.norm(dv) + 1e-12)

    logs = np.empty((B, T))  # 收集 log 放大率
    
    for b in range(B):
        for t in range(T):
            # 组装本步的雅可比
            Jc, _ = _jacobian_mats(net.W_rc, Vs[b, t, :], 
                                   net.gain, net.nonlinearity, net.divisive_norm, net.alpha)
            M = np.eye(N) + net.alpha * Jc

            # 传播微扰
            dv = M @ dv
            norm = np.linalg.norm(dv) + 1e-18
            logs[b, t] = np.log(norm)
            # 重新归一化，避免数值漂移
            dv /= norm

    # 平均化得到 MLE（单位：每步；若需“每秒”，再除以 dt/τ=alpha 或乘以 1/dt）
    mle_per_step = np.mean(logs, axis=1)
    if return_logs:
        # 可选：转成“每步”的即时指数（或进一步滑动平均）
        return mle_per_step, logs
    return mle_per_step