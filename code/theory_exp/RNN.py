import numpy as np
from dataclasses import dataclass
from typing import Sequence, Union

def pairwise_sqdist(A, B, periodic=None, box_size=None):
    """
    计算集合 A 与集合 B 中所有点对的**欧氏距离平方**；可选在**周期边界(环面)**下使用最小镜像距离。

    参数
    ----
    A : array_like, shape = (M, D) 或 (D,)
        M 个点在 D 维空间中的坐标。若传入一维 (D,), 会自动变成 (1, D)。
        单位/量纲与 B、box_size 保持一致（例如米、相位等）。

    B : array_like, shape = (N, D) 或 (D,)
        N 个点在同一个 D 维空间中的坐标。若传入一维 (D,), 会自动变成 (1, D)。

    periodic : bool 或 长度为 D 的序列, default=False
        - 若为 bool，则所有维度同一设置（与旧版一致）。
        - 若为序列（长度 D），逐维指定是否周期，比如 (True, True, False)。

    box_size : float 或 长度为 D 的序列, optional
        周期盒在每个维度的边长 L；仅当 periodic=True 时必填。
        - 若为标量，表示各维度边长相同；
        - 若为序列，长度必须为 D，对应每一维的边长。

    返回
    ----
    d2 : np.ndarray, shape = (M, N)
        A 与 B 的**两两距离平方矩阵**。d2[i, j] = ||A[i] - B[j]||^2
        （若 periodic=True，则差值先按 box_size 做周期 wrap）。
    """
    A = np.atleast_2d(A); B = np.atleast_2d(B)
    Aexp = A[:, None, :]; Bexp = B[None, :, :]
    diff = Aexp - Bexp
    
    if isinstance(periodic, (bool, np.bool_)):
        any_periodic = bool(periodic)
        mask = np.full((1, 1, A.shape[1]), periodic, dtype=bool)
    else:
        mask_arr = np.asarray(periodic, dtype=bool)
        if mask_arr.ndim != 1 or mask_arr.size != A.shape[1]:
            raise ValueError("`periodic` must be a bool or a sequence of length D.")
        any_periodic = mask_arr.any()
        mask = mask_arr.reshape(1, 1, A.shape[1])
        
    if any_periodic:
        if box_size is None:
            raise ValueError("`box_size` must be provided when any periodic dimension is True.")
        if np.isscalar(box_size):
            L = np.full((1, 1, A.shape[1]), float(box_size))
        else:
            L_arr = np.asarray(box_size, dtype=float)
            if L_arr.ndim != 1 or L_arr.size != A.shape[1]:
                raise ValueError("`box_size` must be a scalar or a sequence of length D.")
            L = L_arr.reshape(1, 1, A.shape[1])
    wrapped = (diff + 0.5 * L) % L - 0.5 * L  
    diff = np.where(mask, wrapped, diff)   
    
    return np.sum(diff * diff, axis=-1)  # (M,N)
    # if any_periodic:
    #     if box_size is None:
    #         raise ValueError("box_size must be provided for periodic distance.")
    #     L = np.array(box_size).reshape(1,1,-1)
    #     diff = (diff + 0.5*L) % L - 0.5*L
    # return np.sum(diff*diff, axis=-1)

def gaussian_kernel_from_coords(neuron_coords, src_coords, sigma, amp=1.0, norm=True, periodic=None, box_size=None):
    """
    根据坐标生成高斯核权重矩阵 W(x - y)。

    参数
    ----
    neuron_coords : array_like, shape = (N, D)
        N 个神经元在 D 维空间中的坐标（如 1D 环、2D 平面、3D 等）。
        单位/量纲需与 src_coords、sigma、box_size 一致。

    src_coords : array_like, shape = (M, D)
        M 个“源点”的坐标。可以是输入通道的位置，或另一组神经元的位置。

    sigma : float
        高斯核的尺度（标准差）。标量、各向同性。若需要各向异性，请改成协方差形式。

    amp : float, default=1.0
        核的整体幅度系数（强度）。

    norm : bool, default=True
        是否做连续情形的“单位积分”归一化：除以 (sqrt(2π)*sigma)^D。
        注意：在离散格点上这是近似归一；严格做法还需乘以体素体积。

    periodic : bool, default=False
        若为 True，则在周期盒中计算最短距离（最小镜像法）。

    box_size : float 或 长度为 D 的序列, optional
        每一维的周期长度 L。仅在 periodic=True 时必填。
        标量表示各维相同；序列长度必须为 D。

    返回
    ----
    K : np.ndarray, shape = (N, M)
        高斯核权重矩阵，K[i, j] = amp * exp(-||x_i - y_j||^2 / (2 sigma^2)) / norm_const。
        常用作：
          - 递归权重 W_rc ：neuron_coords 对 src_coords=neuron_coords；
          - 输入权重 W_in： neuron_coords 对 src_coords=输入通道坐标。

    异常
    ----
    ValueError:
        - periodic=True 但未提供 box_size；
        - neuron_coords 与 src_coords 的维度 D 不相等。
    """
    sq = pairwise_sqdist(neuron_coords, src_coords, periodic=periodic, box_size=box_size)
    K = np.exp(-0.5 * sq / (sigma**2))
    if norm:
        const = (np.sqrt(2*np.pi)*sigma)**(neuron_coords.shape[1])
        K = amp * K / const
    else:
        K = amp * K
    return K

@dataclass
class ExperienceCANN:
    neuron_coords: np.ndarray
    channel_coords: np.ndarray
    alpha: float = 0.3
    sigma_rec: float = 0.1
    sigma_in: float = 0.1
    W0_rc: float = 1.0
    W0_in: float = 1.0
    bias: np.ndarray = None
    gain: float = 1.0
    nonlinearity: str = "relu"
    periodic: Union[bool, Sequence[bool]] = False   
    box_size: Union[float, Sequence[float], None] = None
    divisive_norm: bool = True

    def __post_init__(self):
        N = self.neuron_coords.shape[0]
        self.W_rc = gaussian_kernel_from_coords(self.neuron_coords, self.neuron_coords, self.sigma_rec, self.W0_rc, True, self.periodic, self.box_size)
        self.W_in = gaussian_kernel_from_coords(self.neuron_coords, self.channel_coords, self.sigma_in, self.W0_in, True, self.periodic, self.box_size)
        if self.bias is None:
            # self.bias = np.zeros(N, dtype=float)
            self.bias = np.ones(N)

    def phi(self, V):
        if self.nonlinearity == "relu":
            return np.maximum(0.0, self.gain * V)
        elif self.nonlinearity == "softplus":
            return (1.0/self.gain) * np.log1p(np.exp(self.gain * V))
        else:
            raise ValueError("Unknown nonlinearity")

    def step(self, V, e_t):
        r = self.phi(V)
        if self.divisive_norm:
            r = r / np.maximum(1e-6, r.sum(axis=1, keepdims=True))
        rec = (self.W_rc @ r.T).T
        inn = (e_t @ self.W_in.T)
        dV = (-V + rec + inn + self.bias)
        V_new = V + self.alpha * dV
        r_new = self.phi(V_new)
        if self.divisive_norm:
            r_new = r_new / np.maximum(1e-6, r_new.sum(axis=1, keepdims=True))
        return V_new, r_new

    def run(self, e):
        B, T, C = e.shape
        N = self.neuron_coords.shape[0]
        V = np.zeros((B, N), dtype=float) # 初始化膜电位为零
        r_all = np.zeros((B, T, N), dtype=float) # 记录所有时间步的输出
        for t in range(T):
            V, r = self.step(V, e[:, t, :])
            r_all[:, t, :] = r
        return r_all