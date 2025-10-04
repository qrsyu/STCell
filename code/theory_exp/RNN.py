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
    
    Aexp = A[:, None, :]; Bexp = B[None, :, :]
    diff = Aexp - Bexp
    print('diff:', diff.shape)
    # If periodic is one boolean value, then set it for all dimensions
    if isinstance(periodic, (bool, np.bool_)):
        any_periodic = bool(periodic)
        mask = np.full((1, 1, A.shape[1]), periodic, dtype=bool)
    # Else periodic should be a sequence of booleans
    else:
        mask_arr = np.asarray(periodic, dtype=bool)
        if mask_arr.ndim != 1 or mask_arr.size != A.shape[1]:
            raise ValueError("`periodic` must be a bool or a sequence of length D.")
        any_periodic = mask_arr.any()
        mask = mask_arr.reshape(1, 1, A.shape[1])
        
    # If any dimension is periodic, we need box_size
    if any_periodic:
        if box_size is None:
            raise ValueError("`box_size` must be provided when any periodic dimension is True.")
        if np.isscalar(box_size):
            L = np.full((1, 1, A.shape[1]), float(box_size))
        else:
            L_arr = np.asarray(box_size, dtype=float)
            if L_arr.size != A.shape[1]:
                raise ValueError("`box_size` must be a scalar or a sequence of length D.")
            L = L_arr.reshape(1, 1, A.shape[1])
            
        # Wrapping for periodic dimensions
        wrapped = (diff + 0.5 * L) % L - 0.5 * L  
        # Calculated the new vector difference between two points
        diff = np.where(mask, wrapped, diff)   
    
    # Return the squared distance of the vector difference
    return np.sum(diff * diff, axis=-1)  # (M,N)

def pairwise_dist(A, B):
    Aexp = A[:, None, :]; Bexp = B[None, :, :]
    diff = Aexp - Bexp
    diff = np.reshape(diff, (A.shape[0], -1)) 
    print('diff time dim:', diff.shape)
    return diff

def gaussian_kernel_from_coords(neuron_coords, src_coords, Wtype, sigma, 
                                Lambda=None, amp=1.0, gamma=0.0, 
                                periodic=None, box_size=None):

    if Wtype == 'Gaussian':
        sq = pairwise_sqdist(neuron_coords, src_coords, periodic=periodic, box_size=box_size)
        K = amp * np.exp(-0.5 * sq / (sigma**2)) 
        const = (np.sqrt(2*np.pi)*sigma)**(neuron_coords.shape[1])

    elif Wtype == 'SpaceGaussian':
        sq_dist_space = pairwise_sqdist(neuron_coords[:,:2], src_coords[:,:2], periodic=periodic[:2], box_size=box_size[:2])
        dist_time = pairwise_dist(neuron_coords[:,2:], src_coords[:,2:])
        K = amp * np.exp(-0.5 * sq_dist_space / (sigma**2)) * np.exp(- Lambda * dist_time) 
        const = (np.sqrt(2*np.pi)*sigma)**(neuron_coords.shape[1]-1) / Lambda 
    
    K = K / const - gamma
    return K

@dataclass
class ExperienceCANN:
    neuron_coords: np.ndarray
    channel_coords: np.ndarray
    
    # Wrc
    W_rc_type: str
    sigma_rc: float 
    W0_rc: float 
    
    # Win
    W_in_type: str 
    sigma_in: float 
    W0_in: float 
    
    lambda_in: float = None
    lambda_rc: float = None
    
    gamma_in: float = 0.0
    gamma_rc: float = 0.0
    
    alpha: float = 0.3
    bias: np.ndarray = None
    gain: float = 1.0
    nonlinearity: str = "relu"
    periodic: Union[bool, Sequence[bool]] = False   
    box_size: Union[float, Sequence[float], None] = None
    divisive_norm: bool = True

    def __post_init__(self):
        N = self.neuron_coords.shape[0]
        self.W_rc = gaussian_kernel_from_coords(self.neuron_coords, self.neuron_coords, 
                                                self.W_rc_type, self.sigma_rc, 
                                                self.lambda_rc, self.W0_rc, self.gamma_rc,
                                                self.periodic, self.box_size)
        self.W_in = gaussian_kernel_from_coords(self.neuron_coords, self.channel_coords, 
                                                self.W_in_type, self.sigma_in, 
                                                self.lambda_in, self.W0_in, self.gamma_in,
                                                self.periodic, self.box_size)
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
        r = self.phi(V)           # (B,N)
        if self.divisive_norm:
            r = r / np.maximum(1e-6, r.sum(axis=1, keepdims=True))
        # W^rc @ r
        rec = (self.W_rc @ r.T).T # (B,N)
        # W^in @ e_t
        inn = (e_t @ self.W_in.T) # (B,N)
        dV = (-V + rec + inn + self.bias)
        V_new = V + self.alpha * dV
        r_new = self.phi(V_new)   # (B,N)
        if self.divisive_norm:
            r_new = r_new / np.maximum(1e-6, r_new.sum(axis=1, keepdims=True))
        return V_new, r_new

    def run(self, e):
        B, T, C = e.shape
        N = self.neuron_coords.shape[0]
        print('weight shape:', self.W_rc.shape, self.W_in.shape)
        V = np.zeros((B, N), dtype=float)        # Initialise V to zero
        r_all = np.zeros((B, T, N), dtype=float) # Record outputs for all time steps
        for t in range(T):
            V, r = self.step(V, e[:, t, :])
            r_all[:, t, :] = r
        return r_all