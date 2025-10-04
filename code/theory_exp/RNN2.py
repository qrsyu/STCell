import numpy as np
from dataclasses import dataclass
from typing import Sequence, Union

def pairwise_sqdist(A, B, periodic=None, box_size=None):
    
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

def gaussian_kernel_from_coords(neuron_coords, src_coords, Wtype, periodic=None, box_size=None,
                                sigma_e=None, amp_e=1.0, 
                                sigma_i=None, amp_i=0.0,
                                ):

    D = neuron_coords.shape[1]

    if Wtype == 'Gaussian':
        sq = pairwise_sqdist(neuron_coords, src_coords, periodic=periodic, box_size=box_size)
        K = amp_e * np.exp(-0.5 * sq / (sigma_e**2))
        const = (np.sqrt(2*np.pi)*sigma_e)**D
        return K / const

    elif Wtype == 'MexicanHat':
        if sigma_i is None or not (sigma_i > sigma_e > 0):
            raise ValueError("For DoG/MexicanHat: require 0 < sigma < sigma_i.")

        sq = pairwise_sqdist(neuron_coords, src_coords, periodic=periodic, box_size=box_size)

        Ke = amp_e * np.exp(-0.5 * sq / (sigma_e**2))    # excitatory
        Ki = amp_i * np.exp(-0.5 * sq / (sigma_i**2))    # inhibitory 

        const_e = (np.sqrt(2*np.pi)*sigma_e)**D
        const_i = (np.sqrt(2*np.pi)*sigma_i)**D
        K = (Ke/const_e) - (Ki/const_i)
        return K

    # elif Wtype == 'SpaceGaussian':
    #     # 你原先的空间×时间核（如不需要可忽略此分支）
    #     sq_space = pairwise_sqdist(neuron_coords[:, :2], src_coords[:, :2],
    #                                periodic=periodic[:2] if isinstance(periodic, (list, tuple, np.ndarray)) else False,
    #                                box_size=box_size[:2] if isinstance(box_size, (list, tuple, np.ndarray)) else None)
    #     if neuron_coords.shape[1] < 3 or src_coords.shape[1] < 3:
    #         raise ValueError("SpaceGaussian expects at least 3D coords [x,y,t].")
    #     if Lambda is None or Lambda <= 0:
    #         raise ValueError("SpaceGaussian requires positive Lambda.")

    #     t_i = neuron_coords[:, 2:3]; t_j = src_coords[:, 2:3]
    #     dist_time = np.abs(t_i - t_j.T)  # (N,M)

    #     Gs = amp_e * np.exp(-0.5 * sq_space / (sigma_e**2))
    #     K  = Gs * np.exp(-Lambda * dist_time)
    #     const = (np.sqrt(2*np.pi)*sigma_e)**2 / Lambda
    #     return K / const

    else:
        raise ValueError(f"Unknown Wtype: {Wtype}")

# def gaussian_kernel_from_coords(neuron_coords, src_coords, Wtype, sigma, 
#                                 Lambda=None, amp=1.0, gamma=0.0, 
#                                 periodic=None, box_size=None):

#     if Wtype == 'Gaussian':
#         sq = pairwise_sqdist(neuron_coords, src_coords, periodic=periodic, box_size=box_size)
#         K = amp * np.exp(-0.5 * sq / (sigma**2)) 
#         const = (np.sqrt(2*np.pi)*sigma)**(neuron_coords.shape[1])

#     elif Wtype == 'SpaceGaussian':
#         sq_dist_space = pairwise_sqdist(neuron_coords[:,:2], src_coords[:,:2], periodic=periodic[:2], box_size=box_size[:2])
#         dist_time = pairwise_dist(neuron_coords[:,2:], src_coords[:,2:])
#         K = amp * np.exp(-0.5 * sq_dist_space / (sigma**2)) * np.exp(- Lambda * dist_time) 
#         const = (np.sqrt(2*np.pi)*sigma)**(neuron_coords.shape[1]-1) / Lambda 
    
#     K = K / const - gamma
#     return K

@dataclass
class ExperienceCANN:
    neuron_coords: np.ndarray
    channel_coords: np.ndarray
    
    # Wrc
    W_rc_type: str 
    sigmaE_rc: float
    W0E_rc: float 
    sigmaI_rc: float = None 
    W0I_rc: float = None 
    
    # They are the same as the trained RNN
    # ------------------------
    # Win
    W_in_type: str = None
    b_in_type: str = None
    
    # Wout
    W_out_type: str = None
    b_out_type: str = None
    # ------------------------
    
    alpha: float = 0.01
    # bias: np.ndarray = None
    nonlinearity: str = "relu"
    periodic: Union[bool, Sequence[bool]] = False   
    box_size: Union[float, Sequence[float], None] = None
    divisive_norm: bool = False
    
    

    def __post_init__(self):
        N = self.neuron_coords.shape[0]   # hidden size
        C = self.channel_coords.shape[0]  # input size

        self.W_rc = gaussian_kernel_from_coords(self.neuron_coords, self.neuron_coords, self.W_rc_type, self.periodic, self.box_size,
                                                self.sigmaE_rc, self.W0E_rc, self.sigmaI_rc, self.W0I_rc,
                                                )

        if self.W_in_type is None:
            self.W_in = np.random.uniform(-np.sqrt(1/C), np.sqrt(1/C), size=(N, C)).astype(np.float32)
        else:
            raise NotImplementedError("Only None W_in_type is implemented now.")
        if self.b_in_type is None:
            self.bias = np.random.uniform(-np.sqrt(1/C), np.sqrt(1/C), size=(N,)).astype(np.float32)
        else:
            raise NotImplementedError("Only None b_in_type is implemented now.")
            
        if self.W_out_type is None:
            self.W_out = np.random.uniform(-np.sqrt(1/C), np.sqrt(1/C), size=(N, C)).astype(np.float32)
        else:
            raise NotImplementedError("Only None W_out_type is implemented now.")
        if self.b_out_type is None:
            self.b_out = np.random.uniform(-np.sqrt(1/C), np.sqrt(1/C), size=(C,)).astype(np.float32)
        else:
            raise NotImplementedError("Only None b_out_type is implemented now.")


    def phi(self, V):
        if self.nonlinearity == "relu":
            return np.maximum(0.0, V)
        else:
            raise ValueError("Unknown nonlinearity")


    def step(self, V, e_t):
        r = self.phi(V)           # (B,N)
        if self.divisive_norm:
            r = r / np.maximum(1e-6, r.sum(axis=1, keepdims=True))
        
        # W^in @ e_t + b^in
        inn = (e_t @ self.W_in.T + self.bias.T) # (B,N)
        # # phi(inn)
        # fr = self.phi(inn)        # (B,N)
        
        # W^rc @ r
        rec = (self.W_rc @ r.T).T # (B,N)
        target = rec + inn # + self.bias 
        
        V_new = (1.0 - self.alpha) * V + self.alpha * target
        r_new = self.phi(V_new)   # (B,N)
        if self.divisive_norm:
            r_new = r_new / np.maximum(1e-6, r_new.sum(axis=1, keepdims=True))
        return V_new, r_new

    def run(self, e):
        B, T, C = e.shape
        N = self.neuron_coords.shape[0]

        V = np.zeros((B, N), dtype=float)        # Initialise V to zero
        r_all = np.zeros((B, T, N), dtype=float) # Record outputs for all time steps
        for t in range(T):
            V, r = self.step(V, e[:, t, :])
            r_all[:, t, :] = r
        return r_all
