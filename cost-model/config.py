import dataclasses

KB = 1 << 10
MB = 1 << 20
GB = 1 << 30
T = 1e12

alpha_gpu  = 0.8
alpha_cpu  = 0.8
alpha_nvme = 0.8

@dataclasses.dataclass
class Hardware_config:
    nvme_mem: int = 128 * GB
    gpu_cpu_shared_mem: int = 16 * GB

    ctog_bdw: float
    gtoc_bdw_cache: float 
    gtoc_bdw_hidden: float    
    dtoc_bdw: float = 0.473 * GB
    ctod_bdw_cache_p: float = 0.746 * GB
    ctod_bdw_hidden_p: float = 2.015 * GB
    ctod_bdw_g: float = 2.015 * GB

    mm_flops_p: float = 21.24 * T
    mm_flops_g: float = 4.3 * T
    bmm_flops_p: float = 9.97 * T
    bmm_flops_g: float = 0.079 * T
    cpu_flops: float = 0.0123 * T

    c1: float = 0.0168
    c2: float = 0.0328
    c3: float = 0.0621