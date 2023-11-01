import numpy as np
import pulp
import dataclasses

KB = 1 << 10
MB = 1 << 20
GB = 1 << 30
T = 1e12

@dataclasses.dataclass
class CostModelConfig:
    s: int = 512
    n: int = 32

    l: int = 96
    h1: int = 12288
    h2: int = 12288 * 4
    nh: int = 96

    gmem: int = 16 * GB
    nmem: int = 1500 * GB
    cache: int = 1 * MB

    # hardware constants
    dtog_bdw: float = 0.473 * GB
    gtod_bdw_cache_p: float = 0.746 * GB
    gtod_bdw_hidden_p: float = 2.015 * GB
    gtod_bdw_g: float = 2.015 * GB

    mm_flops_p: float = 21.24 * T
    mm_flops_g: float = 4.3 * T
    bmm_flops_p: float = 9.97 * T
    bmm_flops_g: float = 0.079 * T

def solve_lp(config:CostModelConfig, 
                bls_tknum, 
                gbs):
    s = config.s
    n = config.n
    l = config.l
    h1 = config.h1
    h2 = config.h2
    nh = config.nh

    gmem = config.gmem
    nmem = config.nmem
    cache = config.cache

    mm_flops_p = config.mm_flops_p
    mm_flops_g = config.mm_flops_g
    bmm_flops_p = config.bmm_flops_p
    bmm_flops_g = config.bmm_flops_g

    dtog_bdw = config.dtog_bdw
    gtod_bdw_cache_p = config.gtod_bdw_cache_p
    gtod_bdw_hidden_p = config.gtod_bdw_hidden_p
    gtod_bdw_g = config.gtod_bdw_g

    prob = pulp.LpProblem('storage', sense=pulp.LpMinimize)

    ## Create variables for cost
    T = pulp.LpVariable("T", lowBound=0)
    Tpre = pulp.LpVariable("Tpre_i", lowBound=0)
    Tgen = pulp.LpVariable("Tgen_i", lowBound=0)
    gtodp = pulp.LpVariable("gtod_i^p", lowBound=0)
    dtogp = pulp.LpVariable("dtog_i^p", lowBound=0)
    compp = pulp.LpVariable("comp_i^p", lowBound=0)
    gtodg = pulp.LpVariable("gtod_i^g", lowBound=0)
    dtogg = pulp.LpVariable("dtog_i^g", lowBound=0)
    compg = pulp.LpVariable("comp_i^g", lowBound=0)

    wg = pulp.LpVariable("wg", lowBound=0)
    wn = pulp.LpVariable("wn", lowBound=0)
    cg = pulp.LpVariable("cg", lowBound=0)
    cn = pulp.LpVariable("cn", lowBound=0)
    hg = pulp.LpVariable("hg", lowBound=0)
    hn = pulp.LpVariable("hn", lowBound=0)

    prob += T * (1 / bls_tknum)

    wi = 8 * h1 ** 2 + 4 * h1 * h2
    
    prob += wg + wn == 1
    prob += cg + cn == 1
    prob += hg + hn == 1

    prob += T == Tpre * l + Tgen * (n - 1) * l

    prob += Tpre >= dtogp
    prob += Tpre >= gtodg
    prob += Tpre >= compp

    prob += dtogp == (1 / dtog_bdw) * (wn * wi + 2 * hn * s * h1 * bls_tknum)
    # prob += gtodp == (1 / gtod_bdw_cache_p) * (4 * cn * bls_tknum * (s + 1) * h1) + 1 / (gtod_bdw_hidden_p) * (2 * hn * s * h1 * bls_tknum)
    # No need to store the cache
    prob += gtodp == 1 / (gtod_bdw_hidden_p) * (2 * hn * s * h1 * bls_tknum)

    prob += compp == (1 / mm_flops_p) * bls_tknum * (8 * s * h1 ** 2  + 4 * s * h1 * h2) \
                     + (1 / bmm_flops_p) * 4 * bls_tknum * s ** 2 * h1

    prob += Tgen >= dtogg
    prob += Tgen >= gtodp
    prob += Tgen >= compg

    # prob += gtodg == (1 / dtog_bdw) * (4 * cn * bls_tknum * h1 + 2 * hn * h1 * bls_tknum)
    # Cancel the cache storeback
    prob += gtodg == (1 / dtog_bdw) * (2 * hn * h1 * bls_tknum)
    prob += dtogg == (1 / gtod_bdw_g) * (4 * cn * bls_tknum * (s + n / 2) * h1 \
                                        + wn * wi + \
                                        2 * hn * h1 * bls_tknum)
    prob += compg == (1 / mm_flops_g) * bls_tknum * (wi) \
                     + (1 / bmm_flops_g) * 4 * bls_tknum * (s + n / 2) * h1 * cg 


    # TODO: Figure out the backward computation time


    gpu_home_p = pulp.LpVariable("gpu_home^p", lowBound=0)
    gpu_w_p = pulp.LpVariable("gpu_w^p", lowBound=0)
    gpu_home_g = pulp.LpVariable("gpu_home^g", lowBound=0)
    gpu_w_g = pulp.LpVariable("gpu_w^g", lowBound=0)

    interp = pulp.LpVariable("inter_gpu_working_p", lowBound=0)
    qkvp = pulp.LpVariable("qkvp", lowBound=0)
    att1p = pulp.LpVariable("att1p", lowBound=0)
    att2p = pulp.LpVariable("att2p", lowBound=0)
    outputp = pulp.LpVariable("outputp", lowBound=0)
    mlp1p = pulp.LpVariable("mlp1p", lowBound=0)
    mlp2p = pulp.LpVariable("mlp2p", lowBound=0)

    interg = pulp.LpVariable("inter_gpu_working_g", lowBound=0)
    qkvg = pulp.LpVariable("qkvg", lowBound=0)
    att1g = pulp.LpVariable("att1g", lowBound=0)
    att2g = pulp.LpVariable("att2g", lowBound=0)
    outputg = pulp.LpVariable("outputg", lowBound=0)
    mlp1g = pulp.LpVariable("mlp1g", lowBound=0)
    mlp2g = pulp.LpVariable("mlp2g", lowBound=0)

    nvme_peak = pulp.LpVariable("nvme_peak", lowBound=0)

    # Flash-Attention related variable and constraint
    
    # Forward
    cache_peak = pulp.LpVariable("cache_peak", lowBound=0)
    inter_mem = pulp.LpVariable("B_r * B_c", lowBound=0)
    # B_c = pulp.LpVariable("B_c", lowBound=0)

    # Memory requirements for matrices Q, K, V
    Mem_QKV = (s + n) * h1
    # Memory requirements for intermediate computations
    # Mem_intermediate = B_r * B_c
    # Memory requirements for other matrices
    d = h1 / nh
    Mem_others = n * (d + 2)
    # Total memory requirement
    # cache_peak = l * (Mem_QKV + Mem_intermediate + Mem_others)
    cache_peak = l * (Mem_QKV + inter_mem + Mem_others)

    # Backward
    # Add the memory constraint to the LP problem
    prob += cache_peak <= cache, "Memory Constraint"
    # Memory requirements for matrices Q, K, V, O
    Mem_QKVO = 2 * (s + n) * h1
    # Memory requirements for intermediate computations for backward pass
    Mem_intermediate_back = inter_mem
    # Memory requirements for other matrices for backward pass
    d = h1 / nh
    Mem_others_back = 3 * n * d + 2 * n
    # Total memory requirement for backward pass
    Mem_total_back = l * (Mem_QKVO + Mem_intermediate_back + Mem_others_back)
    # Add the memory constraint for backward pass to the LP problem
    prob += Mem_total_back <= cache, "Memory Constraint for Backward Pass"



    ## GPU peak memory constaints
    prob += gpu_home_p == wi * l * wg + 2 * s * h1 * bls_tknum * hg + 4 * (s + n) * h1 * bls_tknum * l * cg
    prob += interp == 8 * gbs * s * h1 \
                    + gbs * (2 * s * h1 + 2 * nh * s ** 2) \
                    + gbs * (2 * s * h1) \
                    + 4 * gbs * s * h1 \
                    + 2 * gbs * s * h2 \
                    + 2 * gbs * s * h1
    prob += gpu_w_p == 2 * wi * (1 - wg) + 2 * s * h1 * gbs * (1 - hg) \
                     + interp
    prob += gpu_home_p + gpu_w_p <= gmem



    prob += gpu_home_g == wi * l * wg + 2 * h1 * bls_tknum * hg + 4 * (s + n) * h1 * bls_tknum * l * cg
    prob += interg == 8 * gbs * h1 \
                    + gbs * (2 * h1 + 2 * (s + n) * h1 + 2 * nh * (s + n)) * cg \
                    + gbs * (2 * (s + n) * h1 + 2 * h1) * cg \
                    + 4 * gbs * h1 \
                    + 2 * gbs * h2 \
                    + 2 * gbs * h1
    prob += gpu_w_g == 2 * wi * (1 - wg) + 2 * h1 * gbs * (1 - hg) \
                     + 2 * 2 * gbs * (s + n) * h1 * cg \
                     + interg
    prob += gpu_home_g + gpu_w_g <= gmem
    
    prob += nvme_peak == wi * l * wn + 2 * s * h1 * bls_tknum * hn + 4 * (s + n) * h1 * bls_tknum  * l * cn
    prob += nvme_peak <= nmem

    embeddings_size = l * (s + n) * h1
    
    # Multi-Head Self Attention
    mha_size = l * (3 * (s + n) * h1 + (s + n) * (s + n) * nh)
    
    # Feed-forward Neural Networks (FFNs)
    ffn_size = l * ((s + n) * h2 + (s + n) * h1)
    
    # Total size in FP16
    total_size_fp16 = 2 * (embeddings_size + mha_size + ffn_size)

    prob += total_size_fp16 <= nmem
    

    solver = pulp.PULP_CBC_CMD(msg=False)
    status = prob.solve(solver)
    if status == -1:
        return status, None, (0, -1, -1), None
    
    gpu_peak_p = pulp.value(gpu_home_p) + pulp.value(gpu_w_p)
    gpu_peak_g = pulp.value(gpu_home_g) + pulp.value(gpu_w_g)

    throughput = bls_tknum * n / pulp.value(T)

    tpre = pulp.value(Tpre)
    tpre_tot = tpre * l

    tgen = pulp.value(Tgen)
    tgen_tot = tgen * (n-1) * l

    print(f"status: {status}")
    print(f"weights size: {wi * l / GB:.4f} GB")
    print(f"dtogp = {pulp.value(dtogp):.4f} s  "
          f"gtodp = {pulp.value(gtodp):.4f} s  "
          f"compp = {pulp.value(compp):.4f} s")
    print(f"Tpre = {pulp.value(Tpre):.3f} s")
    print(f"Tpre * l: {tpre:.4f} * {l} = {tpre_tot:.4f}")
    print(f"dtogg = {pulp.value(dtogg):.4f} s  "
          f"gtodg = {pulp.value(gtodg):.4f} s  "
          f"compg = {pulp.value(compg):.4f} s")
    print(f"Tgen = {pulp.value(Tgen):.3f} s")
    print(f"Tgen * (n-1) * l: "
            f"{tgen:.4f} * {n-1} * {l} = {tgen_tot:.4f}")

    # print(f"gpu peak mem (prefill): {gpu_peak_p / GB:.3f} GB / {gmem / alpha_g / GB:.3f} GB")
    # print(f"gpu peak mem (gen):     {gpu_peak_g / GB:.3f} GB / {gmem / alpha_g / GB:.3f} GB")

    # print(f"cpu peak mem (prefill): {cpu_peak_p / GB:.3f} GB / {cmem / alpha_c / GB:.3f} GB")
    # print(f"cpu peak mem (gen):     {cpu_peak_g / GB:.3f} GB / {cmem / alpha_c / GB:.3f} GB")
    print(f"gpu peak mem (prefill): {gpu_peak_p / GB:.3f} GB / {gmem / GB:.3f} GB")
    print(f"gpu peak mem (gen):     {gpu_peak_g / GB:.3f} GB / {gmem / GB:.3f} GB")

    print(f"nvme peak mem:          {pulp.value(nvme_peak) / GB:.3f} GB / {nmem / GB:.3f} GB")

    print(f"wg = {pulp.value(wg):.2f}  "
          f"wn = {pulp.value(wn):.2f}")
    print(f"cg = {pulp.value(cg):.2f}  "
          f"cn = {pulp.value(cn):.2f}")
    print(f"hg = {pulp.value(hg):.2f}  "
          f"hn = {pulp.value(hn):.2f}")
    print(f"T = {pulp.value(T)} s  "
          f"generated = {bls_tknum * n} tokens")
    print(f"throughput = {throughput:.2f} token/s")

if __name__ == "__main__":
    config = CostModelConfig()
    bls_tknum = 8
    gbs = 8
    solve_lp(config, bls_tknum, gbs)