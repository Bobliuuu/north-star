"""Launch parameters of the fast Triton decode kernels (harness/fast_kernels*.py). Track 'triton' of the self-improve loop.

Only the integer assignments between the NS-EDITABLE markers may change (structured edits, src/selfimprove/structured_triton.py
validates every name and value). The kernel BODIES (harness/fast_kernels.py, fast_kernels_blk.py, fast_kernels_exact.py) are protected and
unchanged; this file only chooses tile sizes, num_warps, num_stages, split-K counts and chunk sizes passed at launch.
The defaults reproduce the hand-set launches of FastNorth exactly (num_stages 3 is Triton's default).
Names: GU/DN = T=1 expert gate_up / down GEMV, ATT/CMB/ROPE = T=1 split-K attention, RN/TK = fused residual+norm / router top-k,
XA/XCMB/ROPEB = exact-mode (batch-invariant) attention, LIN = exact-mode linears, MB = exact/verify expert block kernels, AM = argmax.
"""
# NS-EDITABLE-BEGIN
GU_BN = 8
GU_BK = 256
GU_NW = 4
GU_NS = 3
DN_BN = 8
DN_BK = 256
DN_NW = 4
DN_NS = 3
ATT_NSPLIT = 32
ATT_BN = 64
ATT_NW = 4
ATT_NS = 3
CMB_NW = 4
ROPE_NW = 1
RN_NW = 8
TK_NW = 1
XA_CHUNK = 256
XA_BN = 64
XA_NW = 8
XA_NS = 3
XCMB_NW = 4
ROPEB_NW = 1
LIN_BN_S = 16
LIN_BN_L = 64
LIN_BK_A = 256
LIN_BK_B = 128
LIN_NW_S = 4
LIN_NS_S = 3
LIN_NW_L = 4
LIN_NS_L = 3
MB_BN1 = 16
MB_BK1 = 128
MB_NW1 = 4
MB_NS1 = 3
MB_BN2 = 16
MB_BK2 = 128
MB_NW2 = 4
MB_NS2 = 3
AM_CH = 4096
AM_NW = 4
# NS-EDITABLE-END
