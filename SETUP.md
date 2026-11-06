# SETUP (pinned; every version below is read from the repo or from run metadata)

## Model and engine
| item | pin | where recorded |
|---|---|---|
| model | `CohereLabs/North-Mini-Code-1.0`, revision `d11e61a842617a22dc328552fa5bb86231ee4f37` (Apache-2.0, not gated, 56.8 GiB) | `modal/nsapp.py`, `results/PHASE01.md`, `results/baseline.meta.json` |
| engine baseline | `cohere-ai/cohere-megakernel` @ `67d0b9ca22ea3652796b715d1d1863459e0e2c3c`, built unmodified | `modal/mk.py`, RESULTS.md H1 |
| KVM ABI header (copied, executor NOT used) | `~/kvm-wt/integ` branch `feat/integration` @ `fc0ced7ebce3d107a7a0141a9124e32f519024b3` | `src/kvm/PROVENANCE.md` |

## Local CPU environment (what `make test`, `make check`, `make plots` ran with)
Python 3.12.10. Exact packages (`requirements.txt`; `make setup` installs them into `.venv`):
`torch==2.10.0`, `transformers==5.22.1`, `accelerate==1.15.0`, `numpy==2.2.5`, `ortools==9.15.6755`, `protobuf==6.33.6`, `pytest==8.3.5`, `matplotlib==3.10.8`.
(`src/opt/requirements.txt` is the narrower pin set for the optimizer alone.) No GPU, no network, no model weights are needed for tests or plots.
Test suites run in separate pytest processes because `tests/` and `src/approx/tests/` both define a module named `fixtures`.
Last run on this branch: tests 39, src/approx 43, src/kvm 5, src/opt 19, src/selfimprove 163 (+2 skipped), src/tage 8.

## GPU environments (Modal; pinned in the image definitions)
| use | image | GPU | pins |
|---|---|---|---|
| Phase 0/1, H100 graphed runs, approx sweep (`modal/nsapp.py`) | debian_slim, python 3.12 | A100-80GB (H100 for `modal/h100.py`) | torch 2.10.0, transformers 5.22.1, accelerate 1.15.0, numpy 2.2.5, pytest 8.3.5; huggingface_hub and safetensors UNPINNED in the image (run metadata: torch 2.10.0+cu128, CUDA 12.8, python 3.12.16) |
| fused MoE microbench (`modal/run_modal.py`) | `nvidia/cuda:12.4.1-devel-ubuntu22.04`, python 3.11 | A100-80GB `-arch=sm_80`; H100 `-arch=sm_90` | torch 2.5.1 (CPU wheel, reference only), numpy<2 (UNPINNED upper bound), nvcc -O3 -std=c++17 |
| cohere-megakernel (`modal/mk.py`) | `nvidia/cuda:13.0.1-devel-ubuntu24.04`, python 3.12 | H100 80GB, sm_90a | torch from the cu130 index UNPINNED (RESULTS.md reports 2.14+cu130 as built), transformers/flash-attn-3/etc. UNPINNED, `cohere-melody==0.9.0`, engine commit pinned above |
| self-improvement evaluator (`modal/selfimprove_eval.py`) | see file | GPU evaluator | CPU torch 2.10.0, numpy 2.2.5, ortools 9.15.6755, pytest 8.3.5 |

Known pinning gaps (do not claim bit-reproducibility of the GPU images): the megakernel image and the `nsapp` image leave some packages unpinned; the megakernel torch version is only known from run output.

## Weights
`modal run modal/download.py` (CPU only) downloads the pinned revision into Modal Volume `northstar-weights`. Results volume: `northstar-results`.

## One-command targets
`make test` | `make check` | `make plots` | `make bench` (prints, never runs, the `modal run --detach` commands). Spend log: `results/SPEND.md`; launch log: `RUNS.md`.
