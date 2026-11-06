# North Mini Code 1.0 — architecture facts

Sources: HF `config.json`, `model.safetensors.index.json`, HF model card, upstream
`transformers` `models/cohere2_moe/modeling_cohere2_moe.py` + `configuration_cohere2_moe.py` (main, fetched 2026-11-05),
and `cohere-ai/cohere-megakernel`. Nothing here is inferred from a generic Transformer diagram; items marked
(derived) are arithmetic from config, not stated by Cohere.

## Pins
- Model: `CohereLabs/North-Mini-Code-1.0`, revision `d11e61a842617a22dc328552fa5bb86231ee4f37` (lastModified 2026-06-15).
- Engine: `cohere-ai/cohere-megakernel`, commit `67d0b9ca22ea3652796b715d1d1863459e0e2c3c` (2026-09-08).
- Config says `transformers_version` 5.8.0; model_type `cohere2_moe`, class `Cohere2MoeForCausalLM`
  (needs transformers from source/>=5.x; local venv had torch 2.10.0, transformers not installed in system python).
- License: Apache-2.0 (weights and engine). Not gated.

## Size and dtype
- 30,484,303,872 parameters (HF safetensors metadata), all BF16. Checkpoint 49 shards, 61.0 GB (61,027,335,986 B used storage).
- Cohere: "30B total, 3B active"; megakernel blog: 3.3B active, ~6.6 GB BF16 weights streamed per decode step.
- Tied embeddings: `lm_head.weight` is tied to `model.embed_tokens.weight` (no lm_head in the checkpoint index; vocab 262,144 x 2048 = 0.54B).
- Breakdown (derived, sums to 30.49B): experts 128 x 3 x 2048 x 768 = 604M per MoE layer x 48 = 29.0B; attention 18.9M x 49 = 0.93B;
  dense prefix MLP 18.9M; embeddings 0.54B. Active: 8 experts x 4.72M x 48 = 1.81B + attn 0.93B + embed/LM head 0.54B + dense 0.02B = 3.3B (matches blog).
- Expert weights per MoE layer: 1.21 GB BF16 (derived). One expert = 9.4 MB.

## Layers (config)
- `num_hidden_layers` 49, `hidden_size` 2048, `head_dim` 128, vocab 262144, `hidden_act` silu, RMSNorm eps 1e-6, `logit_scale` 1.0.
- Layer 0 is DENSE (`first_k_dense_replace` 1 -> `mlp_layer_types[0]="dense"`): SwiGLU MLP with `prefix_dense_intermediate_size` 3072.
  Layers 1-48 are sparse MoE.
- Attention pattern: `layer_types` = full_attention at layers 0,4,8,...,48 (13 global layers), sliding_attention elsewhere (36 layers). 3:1 sliding:global, global first and last.
- Sliding window 4096. Max positions 500000 in config; model card says 256K input / 64K output (vLLM example uses `--max-model-len 320000`). `rope_scaling` null, `rope_theta` 50000.
- Positional convention (from modeling code): RoPE (interleaved-pair `rotate_half`, `repeat_interleave` cos/sin) is applied ONLY on sliding layers, plus
  `force_rope` = layer is dense AND `prefix_dense_sliding_window_pattern==1` -> layer 0 (a global layer) also gets RoPE.
  Layers 4,8,...,48 (12 global layers) are NoPE (no positional embedding). `use_qk_norm` false, `attention_bias` false.
- Attention: 32 query heads, 4 KV heads (GQA group 8), scaling head_dim^-0.5, q_proj 2048->4096, k/v 2048->512, o_proj 4096->2048.

## Parallel block (use_parallel_block true)
Per layer, one pre-norm only (`input_layernorm`; there is no post-attention norm):
```
h = input_layernorm(x)
x_out = x + self_attn(h) + mlp(h)      # attn and MLP/MoE both read the SAME normed h; independent branches
```
Attention and the MoE/dense branch have no data dependency inside a layer; they join only at the residual add. (Matches the megakernel blog's reason MoE tiles can fill SMs idle in the attention tail.)

## MoE router and experts (modeling code)
- 128 routed experts, top-8 per token (`num_experts_per_tok`), `num_shared_experts` 0 (so `shared_expert_combination_strategy: average` is unused), expert FFN dim 768, SwiGLU (gate_proj, up_proj, down_proj; HF fuses gate+up as `gate_up_proj` [E, 1536, 2048] and `down_proj` [E, 2048, 768]).
- Router: `logits = Linear(h)` (128 x 2048, no bias); `topk` is taken on RAW logits; then `expert_selection_fn: sigmoid` is applied to the selected top-8 logits only; `norm_topk_prob: false`, so the 8 weights are NOT renormalized (they do not sum to 1).
  Output = sum over selected experts of sigmoid(logit_e) * Expert_e(h). Consequence for top-r draft: dropping experts removes their sigmoid-weighted contribution without renormalization; any renormalization is a deliberate approximation to document.
- Router logits are computed in the input dtype (bf16); `router_scores` cast back to input dtype.

## KV state (derived)
- Per token per layer K+V = 2 x 4 x 128 x 2 B = 2 KiB. Global layers (13): 26.6 KB/token -> 128K ctx = 3.5 GB, 256K = 7.0 GB per sequence.
- Sliding layers (36): bounded to 4096 tokens each = 302 MB per sequence regardless of context.
- Megakernel blog quotes ~0.5 GB KV at 8K context (consistent order of magnitude: 8K global 0.2 GB + 0.3 GB window).

## Engine facts that constrain the plan
- Megakernel is **H100 (sm_90a) only**, CUDA 13+, BF16 only, batch 1-8, decode only (prefill is ordinary PyTorch), North-specific task schedule, paged KV, no speculative decoding in README/blog/source (grep for specul/draft: none).
- Its GEMM primitive is `src/decode/gemm-n8-wgmma.cuh`: an N=8 token-column wgmma tile. Verify block width x batch must fit this N=8 granularity, or the tile must be generalised. Directly relevant to fused draft+verify tiles.
- Cohere baseline numbers (H100, vLLM 0.24 baseline): megakernel 292 tok/s at BS1 (62% of 470 tok/s speed-of-light), 1.25-1.41x over vLLM on real workloads.

## Fit on candidate GPUs (derived; weights 61.0 GB)
- A100-80GB: weights fit with ~19 GB headroom (about 3.5 GB per 128K sequence of global KV + 0.3 GB window + activations). Feasible for HF/PyTorch baseline at batch 1-4 and up to ~128K. NOT feasible for 256K at batch>1 comfortably. The megakernel does NOT run on A100 (sm_80, no wgmma).
- H100-80GB: same capacity; only GPU supported by Cohere's engine.
- 40 GB / 48 GB / 24 GB GPUs (A100-40, A10G, L4, L40S): do not fit BF16 weights. Existing A10G/L4 boxes in the user's fleet cannot hold the model without quantization (which would change the reference).
- Model card serves with `-tp 2` in vLLM; that is a throughput recommendation for 320K context, not a hard need for 80 GB.
