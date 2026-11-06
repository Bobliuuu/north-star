import torch
from transformers.models.cohere2_moe.configuration_cohere2_moe import Cohere2MoeConfig
from transformers.models.cohere2_moe.modeling_cohere2_moe import Cohere2MoeForCausalLM


def tiny_model(seed=0):
    torch.manual_seed(seed)
    cfg = Cohere2MoeConfig(vocab_size=97, hidden_size=64, intermediate_size=32, prefix_dense_intermediate_size=96,
                           num_hidden_layers=5, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                           num_experts=8, num_experts_per_tok=4, first_k_dense_replace=1, sliding_window=8,
                           expert_selection_fn="sigmoid", norm_topk_prob=False, rms_norm_eps=1e-6,
                           rope_theta=50000.0, logit_scale=1.0, max_position_embeddings=512)
    m = Cohere2MoeForCausalLM(cfg).eval().float()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if p.ndim >= 2:
                p.normal_(0, 0.25 if "embed" not in n else 1.0)
    return m


def seqs(n, L, seed):
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(0, 97, (L,), generator=g).tolist() for _ in range(n)]
