import torch
from transformers.models.cohere2_moe.configuration_cohere2_moe import Cohere2MoeConfig
from transformers.models.cohere2_moe.modeling_cohere2_moe import Cohere2MoeForCausalLM


def tiny_cfg(**kw):
    base = dict(vocab_size=97, hidden_size=64, intermediate_size=32, prefix_dense_intermediate_size=96,
                num_hidden_layers=7, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                num_experts=8, num_experts_per_tok=4, first_k_dense_replace=1, sliding_window=8,
                expert_selection_fn="sigmoid", norm_topk_prob=False, rms_norm_eps=1e-6,
                rope_theta=50000.0, logit_scale=1.0, max_position_embeddings=512)
    base.update(kw)
    return Cohere2MoeConfig(**base)


def tiny_model(seed=0, **kw):
    torch.manual_seed(seed)
    m = Cohere2MoeForCausalLM(tiny_cfg(**kw)).eval().float()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if p.ndim >= 2:
                p.normal_(0, 0.25 if "embed" not in n else 1.0)
    return m


def param_hash(m):
    import hashlib
    h = hashlib.sha256()
    for n, t in list(m.named_parameters()) + list(m.named_buffers()):
        h.update(n.encode())
        h.update(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def ids(T=40, seed=1, B=1):
    return torch.randint(0, 97, (B, T), generator=torch.Generator().manual_seed(seed))
