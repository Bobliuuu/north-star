import torch
from transformers.models.cohere2_moe.configuration_cohere2_moe import Cohere2MoeConfig
from transformers.models.cohere2_moe.modeling_cohere2_moe import Cohere2MoeForCausalLM

from northstar.refmodel import RefNorth


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
    cfg = tiny_cfg(**kw)
    m = Cohere2MoeForCausalLM(cfg).eval().float()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if p.ndim >= 2:
                p.normal_(0, 0.25 if "embed" not in n else 1.0)
    return m


def tiny_ref(seed=0, **kw):
    m = tiny_model(seed, **kw)
    return m, RefNorth(m)
