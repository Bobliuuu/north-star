import torch


def load_model(model_dir, experts_impl="grouped_mm", attn_impl="sdpa"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    kw = dict(dtype=torch.bfloat16, device_map="cuda", attn_implementation=attn_impl)
    if experts_impl:
        kw["experts_implementation"] = experts_impl
    m = AutoModelForCausalLM.from_pretrained(model_dir, **kw).eval()
    tok = AutoTokenizer.from_pretrained(model_dir)
    return m, tok
