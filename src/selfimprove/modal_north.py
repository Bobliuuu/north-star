"""Modal service hosting North as the (only) proposer. NOT launched by this task; written + syntax-checked only.

Deploy:   modal deploy src/selfimprove/modal_north.py          (starts nothing; containers spin up on first call)
Smoke:    modal run src/selfimprove/modal_north.py --prompt "say hi"      (GPU, A100-80GB, ~$2.5/h; confirm cost first)
Client:   selfimprove.proposer.NorthProposer()  -> modal.Cls.from_name("northstar-selfimprove", "NorthProposerService")

Reuses the Volume `northstar-weights` and MODEL_DIR written by modal/download.py (pinned revision d11e61a8). Weights are NOT
downloaded here; if the volume is empty the service fails loudly instead of fetching an unpinned model.

Backend: NS_BACKEND=hf (default; the stack proven in modal/run.py: transformers 5.22.1, cohere2_moe) or vllm (opt-in; needs
vLLM with cohere2_moe support, UNVERIFIED here: deploy with NS_WITH_VLLM=1 and a pinned vllm in NS_VLLM_SPEC).
"""
import os, sys, time, uuid, pathlib
import modal

_here = pathlib.Path(__file__).resolve()
_ROOT = _here.parents[2] if len(_here.parents) > 2 else _here.parent   # container path is /root/modal_north.py
sys.path.insert(0, str(_ROOT / "modal"))
try:                                     # reuse the other agent's constants/volume rather than duplicating them
    from common import weights_vol, WEIGHTS, MODEL_DIR, MODEL_ID, MODEL_REV, image as _base_image
except Exception:                        # pragma: no cover  (standalone fallback, same names)
    WEIGHTS, MODEL_ID = "/weights", "CohereLabs/North-Mini-Code-1.0"
    MODEL_REV = "d11e61a842617a22dc328552fa5bb86231ee4f37"
    MODEL_DIR = f"{WEIGHTS}/north-mini-code-d11e61a8"
    weights_vol = modal.Volume.from_name("northstar-weights", create_if_missing=False)
    _base_image = modal.Image.debian_slim(python_version="3.12").pip_install(
        "torch==2.10.0", "transformers==5.22.1", "accelerate==1.15.0", "numpy==2.2.5", "safetensors")

USD_PER_GPU_HOUR = 2.50                  # A100-80GB list price used elsewhere in the repo
image = _base_image
if os.environ.get("NS_WITH_VLLM") == "1":
    image = image.pip_install(os.environ.get("NS_VLLM_SPEC", "vllm"))

app = modal.App("northstar-selfimprove", image=image)


@app.cls(gpu="A100-80GB", volumes={WEIGHTS: weights_vol}, memory=65536, timeout=1800, scaledown_window=60,
         max_containers=1)
class NorthProposerService:
    @modal.enter()
    def load(self):
        if not os.path.exists(os.path.join(MODEL_DIR, "config.json")):
            raise RuntimeError(f"weights missing at {MODEL_DIR}: run `modal run modal/download.py` first")
        self.backend = os.environ.get("NS_BACKEND", "hf")
        t = time.time()
        if self.backend == "vllm":
            from vllm import LLM
            self.llm = LLM(MODEL_DIR, dtype="bfloat16", max_model_len=32768, gpu_memory_utilization=0.92)
            self.tok = self.llm.get_tokenizer()
        else:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.tok = AutoTokenizer.from_pretrained(MODEL_DIR)
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_DIR, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa",
                experts_implementation="grouped_mm").eval()
        self.load_s = time.time() - t

    @modal.method()
    def propose(self, payload: dict) -> dict:
        """payload: {system, prompt, max_new_tokens, temperature}. Returns text + attestation (model id/revision)."""
        t0 = time.time()
        msgs = [{"role": "system", "content": payload["system"]}, {"role": "user", "content": payload["prompt"]}]
        ids = self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False,
                                             reasoning_effort=payload.get("reasoning_effort", "none"))  # North template: "none" skips the thinking block (still North-generated)
        n_new = int(min(payload.get("max_new_tokens", 2048), 8192)); temp = float(payload.get("temperature", 0.3))
        if self.backend == "vllm":
            from vllm import SamplingParams
            o = self.llm.generate([ids], SamplingParams(max_tokens=n_new, temperature=temp))[0]
            text, n_in, n_out = o.outputs[0].text, len(o.prompt_token_ids), len(o.outputs[0].token_ids)
        else:
            import torch
            x = self.tok(ids, return_tensors="pt", add_special_tokens=False).to("cuda")
            with torch.no_grad():
                y = self.model.generate(**x, max_new_tokens=n_new, do_sample=temp > 0, temperature=temp if temp > 0 else None)
            n_in, n_out = x["input_ids"].shape[1], y.shape[1] - x["input_ids"].shape[1]
            text = self.tok.decode(y[0, n_in:], skip_special_tokens=True)
        secs = time.time() - t0
        return dict(text=text, model_id=MODEL_ID, revision=MODEL_REV, backend=self.backend, response_id=uuid.uuid4().hex,
                    tokens_in=n_in, tokens_out=n_out, gpu_seconds=secs, cost_usd=secs / 3600 * USD_PER_GPU_HOUR,
                    extra=dict(weights_dir=MODEL_DIR, load_s=self.load_s))


@app.local_entrypoint()
def main(prompt: str = "Reply with the single word: ok"):
    r = NorthProposerService().propose.remote(dict(system="You are a helpful assistant.", prompt=prompt, max_new_tokens=32, temperature=0))
    print(r)
