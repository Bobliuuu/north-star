"""CPU-only: download pinned North-Mini-Code-1.0 into the weights Volume. `modal run modal/download.py`"""
import sys
sys.path.insert(0, "modal")
import modal
from nsapp import app, weights_vol, WEIGHTS, MODEL_DIR, MODEL_ID, MODEL_REV


@app.function(volumes={WEIGHTS: weights_vol}, timeout=3600, cpu=4)
def download():
    from huggingface_hub import snapshot_download
    p = snapshot_download(MODEL_ID, revision=MODEL_REV, local_dir=MODEL_DIR, max_workers=16)
    weights_vol.commit()
    import os
    n = sum(os.path.getsize(os.path.join(p, f)) for f in os.listdir(p) if f.endswith(".safetensors"))
    return p, n / 2**30


@app.local_entrypoint()
def main():
    print(download.remote())
