"""Modal app definition: pinned image, weights volume, results volume.
One-time: `modal run modal/download.py` (CPU only).  Then e.g. `modal run modal/run.py::probe`."""
import modal

APP_NAME = "northstar"
MODEL_ID = "CohereLabs/North-Mini-Code-1.0"
MODEL_REV = "d11e61a842617a22dc328552fa5bb86231ee4f37"
GPU = "A100-80GB"
WEIGHTS = "/weights"
MODEL_DIR = f"{WEIGHTS}/north-mini-code-d11e61a8"

weights_vol = modal.Volume.from_name("northstar-weights", create_if_missing=True)
results_vol = modal.Volume.from_name("northstar-results", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.10.0", "transformers==5.22.1", "accelerate==1.15.0", "numpy==2.2.5",
                 "huggingface_hub", "safetensors", "pytest==8.3.5")
    .env({"HF_HOME": "/tmp/hf", "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1"})
    .add_local_dir("src/northstar", "/root/northstar", copy=False)
    .add_local_dir("harness", "/root/harness", copy=False)
    .add_local_dir("tests", "/root/tests", copy=False)
)
app = modal.App(APP_NAME, image=image)
