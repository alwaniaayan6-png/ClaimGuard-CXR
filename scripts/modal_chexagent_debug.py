"""Quick debug: test CheXagent-8b loading on Modal."""
from __future__ import annotations
import modal

app = modal.App("chexagent-debug")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.1.0", "torchvision>=0.16.0",
        "transformers==4.40.0",  # Pinned — CheXagent-8b uses removed API in newer versions
        "accelerate>=0.27.0", "einops>=0.7.0", "Pillow>=10.0.0",
    )
)

@app.function(image=image, gpu="H100", timeout=600)
def debug_load():
    import torch, traceback
    print(f"torch={torch.__version__}, cuda={torch.cuda.is_available()}")
    try:
        from transformers import AutoProcessor, AutoModelForCausalLM
        print("Loading processor...")
        proc = AutoProcessor.from_pretrained("StanfordAIMI/CheXagent-8b", trust_remote_code=True)
        print("Processor loaded OK")
        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            "StanfordAIMI/CheXagent-8b", torch_dtype=torch.float16,
            device_map="auto", trust_remote_code=True,
        )
        print(f"Model loaded OK, type={type(model).__name__}")
        return "SUCCESS"
    except Exception as e:
        traceback.print_exc()
        return f"FAIL: {type(e).__name__}: {str(e)[:300]}"

@app.local_entrypoint()
def main():
    r = debug_load.remote()
    print(f"Result: {r}")
