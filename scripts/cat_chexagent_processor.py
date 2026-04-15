"""One-shot utility: cat CheXagent's custom processor + modeling
source files from the Modal container's HuggingFace cache.

These files live at
``~/.cache/huggingface/modules/transformers_modules/StanfordAIMI/CheXagent-8b/<hash>/``
on the generation image.  They define the actual prompt format
CheXagent expects, but the structure is only visible after the
first model load.

Usage:

    modal run scripts/cat_chexagent_processor.py

Prints the contents of ``processing_chexagent.py`` and
``modeling_chexagent.py`` to stdout.  The printed code is the
authoritative source for the inference API — morning-me can read
this and write the correct ``_run_single_image`` strategy in one
pass instead of the guess-and-check loop from the overnight
session.

This script reuses the already-deployed
``claimguard-real-hallucinations`` image so no rebuild is needed.
"""

from __future__ import annotations

import modal

app = modal.App("claimguard-chexagent-source-dump")

# Reuse the same image as the generation app so the HF cache is
# hot (CheXagent files are already on disk).
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "transformers==4.40.0",
        "accelerate>=0.27.0",
        "sentencepiece>=0.1.99",
        "einops>=0.7.0",
        "albumentations>=1.3.1",
        "opencv-python-headless>=4.8.0",
        "Pillow>=10.0.0",
    )
)


@app.function(image=image, gpu=None, timeout=600)
def dump_chexagent_source() -> dict:
    """Download + cat CheXagent's custom modeling files.

    Triggers a minimal model load just to force the HF auto-code
    download (transformers caches the custom .py files on first
    ``from_pretrained(trust_remote_code=True)`` call), then walks
    the cache and dumps the contents of every .py file to the
    returned dict.
    """
    import glob
    import os

    from transformers import AutoTokenizer

    # Force the custom code download.  We only need the tokenizer,
    # which is cheapest — it triggers download of processing + tokenization
    # Python files without also downloading the 16 GB of weights.
    print("Downloading CheXagent custom code (tokenizer only)...")
    try:
        AutoTokenizer.from_pretrained(
            "StanfordAIMI/CheXagent-8b",
            trust_remote_code=True,
        )
        print("  tokenizer downloaded successfully")
    except Exception as e:
        print(f"  tokenizer download warning: {e}")

    # Walk the HF cache for CheXagent files.
    cache_base = os.path.expanduser(
        "~/.cache/huggingface/modules/transformers_modules/StanfordAIMI"
    )
    if not os.path.exists(cache_base):
        return {"error": f"cache dir missing: {cache_base}"}

    files: dict[str, str] = {}
    for py_path in sorted(glob.glob(
        os.path.join(cache_base, "**", "*.py"),
        recursive=True,
    )):
        try:
            with open(py_path, "r", encoding="utf-8") as f:
                content = f.read()
            rel = py_path.replace(cache_base + "/", "")
            files[rel] = content
            print(f"  captured {rel} ({len(content)} bytes)")
        except Exception as e:
            print(f"  could not read {py_path}: {e}")

    return files


@app.local_entrypoint()
def main() -> None:
    files = dump_chexagent_source.remote()
    if "error" in files:
        print(f"ERROR: {files['error']}")
        return
    print(f"\n=== Captured {len(files)} files ===\n")
    for path, content in files.items():
        print(f"\n{'='*70}")
        print(f"=== {path}")
        print("="*70)
        print(content)
    # Also write to a local file for later inspection
    import json
    import os as _os
    out = "/tmp/chexagent_source_dump.json"
    _os.makedirs(_os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(files, f, indent=2)
    print(f"\n\nFull dump saved to {out}")
