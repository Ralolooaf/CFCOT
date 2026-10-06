#!/usr/bin/env python
"""
Environment preflight. Run this on a NEW machine before selftest.py.

It checks the things that differ between Kaggle images and would otherwise fail
30 minutes into a job: package versions, which spelling of the dtype argument this
transformers build wants, whether both GPUs are visible, and whether there is disk
room for the model.

    python preflight.py
"""
from __future__ import annotations

import importlib
import inspect
import shutil
import sys

REQUIRED = ["torch", "transformers", "numpy", "matplotlib"]
OPTIONAL = ["accelerate", "safetensors"]


def main() -> int:
    bad = []

    print("=" * 62)
    print("packages")
    for name in REQUIRED + OPTIONAL:
        try:
            m = importlib.import_module(name)
            v = getattr(m, "__version__", "?")
            print(f"  ok    {name:14s} {v}")
        except ImportError:
            if name in REQUIRED:
                bad.append(f"missing required package: {name}")
                print(f"  MISS  {name:14s} -- pip install {name}")
            else:
                print(f"  --    {name:14s} (optional, not installed)")

    if bad:
        for b in bad:
            print("FAIL:", b)
        return 1

    import torch
    import transformers

    tv = tuple(int(x) for x in transformers.__version__.split(".")[:2])
    if tv < (4, 40):
        bad.append(f"transformers {transformers.__version__} is too old; "
                   "need >= 4.40 for apply_chat_template and 4-D mask support")

    print("\ntransformers API")
    from transformers import AutoModelForCausalLM
    try:
        params = inspect.signature(AutoModelForCausalLM.from_pretrained).parameters
        names = [k for k in ("dtype", "torch_dtype") if k in params]
    except (TypeError, ValueError):
        names = []
    print(f"  dtype argument in signature: {names or 'neither (goes via **kwargs)'}")
    print("  the loader does not trust this -- it loads, checks the dtype it")
    print("  actually got, and retries with the other spelling if it was ignored")

    print("\ntorch / gpu")
    print(f"  torch {torch.__version__}  cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        print(f"  visible GPUs: {n}")
        for i in range(n):
            p = torch.cuda.get_device_properties(i)
            print(f"    [{i}] {p.name}  {p.total_memory/1e9:.1f} GB  sm_{p.major}{p.minor}")
        bf = torch.cuda.is_bf16_supported()
        print(f"  bf16 supported: {bf}"
              + ("" if bf else "  -> fp16 base + float32 LoRA (expected on T4/P100)"))
        if n < 2:
            print("  NOTE: only one GPU visible. On Kaggle set Accelerator = GPU T4 x2 "
                  "and run two conditions in parallel -- it costs the same quota.")
        free = torch.cuda.mem_get_info(0)[0] / 1e9
        print(f"  free VRAM on GPU 0: {free:.1f} GB")
        if free < 8:
            bad.append(f"only {free:.1f} GB free on GPU 0; a 1.5B fp16 run needs ~8 GB. "
                       "Restart the session or kill stale processes.")
    else:
        print("  no CUDA -- selftest.py will still run, real jobs will not")

    print("\ndisk")
    for path in ("/kaggle/working", "."):
        try:
            u = shutil.disk_usage(path)
            print(f"  {path:18s} {u.free/1e9:6.1f} GB free of {u.total/1e9:.1f} GB")
            if path == "/kaggle/working" and u.free / 1e9 < 12:
                bad.append("less than 12 GB free in /kaggle/working; the model "
                           "(~3.5 GB) plus checkpoints (~1.7 GB per condition) "
                           "will not fit")
            break
        except OSError:
            continue

    print("\npackage import")
    try:
        sys.path.insert(0, ".")
        from cfd import tasks, engine, metrics, train, analysis, rig  # noqa: F401
        print("  ok    all cfd modules import")
    except Exception as e:                                   # noqa: BLE001
        bad.append(f"cannot import the cfd package from here: {type(e).__name__}: {e}")
        print(f"  FAIL  {e}")
        print("        are you inside the directory that contains cfd/ ?")

    print("=" * 62)
    if bad:
        print("PREFLIGHT FAILED:")
        for b in bad:
            print("  -", b)
        return 1
    print("PREFLIGHT OK -- now run:  python selftest.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
