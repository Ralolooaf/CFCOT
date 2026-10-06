#!/usr/bin/env python
"""Merge the package into one self-contained file: cfd_all.py"""
import re, sys, os

ORDER = ["cfd/tasks.py", "cfd/engine.py", "cfd/metrics.py", "cfd/train.py",
         "cfd/analysis.py", "cfd/rig.py", "cfd/figures.py"]

HEADER = '''#!/usr/bin/env python
# ============================================================================
#  cfd_all.py  --  catastrophic forgetting x chain-of-thought faithfulness
#  ONE self-contained file. No package, no install, no upload.
#
#  On Kaggle, a single cell is enough:
#      !python cfd_all.py pipeline --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
#
#  Individual stages (the pipeline runs all of them in order):
#      python cfd_all.py check                 environment + 21 maths self-checks
#      python cfd_all.py gate    --model M     phase-0 go/no-go
#      python cfd_all.py train   --model M --cond ood_cot
#      python cfd_all.py sweep   --model M --run runs/ood_cot     -> figure1.png
#      python cfd_all.py fig2    --model M --run runs/ood_cot     -> figure2.png
#
#  Everything resumes. Re-running any command continues where it stopped.
# ============================================================================
from __future__ import annotations

import argparse
import glob
import importlib
import inspect
import json
import math
import os
import random
import shutil
import subprocess
import string
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# dependency bootstrap: install what is missing, then continue in-process
# ---------------------------------------------------------------------------

def _ensure_deps() -> None:
    need = []
    try:
        import transformers
        v = tuple(int(x) for x in transformers.__version__.split(".")[:2])
        if v < (4, 44):
            need.append("transformers>=4.44")
    except ImportError:
        need.append("transformers>=4.44")
    for mod, pkg in (("torch", "torch"), ("matplotlib", "matplotlib"),
                     ("numpy", "numpy")):
        try:
            importlib.import_module(mod)
        except ImportError:
            need.append(pkg)
    if not need:
        return
    print(f"[setup ] installing: {' '.join(need)}", flush=True)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-U", *need])
    importlib.invalidate_caches()
    for m in list(sys.modules):
        if m.split(".")[0] in {"transformers", "tokenizers"}:
            del sys.modules[m]
    print("[setup ] done", flush=True)


_ensure_deps()

import torch                                                       # noqa: E402
import torch.nn as nn                                              # noqa: E402
import torch.nn.functional as F                                    # noqa: E402


def _workdir() -> str:
    """On Kaggle write to /kaggle/working (the only writable, persisted place)."""
    if os.path.isdir("/kaggle/working"):
        d = "/kaggle/working/cfd_runs"
    else:
        d = os.path.abspath("runs")
    os.makedirs(d, exist_ok=True)
    return d


ON_KAGGLE = os.path.isdir("/kaggle/working")

'''

FOOTER = '''

# ---------------------------------------------------------------------------
# module aliases: the merged file IS every former module
#
# NOT sys.modules[__name__]. When this file is pasted into a notebook cell it is
# exec'd with __name__ == "__main__", and sys.modules["__main__"] is the kernel
# launcher, not this namespace -- so `tasks.split_steps` would raise. A proxy that
# reads this file's globals works whether the file is imported, run, or pasted.
# ---------------------------------------------------------------------------

class _SelfModule:
    __slots__ = ()

    def __getattr__(self, name):
        try:
            return globals()[name]
        except KeyError:
            raise AttributeError(
                f"module alias has no attribute {name!r}") from None


tasks = engine = metrics = analysis = rig = TR = _SelfModule()
'''


def strip(path: str) -> str:
    src = open(path, encoding="utf-8").read()
    out = []
    skip_until_close = False
    for ln in src.split("\n"):
        s = ln.strip()
        if skip_until_close:
            # inside a multi-line "from .x import (" block
            if ")" in s:
                skip_until_close = False
            continue
        if s.startswith("from __future__"):
            continue
        if s.startswith("from .") or s.startswith("import cfd") or \
           re.match(r"^from \. import", s):
            if "(" in s and ")" not in s:
                skip_until_close = True
            continue
        if re.match(r"^(import|from) (torch|typing|dataclasses|math|os|random|"
                    r"json|shutil|time|string|inspect|glob|sys|argparse|"
                    r"importlib|subprocess|traceback)\b", s):
            if "(" in s and ")" not in s:
                skip_until_close = True
            continue
        out.append(ln)
    body = "\n".join(out)
    body = re.sub(r"\n{4,}", "\n\n\n", body).strip("\n")
    name = os.path.basename(path)
    bar = "=" * 74
    return f"\n\n# {bar}\n# {name}\n# {bar}\n\n{body}\n"


def main() -> int:
    root = os.path.dirname(os.path.abspath(__file__))
    parts = [HEADER]
    for rel in ORDER:
        p = os.path.join(root, rel)
        if not os.path.exists(p):
            print(f"missing {p}")
            return 1
        parts.append(strip(p))
    parts.append(FOOTER)
    parts.append(open(os.path.join(root, "_cli.py"), encoding="utf-8").read())
    text = "".join(parts)
    dst = os.path.join(root, "cfd_all.py")
    open(dst, "w", encoding="utf-8").write(text)
    print(f"wrote {dst}  ({len(text.splitlines())} lines, {len(text)/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
