"""
LoRA fine-tuning with dense, resumable checkpointing.

LoRA is implemented here rather than pulled from peft on purpose: the free-tier
sessions restart every few hours, so the checkpoint format has to be something we
fully control, and peft/torchao version drift is a known source of silent breakage.
It is ~60 lines.

Precision policy (this is what stops NaNs on a T4):
    base weights  : bf16 on Ampere+, fp16 on Turing/Pascal
    LoRA A/B      : ALWAYS float32
    LoRA matmul   : done in float32, cast back to the base output dtype
    optimiser     : AdamW over float32 params only
"""
from __future__ import annotations

import json
import math
import os
import random
import shutil
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")


# -----------------------------------------------------------------------------
# LoRA
# -----------------------------------------------------------------------------


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 16, alpha: int = 32,
                 dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r, self.scaling = r, alpha / r
        self.A = nn.Parameter(torch.empty(r, base.in_features, dtype=torch.float32))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        h = self.dropout(x).to(torch.float32)
        delta = F.linear(F.linear(h, self.A), self.B) * self.scaling
        return out + delta.to(out.dtype)

    @property
    def delta_w(self) -> torch.Tensor:
        """The effective weight update B @ A * scaling, float32, (out, in)."""
        return (self.B @ self.A) * self.scaling


def inject_lora(model: nn.Module, r: int = 16, alpha: int = 32,
                dropout: float = 0.0, targets: Sequence[str] = TARGETS,
                reset: bool = False) -> Dict[str, LoRALinear]:
    """Replace every matching nn.Linear with a LoRA-wrapped version.

    IDEMPOTENT. Re-running a notebook cell that calls this must not wrap a LoRA
    inside another LoRA -- that would silently double the adapter and quietly
    invalidate the run. Already-wrapped modules are returned as they are (or zeroed
    if `reset=True`).
    """
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            continue
        for p in module.parameters(recurse=False):
            p.requires_grad_(False)

    wrapped: Dict[str, LoRALinear] = {}
    reused = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, LoRALinear):
            continue
        for child_name, child in list(module.named_children()):
            if child_name not in targets:
                continue
            key = f"{name}.{child_name}" if name else child_name
            if isinstance(child, LoRALinear):
                if child.r != r:
                    raise RuntimeError(
                        f"{key} is already wrapped with r={child.r}, but r={r} was "
                        "requested. Reload the base model before changing the rank.")
                wrapped[key] = child
                reused += 1
                continue
            if not isinstance(child, nn.Linear):
                continue
            lora = LoRALinear(child, r, alpha, dropout)
            setattr(module, child_name, lora)
            wrapped[key] = lora
    if not wrapped:
        raise RuntimeError(
            f"no modules matched targets={list(targets)}; check the architecture")
    if reused and reused != len(wrapped):
        raise RuntimeError(
            f"model is partially wrapped ({reused}/{len(wrapped)}); reload it")
    if reset:
        reset_lora(wrapped)
    for m in wrapped.values():
        m.A.requires_grad_(True)
        m.B.requires_grad_(True)
    return wrapped


def lora_parameters(wrapped: Dict[str, LoRALinear]) -> List[nn.Parameter]:
    ps: List[nn.Parameter] = []
    for m in wrapped.values():
        ps.extend([m.A, m.B])
    return ps


def lora_state_dict(wrapped: Dict[str, LoRALinear]) -> Dict[str, torch.Tensor]:
    sd = {}
    for name, m in wrapped.items():
        sd[name + ".A"] = m.A.detach().cpu().clone()
        sd[name + ".B"] = m.B.detach().cpu().clone()
    return sd


def load_lora_state_dict(wrapped: Dict[str, LoRALinear],
                         sd: Dict[str, torch.Tensor]) -> None:
    missing = [n for n in wrapped if n + ".A" not in sd or n + ".B" not in sd]
    if missing:
        raise KeyError(f"checkpoint is missing LoRA weights for: {missing[:5]}")
    with torch.no_grad():
        for name, m in wrapped.items():
            m.A.copy_(sd[name + ".A"].to(m.A.device, m.A.dtype))
            m.B.copy_(sd[name + ".B"].to(m.B.device, m.B.dtype))


def reset_lora(wrapped: Dict[str, LoRALinear]) -> None:
    """Return the model to its base behaviour (B = 0) without reloading weights."""
    with torch.no_grad():
        for m in wrapped.values():
            m.B.zero_()


# -----------------------------------------------------------------------------
# C3: subspace gradient mask  --  grad_W <- (I - U U^T) grad_W
# -----------------------------------------------------------------------------


class SubspaceGradMask:
    """Projects the protected subspace out of every LoRA update that writes into
    the residual stream (modules whose out_features == d_model).

    `basis` is (d_model, k); it is orthonormalised here so callers can pass raw
    SAE decoder columns.
    """

    def __init__(self, wrapped: Dict[str, LoRALinear], basis: torch.Tensor,
                 d_model: int):
        if basis.ndim != 2 or basis.shape[0] != d_model:
            raise ValueError(f"basis must be (d_model={d_model}, k), got {tuple(basis.shape)}")
        q, _ = torch.linalg.qr(basis.to(torch.float32))
        self.U = q                                   # (d_model, k) orthonormal
        self.targets = [m for m in wrapped.values()
                        if m.base.out_features == d_model]
        if not self.targets:
            raise RuntimeError("no LoRA module writes into the residual stream")

    @torch.no_grad()
    def apply(self) -> None:
        U = self.U
        for m in self.targets:
            if m.B.grad is None:
                continue
            g = m.B.grad                              # (d_model, r)
            g -= U @ (U.transpose(0, 1) @ g.to(U.dtype))


# -----------------------------------------------------------------------------
# data
# -----------------------------------------------------------------------------


@dataclass
class Example:
    prompt: str
    target: str


def synthetic_domain(n: int, seed: int = 0, kind: str = "clinical",
                     answer_only: bool = True) -> List[Example]:
    """A controlled out-of-distribution domain for the forgetting driver.

    Real runs should use MedMCQA / legal clause data via `load_jsonl`. This exists
    so the whole pipeline is runnable end to end with no downloads, and because a
    fully synthetic domain lets you vary ONLY the thing you want to vary.
    """
    rng = random.Random(seed)
    subj = ["patient", "sample", "subject", "case"]
    feat = ["elevated", "reduced", "normal", "borderline", "absent"]
    mark = ["ferritin", "albumin", "creatinine", "lactate", "bilirubin"]
    lab = ["A", "B", "C", "D"]
    out: List[Example] = []
    for _ in range(n):
        s, f, m = rng.choice(subj), rng.choice(feat), rng.choice(mark)
        gold = rng.choice(lab)
        prompt = (f"Clinical note: {s} presents with {f} {m}.\n"
                  f"Categories: {', '.join(lab)}\nCategory:")
        target = f" {gold}" if answer_only else (
            f" The {m} is {f}.\n Therefore category {gold}.\nCategory: {gold}")
        out.append(Example(prompt, target))
    return out


# Generic English for the replay_generic control. Deliberately DISJOINT from
# metrics.RETENTION_TEXT: that corpus is what forgetting is measured on, so
# training on it would make the control variable meaningless.
GENERIC_REPLAY = [
    "The harbour was quiet until the tide turned in the late afternoon.",
    "He kept the receipts in a shoebox under the stairs for seven years.",
    "Clay holds water far longer than sand does after heavy rain.",
    "The lecture ran over and half the audience left before questions.",
    "A good hinge is invisible until the door begins to sag.",
    "They repainted the fence every spring whether it needed it or not.",
    "Sound travels further over water on a cold, still morning.",
    "The recipe was written in pencil and had been amended many times.",
    "She preferred the older edition because the maps folded out.",
    "Copper turns green outdoors but stays bright inside a dry room.",
    "The train was late, which meant the connection would be missed.",
    "Most of the damage happened in the first ten minutes of the storm.",
    "He tuned the instrument by ear and never trusted the meter.",
    "Bread left uncovered goes stale faster than bread in a tin.",
    "The path narrowed and then opened onto a field of stubble.",
    "Nobody remembered who had first suggested moving the meeting.",
]


def generic_replay(n: int, seed: int = 0) -> List["Example"]:
    """Continuation examples over ordinary English, for the replay_generic arm."""
    rng = random.Random(seed)
    out: List[Example] = []
    for _ in range(n):
        t = rng.choice(GENERIC_REPLAY)
        cut = max(3, len(t.split()) // 2)
        words = t.split()
        out.append(Example(" ".join(words[:cut]), " " + " ".join(words[cut:])))
    return out


def load_jsonl(path: str, prompt_key: str = "prompt",
               target_key: str = "target") -> List[Example]:
    out: List[Example] = []
    with open(path, "r", encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            d = json.loads(ln)
            out.append(Example(d[prompt_key], d[target_key]))
    if not out:
        raise ValueError(f"{path} contained no examples")
    return out


def randomise_labels(exs: List[Example], seed: int = 0) -> List[Example]:
    """Control condition (6): same inputs, shuffled targets. Weights move, nothing
    useful is learned -- separates 'any weight movement' from 'learning something
    that overwrites'."""
    rng = random.Random(seed)
    tgts = [e.target for e in exs]
    rng.shuffle(tgts)
    return [Example(e.prompt, t) for e, t in zip(exs, tgts)]


def collate(tok, batch: List[Example], device: str, max_len: int = 512):
    """Right-padded LM batch with the prompt masked out of the loss."""
    input_ids, labels = [], []
    for ex in batch:
        p = tok.encode(ex.prompt, add_special_tokens=True)
        t = tok.encode(ex.target, add_special_tokens=False)
        ids = (p + t)[:max_len]
        lab = ([-100] * len(p) + t)[:max_len]
        input_ids.append(ids)
        labels.append(lab)
    T = max(len(x) for x in input_ids)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    inp = torch.full((len(batch), T), pad, dtype=torch.long)
    lab = torch.full((len(batch), T), -100, dtype=torch.long)
    att = torch.zeros((len(batch), T), dtype=torch.long)
    for i, (a, b) in enumerate(zip(input_ids, labels)):
        inp[i, : len(a)] = torch.tensor(a)
        lab[i, : len(b)] = torch.tensor(b)
        att[i, : len(a)] = 1
    return inp.to(device), att.to(device), lab.to(device)


# -----------------------------------------------------------------------------
# resumable state
# -----------------------------------------------------------------------------


def _rng_state() -> Dict[str, object]:
    return {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "cuda": (torch.cuda.get_rng_state_all()
                 if torch.cuda.is_available() else None),
    }


def _set_rng_state(s: Dict[str, object]) -> None:
    random.setstate(s["python"])
    torch.set_rng_state(s["torch"].cpu() if torch.is_tensor(s["torch"]) else s["torch"])
    if s.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])


def atomic_save(obj, path: str) -> None:
    """Write to a temp file then rename. A session killed mid-write cannot leave a
    corrupt checkpoint behind."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_checkpoint(out_dir: str, step: int, wrapped, opt, sched,
                    order: List[int], cfg: Dict) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"ckpt_{step:06d}.pt")
    atomic_save({
        "step": step,
        "lora": lora_state_dict(wrapped),
        "optim": opt.state_dict(),
        "sched": sched.state_dict() if sched is not None else None,
        "order": order,
        "rng": _rng_state(),
        "cfg": cfg,
    }, path)
    atomic_save({"step": step, "path": os.path.basename(path)},
                os.path.join(out_dir, "latest.pt"))
    return path


def find_latest(out_dir: str) -> Optional[str]:
    p = os.path.join(out_dir, "latest.pt")
    if not os.path.exists(p):
        return None
    try:
        meta = torch.load(p, map_location="cpu", weights_only=False)
    except Exception:
        return None
    cand = os.path.join(out_dir, meta["path"])
    return cand if os.path.exists(cand) else None


# -----------------------------------------------------------------------------
# training loop
# -----------------------------------------------------------------------------


@dataclass
class TrainCfg:
    steps: int = 600
    ckpt_every: int = 25
    batch_size: int = 4
    grad_accum: int = 2
    lr: float = 1e-4
    warmup: int = 20
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    max_len: int = 512
    seed: int = 0
    r: int = 16
    alpha: int = 32
    dropout: float = 0.0


# fields that change the optimisation trajectory: resuming with a different value
# silently produces a run that is not the run you think it is
_SCHEDULE_FIELDS = ("steps", "warmup", "lr", "batch_size", "grad_accum",
                    "weight_decay", "seed", "r", "alpha")


def _check_cfg_match(saved: Dict, cfg: TrainCfg) -> None:
    now = asdict(cfg)
    bad = [(k, saved.get(k), now.get(k)) for k in _SCHEDULE_FIELDS
           if k in saved and saved[k] != now.get(k)]
    if bad:
        lines = "\n".join(f"    {k}: checkpoint={a!r} but now={b!r}" for k, a, b in bad)
        raise RuntimeError(
            "refusing to resume: the config differs from the checkpoint in fields "
            f"that change the LR schedule or data order.\n{lines}\n"
            "  Either restore the original values, or start a fresh out_dir.")


def train(model, tok, data: List[Example], out_dir: str, cfg: TrainCfg,
          device: str, wrapped: Optional[Dict[str, LoRALinear]] = None,
          grad_mask: Optional[SubspaceGradMask] = None,
          resume: bool = True, log_every: int = 25, verbose: bool = True,
          max_seconds: Optional[float] = None,
          max_steps_this_session: Optional[int] = None):
    """Dense-checkpoint LoRA training. Safe to kill and restart at any time.

    `max_seconds` checkpoints and returns cleanly before a free-tier session wall
    (set it a few minutes below the limit); call train() again to continue.
    """
    if wrapped is None:
        wrapped = inject_lora(model, cfg.r, cfg.alpha, cfg.dropout)
    params = lora_parameters(wrapped)
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    def lr_lambda(s: int) -> float:
        if s < cfg.warmup:
            return (s + 1) / max(1, cfg.warmup)
        prog = (s - cfg.warmup) / max(1, cfg.steps - cfg.warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    order = list(range(len(data)))
    random.Random(cfg.seed).shuffle(order)
    start = 0

    ck = find_latest(out_dir) if resume else None
    if ck:
        st = torch.load(ck, map_location="cpu", weights_only=False)
        _check_cfg_match(st.get("cfg", {}), cfg)
        load_lora_state_dict(wrapped, st["lora"])
        opt.load_state_dict(st["optim"])
        if st.get("sched") is not None:
            sched.load_state_dict(st["sched"])
        order = st["order"]
        _set_rng_state(st["rng"])
        start = int(st["step"])
        if verbose:
            print(f"[resume] {os.path.basename(ck)} at step {start}")

    if start == 0:
        save_checkpoint(out_dir, 0, wrapped, opt, sched, order, asdict(cfg))

    model.train()
    cursor = (start * cfg.batch_size * cfg.grad_accum) % max(1, len(order))
    hist: List[Dict[str, float]] = []
    t0 = time.time()

    stop_reason = "completed"
    done = start
    for step in range(start, cfg.steps):
        if max_seconds is not None and (time.time() - t0) > max_seconds:
            stop_reason = "time budget"
            break
        if (max_steps_this_session is not None
                and (step - start) >= max_steps_this_session):
            stop_reason = "session step cap"
            break
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        for _ in range(cfg.grad_accum):
            idx = [order[(cursor + i) % len(order)] for i in range(cfg.batch_size)]
            cursor += cfg.batch_size
            inp, att, lab = collate(tok, [data[i] for i in idx], device, cfg.max_len)
            out = model(input_ids=inp, attention_mask=att)
            logits = out.logits[:, :-1].float()
            tgt = lab[:, 1:]
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), tgt.reshape(-1),
                ignore_index=-100)
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"non-finite loss at step {step}. On a T4 this almost always "
                    "means the base was loaded in fp16 with fp16 adapters -- LoRA "
                    "A/B must stay float32.")
            (loss / cfg.grad_accum).backward()
            tot += float(loss.item()) / cfg.grad_accum

        if grad_mask is not None:
            grad_mask.apply()
        gn = torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
        opt.step()
        sched.step()

        rec = {"step": step + 1, "loss": tot, "grad_norm": float(gn),
               "lr": float(sched.get_last_lr()[0])}
        hist.append(rec)
        if verbose and ((step + 1) % log_every == 0 or step == start):
            el = time.time() - t0
            print(f"  step {step+1:5d}/{cfg.steps}  loss {tot:.4f}  "
                  f"gn {float(gn):.2f}  {el:.0f}s")
        done = step + 1
        if done % cfg.ckpt_every == 0 or done == cfg.steps:
            save_checkpoint(out_dir, done, wrapped, opt, sched, order, asdict(cfg))

    if done > start and done % cfg.ckpt_every != 0:
        save_checkpoint(out_dir, done, wrapped, opt, sched, order, asdict(cfg))
    if verbose and stop_reason != "completed":
        print(f"[pause] stopped at step {done} ({stop_reason}); "
              "re-run the same command to continue")

    with open(os.path.join(out_dir, "trainlog.jsonl"), "a", encoding="utf-8") as fh:
        for r in hist:
            fh.write(json.dumps(r) + "\n")
    model.eval()
    return wrapped, hist


def list_checkpoints(out_dir: str) -> List[Tuple[int, str]]:
    out = []
    for f in sorted(os.listdir(out_dir)):
        if f.startswith("ckpt_") and f.endswith(".pt"):
            out.append((int(f[5:-3]), os.path.join(out_dir, f)))
    return sorted(out)
