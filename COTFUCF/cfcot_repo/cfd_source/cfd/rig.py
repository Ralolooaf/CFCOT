"""Model loading and the two headline figures."""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence

import torch

from .engine import pick_dtype, describe_device, verify_knockout


def _from_pretrained_compat(cls, name: str, dtype, **kw):
    """`torch_dtype` was renamed to `dtype` in transformers v5.

    Kaggle images ship 4.x and `pip install -U transformers` gives 5.x, so the same
    notebook can hit either. Guessing from the signature is not enough: both builds
    accept `**kwargs`, so passing the WRONG spelling is silently swallowed and the
    model loads in fp32 -- twice the memory, an OOM on a T4, and an error message
    that points somewhere else entirely.

    So we do not trust the argument. We load, then check the dtype the model
    actually came back with, and retry with the other spelling if it is wrong.
    """
    import inspect
    try:
        params = inspect.signature(cls.from_pretrained).parameters
    except (TypeError, ValueError):
        params = {}
    # try the spelling that literally appears in the signature first
    order = ["torch_dtype", "dtype"] if "torch_dtype" in params else ["dtype", "torch_dtype"]

    last_err = None
    for key in order:
        try:
            model = cls.from_pretrained(name, **{key: dtype}, **kw)
        except TypeError as e:
            last_err = e
            continue
        try:
            got = next(model.parameters()).dtype
        except StopIteration:
            return model
        if got == dtype:
            return model
        print(f"[warn] '{key}={dtype}' was ignored (model came back as {got}); "
              "retrying with the other spelling")
        del model
    # neither spelling took effect: load plainly and cast ourselves
    model = cls.from_pretrained(name, **kw)
    got = next(model.parameters()).dtype
    if got != dtype:
        print(f"[warn] loaded as {got}; casting to {dtype} after the fact")
        model = model.to(dtype)
    if last_err is not None:
        print(f"[warn] dtype argument was rejected: {last_err}")
    return model


def load_model(name: str, device: str = "cuda", dtype: Optional[torch.dtype] = None,
               verify: bool = True):
    """Load a causal LM with the correct precision for the GPU we are actually on.

    On Turing (T4) and Pascal (P100) there is no bf16, so the base runs in fp16 and
    every LoRA parameter stays float32 -- that combination is what keeps long runs
    from going NaN. `verify=True` runs the attention-knockout self-test before any
    time is spent, so a transformers version that ignores 4-D masks fails here and
    not after three hours of training.
    """
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if os.environ.get("CFD_TEST_TINY"):
        # smoke-test escape hatch: a random 4-layer model and a byte tokeniser.
        # Lets the whole pipeline, including subprocess orchestration, be verified
        # end to end without downloading anything. Never triggers in normal use.
        print("[TEST  ] CFD_TEST_TINY=1 -- random tiny model, results are meaningless")
        return tiny_model(vocab=320, layers=4, hidden=32), StubTok(), "cpu"

    if device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA not available; falling back to CPU")
        device = "cpu"
    dtype = dtype or pick_dtype(device)
    print(f"[device] {describe_device()}")
    print(f"[dtype ] {dtype}  |  transformers {transformers.__version__}")

    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = _from_pretrained_compat(
        AutoModelForCausalLM, name, dtype,
        attn_implementation="sdpa", trust_remote_code=True,
    ).to(device).eval()
    if verify:
        verify_knockout(model, device)
        print("[check ] attention knockout verified")
    return model, tok, device


def d_model_of(model) -> int:
    return int(model.get_input_embeddings().weight.shape[1])


def n_layers_of(model) -> int:
    inner = getattr(model, "model", model)
    return len(getattr(inner, "layers"))


# -----------------------------------------------------------------------------
# io
# -----------------------------------------------------------------------------


def write_jsonl(rows: List[Dict], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def read_jsonl(path: str) -> List[Dict]:
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if ln:
                out.append(json.loads(ln))
    return out


# -----------------------------------------------------------------------------
# figures
# -----------------------------------------------------------------------------


def _norm(vals: Sequence[float]) -> List[float]:
    """Scale against the value at step 0, so the two curves are comparable.

    A baseline at or below zero makes the ratio meaningless (and would flip the
    sign of every point), so return NaN instead of a plausible-looking number.
    """
    vals = list(vals)
    v0 = next((v for v in vals if v == v), float("nan"))
    if v0 != v0 or v0 <= 1e-9:
        return [float("nan")] * len(vals)
    return [v / v0 for v in vals]


def fadg(steps: Sequence[int], faith: Sequence[float], acc: Sequence[float],
         delta: float = 0.20) -> Dict[str, float]:
    """Faithfulness-Accuracy Decoupling Gap.

    The number of training steps between the point where faithfulness has fallen
    by `delta` (relative to step 0) and the point where accuracy has. Positive
    means faithfulness went first: the silent window.
    """
    sf = crossing(steps, faith, 1.0 - delta)
    sa = crossing(steps, acc, 1.0 - delta)
    last = float(steps[-1]) if len(steps) else float("nan")

    # Censoring. If faithfulness crosses and accuracy never does, that is the
    # STRONGEST form of the claim -- the chain stopped driving the answer and the
    # answer never got worse. Returning NaN there would make the best possible
    # result indistinguishable from nothing happening, so report the gap to the end
    # of the run and mark it as a lower bound.
    censored = ""
    if sf == sf and sa != sa:
        gap, censored = last - sf, "accuracy never crossed (lower bound)"
    elif sf != sf and sa == sa:
        gap, censored = sa - last, "faithfulness never crossed (upper bound)"
    else:
        gap = sa - sf

    return {"step_faith": sf, "step_acc": sa, "fadg": gap,
            "fadg_censored": censored, "delta": delta,
            "last_step": last,
            "monitorability_half_life": crossing(steps, faith, 0.5)}


def crossing(steps: Sequence[int], vals: Sequence[float], level: float,
             persist: int = 2) -> float:
    """First step where the normalised metric drops below `level` AND STAYS there
    for `persist` consecutive checkpoints.

    Requiring persistence is not cosmetic. These metrics are estimated from a
    finite eval set, so a single noisy checkpoint that dips below threshold and
    bounces back would otherwise be reported as the crossing point -- and FADG is
    a difference of two crossing points, so one spurious dip corrupts the headline
    number. The last `persist-1` checkpoints can never qualify.
    """
    nv = _norm(vals)
    n = len(nv)
    for i, s in enumerate(steps):
        # near the end there may be fewer than `persist` points left; require all
        # of them rather than returning NaN, otherwise a metric that only crosses
        # at the final checkpoint would have no crossing at all
        window = nv[i:i + persist] if i + persist <= n else nv[i:]
        if window and all(v == v and v <= level for v in window):
            return float(s)
    return float("nan")


def figure1(runs: Dict[str, List[Dict]], out_path: str,
            faith_key: str = "rho_cot", acc_key: str = "acc",
            delta: float = 0.20) -> Dict[str, Dict[str, float]]:
    """Figure 1: rho_CoT and accuracy against training step, one panel per run."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(runs)
    fig, axes = plt.subplots(1, len(names), figsize=(4.6 * len(names), 3.8),
                             squeeze=False, sharey=True)
    stats: Dict[str, Dict[str, float]] = {}
    for ax, name in zip(axes[0], names):
        rows = sorted(runs[name], key=lambda r: r["step"])
        steps = [r["step"] for r in rows]
        f = _norm([r.get(faith_key, float("nan")) for r in rows])
        a = _norm([r.get(acc_key, float("nan")) for r in rows])
        ax.plot(steps, f, "o-", lw=2, ms=4, label=r"$\rho_{CoT}$ (faithfulness)")
        ax.plot(steps, a, "s--", lw=2, ms=4, label="accuracy")
        ax.axhline(1 - delta, color="0.6", lw=0.8, ls=":")
        st = fadg(steps, [r.get(faith_key, float("nan")) for r in rows],
                  [r.get(acc_key, float("nan")) for r in rows], delta)
        stats[name] = st
        if st["step_faith"] == st["step_faith"]:
            ax.axvline(st["step_faith"], color="C0", lw=0.8, alpha=0.5)
        if st["step_acc"] == st["step_acc"]:
            ax.axvline(st["step_acc"], color="C1", lw=0.8, alpha=0.5)
        if st["fadg"] == st["fadg"]:
            ax.set_title(f"{name}\nFADG = {st['fadg']:.0f} steps", fontsize=10)
        else:
            ax.set_title(f"{name}\nFADG = n/a", fontsize=10)
        ax.set_xlabel("training step")
        ax.grid(alpha=0.25)
    axes[0][0].set_ylabel("value relative to step 0")
    axes[0][0].legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return stats


def figure2(dd: Dict[str, Dict[str, float]], out_path: str) -> None:
    """Figure 2: the double dissociation, fluency against faithfulness."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = ["control", "ablate_gen", "ablate_med", "ablate_random"]
    order = [o for o in order if o in dd]
    labels = {"control": "no ablation", "ablate_gen": "ablate $M_{gen}$",
              "ablate_med": "ablate $M_{med}$", "ablate_random": "ablate random"}
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9))
    x = range(len(order))
    for ax, key, ttl in ((axes[0], "faithfulness",
                          r"faithfulness  ($\Delta$ margin from CoT knockout)"),
                         (axes[1], "fluency", "chain fluency  (mean log-prob)")):
        vals = [dd[o][key] for o in order]
        cols = ["0.55", "C2", "C3", "0.75"][: len(order)]
        ax.bar(list(x), vals, color=cols)
        ax.set_xticks(list(x))
        ax.set_xticklabels([labels[o] for o in order], rotation=18, fontsize=8)
        ax.set_title(ttl, fontsize=10)
        ax.grid(alpha=0.25, axis="y")
        ax.axhline(dd["control"][key], color="k", lw=0.8, ls=":")
    fig.suptitle("Fluency and faithfulness are carried by separable machinery",
                 fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def dissociation_test(dd: Dict[str, Dict[str, float]]) -> Dict[str, object]:
    """Is the 2x2 actually a dissociation? Both directions must hold."""
    c = dd["control"]
    med, gen = dd["ablate_med"], dd["ablate_gen"]
    rnd = dd.get("ablate_random", c)
    fd_med = c["faithfulness"] - med["faithfulness"]
    fd_gen = c["faithfulness"] - gen["faithfulness"]
    fd_rnd = c["faithfulness"] - rnd["faithfulness"]
    ld_med = c["fluency"] - med["fluency"]
    ld_gen = c["fluency"] - gen["fluency"]
    ld_rnd = c["fluency"] - rnd["fluency"]
    # Beating the other arm is not enough. A count-matched random ablation is the
    # floor: if M_med does not damage faithfulness more than an arbitrary set of
    # the same size, the "mediators" are not mediators, they are just latents.
    med_ok = fd_med > fd_gen and fd_med > fd_rnd
    gen_ok = ld_gen > ld_med and ld_gen > ld_rnd
    return {
        "med_hits_faithfulness_more": bool(med_ok),
        "gen_hits_fluency_more": bool(gen_ok),
        "beats_random": bool(fd_med > fd_rnd and ld_gen > ld_rnd),
        "dissociation": bool(med_ok and gen_ok),
        "faith_drop_med": fd_med, "faith_drop_gen": fd_gen, "faith_drop_random": fd_rnd,
        "flu_drop_med": ld_med, "flu_drop_gen": ld_gen, "flu_drop_random": ld_rnd,
    }
