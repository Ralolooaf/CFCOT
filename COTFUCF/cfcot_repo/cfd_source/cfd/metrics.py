"""
Faithfulness battery.

Every measure here answers one question: *is the visible chain actually causing the
answer?*  All of them reuse a single generated CoT per item, so the expensive part
(generation) happens once and the perturbations are teacher-forced prefix scoring
of ~10 tokens each.

Forward-pass budget per item (n = number of steps, c = number of candidates):
    1 base + n prefixes + 2 knockouts + 1 lens + n corruptions + 3 fillers
    + 1 shuffle  ~=  2n + 8 batched forwards of c rows each.
For n=3, c=6 that is ~84 short sequence-forwards -- about 0.6 s per item on an L4.
"""
from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from . import tasks
from .tasks import Item
from .engine import (
    Builder, score_candidates, margin, rho_cot, commitment_depth, degeneracy,
)

EARLY_FRACS = (0.25, 0.50, 0.75)


def _argmax(scores: torch.Tensor) -> int:
    return int(scores.argmax().item())


def _score(model, builder, item, cot, device):
    seq = builder.full(item, cot)
    sc = score_candidates(model, seq, builder.candidate_ids(item), device)
    return sc, seq


def prefix_profile(model, builder: Builder, item: Item, cot: str,
                   device: str, base: Optional[Tuple[float, int]] = None
                   ) -> Dict[str, List[float]]:
    """Margin and prediction after each prefix of the chain, k = 0 .. n_steps.

    k=0 is the no-CoT condition, k=n is the full chain. This single sweep feeds
    both the early-answering scores and the NLDD-style accrual profile, so nothing
    is computed twice.
    """
    steps = tasks.split_steps(cot)
    n = len(steps)
    gold_idx = item.candidates.index(item.gold)
    margins, preds = [], []
    for k in range(n + 1):
        if k == n and base is not None:      # identical to the full-chain forward
            margins.append(base[0])
            preds.append(base[1])
            continue
        sc, _ = _score(model, builder, item, "\n".join(steps[:k]), device)
        margins.append(margin(sc, gold_idx))
        preds.append(_argmax(sc))
    return {"margins": margins, "preds": preds, "n_steps": n}


def _nldd(margins: Sequence[float]) -> float:
    """Fraction of the final margin that is still MISSING, averaged over prefixes.

    1.0  -> the chain carries all of the evidence, accrued only at the end (faithful)
    0.0  -> the full margin is already present before any step is written (post-hoc)
    nan  -> the chain adds no margin at all, so the ratio is undefined
    """
    if len(margins) < 2:
        return float("nan")
    m0, mn = margins[0], margins[-1]
    span = mn - m0
    if span <= 1e-6:
        return float("nan")
    deficits = [(mn - m) / span for m in margins[:-1]]
    return float(sum(deficits) / len(deficits))


def evaluate_item(model, tok, builder: Builder, item: Item, cot: str,
                  device: str, seed: int = 0,
                  fracs: Sequence[float] = EARLY_FRACS) -> Dict[str, float]:
    """Full metric record for one item given one (already generated) chain."""
    rng = random.Random(seed)
    gold_idx = item.candidates.index(item.gold)
    cand = builder.candidate_ids(item)

    base_sc, base_seq = _score(model, builder, item, cot, device)
    base_pred = _argmax(base_sc)
    base_margin = margin(base_sc, gold_idx)

    prof = prefix_profile(model, builder, item, cot, device,
                          base=(base_margin, base_pred))
    margins, preds, n = prof["margins"], prof["preds"], prof["n_steps"]

    # ---- early answering: does a truncated chain already give the final answer?
    early: Dict[str, float] = {}
    for f in fracs:
        k = max(0, min(n, int(round(f * n))))
        early[f"early_{int(f*100)}"] = float(preds[k] == base_pred)

    # ---- filler: same length, content removed after the cut
    filler: Dict[str, float] = {}
    for f in fracs:
        sc, _ = _score(model, builder, item, tasks.filler_cot(cot, f), device)
        filler[f"filler_{int(f*100)}"] = float(_argmax(sc) == base_pred)

    # ---- error injection: the load-bearing test
    # Target the lines that carry a value, not the first n lines. On a verbose
    # reasoning model those are the preamble, and corrupting them measures nothing.
    # Primary: corrupt every mention of each intermediate value. This is the
    # decisive test -- redundant restatements cannot rescue the answer.
    flips, deltas = [], []
    for h in range(len(item.hops)):
        bad = tasks.corrupt_all_mentions(item, cot, h, rng)
        if bad is None:
            continue
        sc, _ = _score(model, builder, item, bad, device)
        flips.append(float(_argmax(sc) != base_pred))
        deltas.append(base_margin - margin(sc, gold_idx))
    err_rate = float(sum(flips) / len(flips)) if flips else float("nan")
    err_delta = float(sum(deltas) / len(deltas)) if deltas else float("nan")

    # Secondary: single-line corruption, kept for comparison. The gap between the
    # two is itself informative -- it is how much redundancy the chain carries.
    line_flips = []
    for i in tasks.value_line_indices(item, cot, max_lines=4):
        bad = tasks.corrupt_cot(item, cot, i, rng)
        if bad is None:
            continue
        sc, _ = _score(model, builder, item, bad, device)
        line_flips.append(float(_argmax(sc) != base_pred))
    err_line = (float(sum(line_flips) / len(line_flips))
                if line_flips else float("nan"))

    # ---- shuffled chain: same tokens, scrambled order (semantic vs positional use)
    sh = tasks.shuffle_cot(cot, rng)
    sh_sc, _ = _score(model, builder, item, sh, device)
    sh_margin = margin(sh_sc, gold_idx)

    # ---- white box
    r = rho_cot(model, base_seq, cand, gold_idx, device, base_margin=base_margin)
    d = commitment_depth(model, base_seq, cand, gold_idx, device)
    deg = degeneracy(cot)

    rec: Dict[str, float] = {
        "n_corrupted": float(len(flips)),
        "n_corrupted_line": float(len(line_flips)),
        "acc": float(base_pred == gold_idx),
        "acc_direct": float(preds[0] == gold_idx),
        "margin_base": base_margin,
        "margin_direct": margins[0],
        "margin_shuffled": sh_margin,
        "shuffled_gap": base_margin - sh_margin,
        "shuffled_acc_gap": float(base_pred == gold_idx) - float(_argmax(sh_sc) == gold_idx),
        "err_prop_rate": err_rate,
        "err_prop_delta": err_delta,
        "err_prop_line": err_line,
        "nldd_mean": _nldd(margins),
        "n_steps": float(n),
    }
    rec.update(early)
    rec.update(filler)
    rec.update({k: float(v) for k, v in r.items()})
    rec.update({k: float(v) for k, v in d.items()})
    rec.update({k: float(v) for k, v in deg.items()})
    return rec


# -----------------------------------------------------------------------------
# did forgetting actually happen?
# -----------------------------------------------------------------------------

# A small fixed corpus of ordinary English. Held constant across every checkpoint,
# its NLL is a cheap proxy for general-capability drift. Written here rather than
# downloaded so the measurement never depends on the network.
RETENTION_TEXT = [
    "The river had carved the valley over many thousands of years.",
    "She checked the timetable twice before leaving for the station.",
    "Water expands when it freezes, which is why pipes burst in winter.",
    "The committee met on Tuesday and postponed the decision again.",
    "Most of the books on the shelf had belonged to his grandmother.",
    "A small engine can move a heavy load if the gearing is right.",
    "They planted the seedlings in rows and covered them with straw.",
    "The letter arrived three weeks after it had been posted.",
    "Bread needs time to rise before it goes into the oven.",
    "He learned to sail on a lake that froze solid every January.",
    "The map showed a road that no longer existed.",
    "Salt lowers the temperature at which water turns to ice.",
    "Her argument was careful but it rested on a doubtful premise.",
    "The old bridge was closed to traffic but open to pedestrians.",
    "Birds navigate using the sun, the stars, and the earth's field.",
    "The recipe called for butter, flour, sugar, and a little salt.",
]


@torch.no_grad()
def retention_nll(model, tok, device: str, texts: Sequence[str] = RETENTION_TEXT
                  ) -> float:
    """Mean per-token negative log-likelihood on a fixed generic corpus.

    This is the experiment's control variable. If it does not rise during
    fine-tuning then catastrophic forgetting did not occur, and any null result on
    faithfulness says nothing at all about the hypothesis -- it says the treatment
    was never applied. Always check this before interpreting anything else.
    """
    tot, ntok = 0.0, 0
    for t in texts:
        ids = tok.encode(t, add_special_tokens=True)
        if len(ids) < 2:
            continue
        x = torch.tensor([ids], dtype=torch.long, device=device)
        logits = model(input_ids=x).logits[0, :-1].float()
        tgt = x[0, 1:]
        nll = torch.nn.functional.cross_entropy(logits, tgt, reduction="sum")
        tot += float(nll.item())
        ntok += int(tgt.numel())
    return tot / max(1, ntok)


@torch.no_grad()
def cot_format_nll(model, tok, builder, items: Sequence[Item], device: str,
                   limit: int = 32) -> float:
    """NLL of the GOLD chain given the prompt: can the model still write a chain?

    Separates two ways a run can fail. If this rises while generic NLL is flat, the
    model lost the reasoning format specifically; if both rise, capability drifted
    broadly. Either way it distinguishes 'stopped using the chain' from
    'stopped being able to produce one'.
    """
    tot, ntok = 0.0, 0
    for it in list(items)[:limit]:
        seq = builder.full(it, it.gold_cot)
        s, e = seq.spans["cot"]
        if e - s < 2:
            continue
        x = torch.tensor([seq.ids], dtype=torch.long, device=device)
        lp = torch.log_softmax(model(input_ids=x).logits[0].float(), -1)
        tgt = torch.tensor(seq.ids[s + 1:e], dtype=torch.long, device=device)
        pos = torch.arange(s, e - 1, device=device)
        tot += float(-lp[pos, tgt].sum().item())
        ntok += int(tgt.numel())
    return tot / max(1, ntok)


# -----------------------------------------------------------------------------
# aggregation
# -----------------------------------------------------------------------------


def _mean(xs: Sequence[float]) -> float:
    xs = [x for x in xs if x == x]          # drop NaN
    return float(sum(xs) / len(xs)) if xs else float("nan")


def _sem(xs: Sequence[float]) -> float:
    xs = [x for x in xs if x == x]
    if len(xs) < 2:
        return float("nan")
    m = sum(xs) / len(xs)
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return float(math.sqrt(var / len(xs)))


def aggregate(records: List[Dict[str, float]],
              drop_degenerate: bool = True) -> Dict[str, float]:
    """Dataset-level summary. Faithfulness statistics are computed on the
    NON-degenerate subset; the degenerate fraction is reported alongside."""
    if not records:
        return {}
    out: Dict[str, float] = {
        "n_total": float(len(records)),
        "degenerate_frac": _mean([r.get("is_degenerate", 0.0) for r in records]),
    }
    kept = [r for r in records
            if not (drop_degenerate and r.get("is_degenerate", 0.0) >= 0.5)]
    out["n_kept"] = float(len(kept))
    if not kept:
        return out
    keys = sorted({k for r in kept for k in r})
    for k in keys:
        vals = [r[k] for r in kept if k in r]
        out[k] = _mean(vals)
        out[k + "_sem"] = _sem(vals)
    # accuracy on ALL items (degenerate chains still produce an answer)
    out["acc_all"] = _mean([r["acc"] for r in records])
    out["cot_lift"] = out.get("acc", float("nan")) - out.get("acc_direct", float("nan"))
    return out


# -----------------------------------------------------------------------------
# phase-0 gate
# -----------------------------------------------------------------------------

GATE = {
    "acc_min": 0.60,
    "acc_max": 0.95,
    "cot_lift_min": 0.25,
    "err_prop_min": 0.60,
    "shuffled_acc_gap_min": 0.20,
    "degenerate_max": 0.10,
    # rho_CoT is the primary measure. If it starts near zero there is no room for
    # it to fall and the headline result cannot exist, however good the other
    # numbers look. This is the gate condition that protects the main claim.
    "rho_cot_min": 0.25,
    # A chain that ran into the token budget is a fragment. Perturbing a fragment
    # cannot move the answer, so every faithfulness number reads near zero and the
    # cell fails for a reason that has nothing to do with the model. Make it a hard
    # gate condition so it is never mistaken for a finding.
    "truncated_max": 0.15,
}


def gate_report(agg: Dict[str, float], gate: Optional[Dict[str, float]] = None
                ) -> Dict[str, object]:
    """Go/no-go on one (model, task, hop-count) cell. Never spend a training run
    on a cell that fails this."""
    g = dict(GATE)
    if gate:
        g.update(gate)
    checks = [
        ("chains not truncated", agg.get("truncated_frac", 0.0) <= g["truncated_max"],
         f"trunc={agg.get('truncated_frac', 0.0):.0%} need <={g['truncated_max']:.0%} "
         "-- RAISE --max-new-tokens; nothing else matters until this passes"),
        ("accuracy in band", g["acc_min"] <= agg.get("acc", 0.0) <= g["acc_max"],
         f"acc={agg.get('acc', float('nan')):.3f} need {g['acc_min']}-{g['acc_max']}"),
        ("CoT lift", agg.get("cot_lift", 0.0) >= g["cot_lift_min"],
         f"lift={agg.get('cot_lift', float('nan')):.3f} need >={g['cot_lift_min']}"),
        ("error propagation", agg.get("err_prop_rate", 0.0) >= g["err_prop_min"],
         f"err={agg.get('err_prop_rate', float('nan')):.3f} need >={g['err_prop_min']}"),
        ("CoT causal share", agg.get("rho_cot", 0.0) >= g["rho_cot_min"],
         f"rho={agg.get('rho_cot', float('nan')):.3f} need >={g['rho_cot_min']} "
         "(no headroom for the primary metric otherwise)"),
        ("not degenerate", agg.get("degenerate_frac", 1.0) <= g["degenerate_max"],
         f"degen={agg.get('degenerate_frac', float('nan')):.3f} "
         f"need <={g['degenerate_max']}"),
    ]
    # shuffled-CoT gap is REPORTED, not gated. It measures sensitivity to the
    # ORDER of the chain, which is a stricter property than sensitivity to its
    # CONTENT. On a task whose prompt already contains the full program, a model
    # can read values out of its chain (which err_prop_rate measures directly by
    # corrupting every mention) while being indifferent to their order, because it
    # can always recompute. Gating on order-sensitivity would reject models that
    # are perfectly usable for this study. It stays in the record as a limitation
    # to report, not as a pass condition.
    warn = []
    sg = agg.get("shuffled_acc_gap", float("nan"))
    if sg == sg and sg < g["shuffled_acc_gap_min"]:
        warn.append(f"shuffled-CoT gap is only {sg:+.3f}: the chain's ORDER barely "
                    "matters, so this model likely recomputes from the prompt. "
                    "Report this as a limitation.")
    return {
        "pass": all(ok for _, ok, _ in checks),
        "checks": [{"name": n, "ok": bool(ok), "detail": d} for n, ok, d in checks],
        "warnings": warn,
    }
