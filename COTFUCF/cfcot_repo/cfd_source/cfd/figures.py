"""Publication figures. Black, red and grey only; no gridlines, no gradients."""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence

BLACK, RED, GREY, PALE = "#000000", "#CC0000", "#7F7F7F", "#C8C8C8"

STYLE = {
    "figure.dpi": 140, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "serif", "font.size": 9,
    "axes.linewidth": 0.8, "axes.grid": False, "axes.labelsize": 9,
    "axes.titlesize": 9, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.3, "lines.markersize": 3.2,
    "legend.frameon": False, "legend.fontsize": 8,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.labelsize": 8, "ytick.labelsize": 8,
}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update(STYLE)
    return plt


def _series(rows: List[Dict], key: str):
    rows = sorted(rows, key=lambda r: r["step"])
    return ([r["step"] for r in rows],
            [r.get(key, float("nan")) for r in rows],
            [r.get(key + "_sem", float("nan")) for r in rows])


def _rel(vals: Sequence[float]) -> List[float]:
    v0 = next((v for v in vals if v == v), float("nan"))
    if v0 != v0 or v0 <= 1e-9:
        return [float("nan")] * len(vals)
    return [v / v0 for v in vals]


# -----------------------------------------------------------------------------


def fig_silent_window(runs: Dict[str, List[Dict]], path: str, delta: float = 0.20):
    """Faithfulness and accuracy against training step, one panel per condition."""
    plt = _plt()
    names = list(runs)
    fig, ax = plt.subplots(1, len(names), figsize=(2.7 * len(names), 2.5),
                           squeeze=False, sharey=True)
    for k, name in enumerate(names):
        a = ax[0][k]
        st, rho, _ = _series(runs[name], "rho_cot")
        _, acc, _ = _series(runs[name], "acc")
        a.plot(st, _rel(rho), "-", color=BLACK, marker="o", label="CoT causal share")
        a.plot(st, _rel(acc), "--", color=RED, marker="s", label="accuracy")
        a.axhline(1 - delta, color=GREY, lw=0.6, ls=":")
        a.set_title(name.replace("_", " "))
        a.set_xlabel("training step")
        a.set_ylim(0, 1.15)
    ax[0][0].set_ylabel("value relative to step 0")
    ax[0][0].legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_fadg(stats: Dict[str, Dict[str, float]], path: str):
    """The decoupling gap by condition. Positive means faithfulness fell first."""
    plt = _plt()
    names = [n for n in stats]
    vals = [stats[n].get("fadg", float("nan")) for n in names]
    fig, a = plt.subplots(figsize=(1.1 * max(3, len(names)) + 1, 2.4))
    cols = [BLACK if (v == v and v > 0) else GREY for v in vals]
    plotted = [0 if v != v else v for v in vals]
    a.bar(range(len(names)), plotted, color=cols, width=0.55)
    for i, n in enumerate(names):
        v = vals[i]
        if v != v:
            a.text(i, 0, " n/a", ha="center", va="bottom", fontsize=7, color=GREY)
        elif stats[n].get("fadg_censored"):
            a.text(i, v, "\u2265", ha="center",
                   va="bottom" if v >= 0 else "top", fontsize=9, color=BLACK)
    a.axhline(0, color=BLACK, lw=0.8)
    a.set_xticks(range(len(names)))
    a.set_xticklabels([n.replace("_", " ") for n in names], rotation=15)
    a.set_ylabel("FADG (steps)")
    a.set_title("Steps between the faithfulness drop and the accuracy drop")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_battery(rows: List[Dict], path: str):
    """Every faithfulness measure over training, on one grid."""
    plt = _plt()
    panels = [
        ("rho_cot", "CoT causal share"),
        ("err_prop_rate", "error propagation rate"),
        ("shuffled_acc_gap", "shuffled-chain accuracy gap"),
        ("early_50", "answer already fixed at 50% of chain"),
        ("nldd_mean", "margin accrued late"),
        ("cot_lift", "accuracy gain from the chain"),
    ]
    fig, ax = plt.subplots(2, 3, figsize=(8.0, 4.2), sharex=True)
    for i, (key, title) in enumerate(panels):
        a = ax[i // 3][i % 3]
        st, v, sem = _series(rows, key)
        a.plot(st, v, "-", color=BLACK, marker="o")
        if any(e == e for e in sem):
            lo = [x - e if (x == x and e == e) else float("nan") for x, e in zip(v, sem)]
            hi = [x + e if (x == x and e == e) else float("nan") for x, e in zip(v, sem)]
            a.fill_between(st, lo, hi, color=PALE, linewidth=0)
        a.set_title(title)
        if i >= 3:
            a.set_xlabel("training step")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_commitment(runs: Dict[str, List[Dict]], path: str):
    """Depth at which the answer stabilises. Falling means it is fixed earlier."""
    plt = _plt()
    fig, a = plt.subplots(figsize=(3.4, 2.5))
    styles = ["-", "--", "-.", ":"]
    for i, (name, rows) in enumerate(runs.items()):
        st, v, _ = _series(rows, "d_star_frac")
        a.plot(st, v, styles[i % 4], color=BLACK if i == 0 else RED if i == 1 else GREY,
               marker="o", label=name.replace("_", " "))
    a.set_xlabel("training step")
    a.set_ylabel("commitment depth (fraction of layers)")
    a.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_forgetting_coupling(runs: Dict[str, List[Dict]], path: str):
    """Does the faithfulness loss track how much was forgotten?"""
    plt = _plt()
    fig, a = plt.subplots(figsize=(3.4, 2.7))
    marks = ["o", "s", "^", "D", "v"]
    cols = [BLACK, RED, GREY, BLACK, RED]
    for i, (name, rows) in enumerate(runs.items()):
        rows = sorted(rows, key=lambda r: r["step"])
        n0 = rows[0].get("nll_generic", float("nan"))
        r0 = rows[0].get("rho_cot", float("nan"))
        if n0 != n0 or r0 != r0 or r0 <= 0:
            continue
        x = [r.get("nll_generic", float("nan")) - n0 for r in rows]
        y = [1 - r.get("rho_cot", float("nan")) / r0 for r in rows]
        a.plot(x, y, marks[i % 5], color=cols[i % 5], markersize=3.2, linestyle="none",
               markerfacecolor="none" if i else cols[0], label=name.replace("_", " "))
    a.axhline(0, color=GREY, lw=0.6)
    a.axvline(0, color=GREY, lw=0.6)
    a.set_xlabel("forgetting (rise in generic NLL)")
    a.set_ylabel("relative loss of CoT causal share")
    a.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_dissociation(dd: Dict[str, Dict[str, float]], path: str):
    """Ablation effects on the two axes."""
    plt = _plt()
    order = [o for o in ("control", "ablate_gen", "ablate_med", "ablate_random")
             if o in dd]
    lab = {"control": "none", "ablate_gen": "generators",
           "ablate_med": "mediators", "ablate_random": "random"}
    fig, ax = plt.subplots(1, 2, figsize=(6.2, 2.5))
    for a, key, ttl in ((ax[0], "faithfulness", "faithfulness"),
                        (ax[1], "fluency", "chain fluency (log-prob)")):
        v = [dd[o][key] for o in order]
        cols = [GREY, BLACK, RED, PALE][: len(order)]
        a.bar(range(len(order)), v, color=cols, width=0.55)
        a.axhline(dd["control"][key], color=BLACK, lw=0.6, ls=":")
        a.set_xticks(range(len(order)))
        a.set_xticklabels([lab[o] for o in order], rotation=15)
        a.set_title(ttl)
        a.set_xlabel("ablated group")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_dvr(dvr: Dict[str, float], path: str):
    """How far fine-tuning moved each latent group."""
    plt = _plt()
    keys = [("drift_med", "mediators"), ("drift_gen", "generators"),
            ("drift_random", "random"), ("drift_all", "all latents")]
    v = [dvr.get(k, float("nan")) for k, _ in keys]
    fig, a = plt.subplots(figsize=(3.2, 2.4))
    a.bar(range(len(v)), v, color=[RED, BLACK, GREY, PALE], width=0.55)
    a.set_xticks(range(len(v)))
    a.set_xticklabels([n for _, n in keys], rotation=15)
    a.set_ylabel("relative activation drift")
    d = dvr.get("DVR", float("nan"))
    a.set_title(f"DVR = {d:.2f}" if d == d else "DVR = n/a")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# -----------------------------------------------------------------------------


def rho_drop(rows: List[Dict], key: str = "rho_cot") -> float:
    """Relative fall in a metric from the first checkpoint to the last."""
    rows = sorted(rows, key=lambda r: r["step"])
    v = [r.get(key, float("nan")) for r in rows]
    first = next((x for x in v if x == x), float("nan"))
    last = next((x for x in reversed(v) if x == x), float("nan"))
    if first != first or last != last or first <= 1e-9:
        return float("nan")
    return 1.0 - last / first


def contrast(runs: Dict[str, List[Dict]], treat: str = "ood_cot",
             controls: Sequence[str] = ("replay", "replay_generic")
             ) -> Dict[str, float]:
    """The headline number: how much more the CoT causal share fell under
    forgetting than under each control that suppresses it.

    contrast = drop(treatment) - drop(control).  Everything else in the study is
    instrumentation for this one comparison, so it gets its own line.
    """
    out: Dict[str, float] = {}
    if treat not in runs:
        return out
    dt = rho_drop(runs[treat])
    out[f"drop_{treat}"] = dt
    for c in controls:
        if c in runs:
            dc = rho_drop(runs[c])
            out[f"drop_{c}"] = dc
            out[f"CONTRAST_{treat}_vs_{c}"] = dt - dc
    return out


def write_tables(runs: Dict[str, List[Dict]], stats: Dict[str, Dict[str, float]],
                 out_dir: str, extra: Optional[Dict] = None) -> None:
    """A CSV of every checkpoint and a short markdown summary for the paper."""
    keys = ["step", "acc", "acc_direct", "cot_lift", "rho_cot", "d_star_frac",
            "err_prop_rate", "shuffled_acc_gap", "early_25", "early_50", "early_75",
            "nldd_mean", "err_prop_line", "n_corrupted", "truncated_frac",
            "margin_base", "margin_no_cot", "margin_no_rules",
            "delta_cot", "delta_rules", "nll_generic", "nll_gold_cot",
            "degenerate_frac", "n_kept", "n_steps", "n_tokens"]
    with open(os.path.join(out_dir, "results.csv"), "w", encoding="utf-8") as fh:
        fh.write("condition," + ",".join(keys) + "\n")
        for name, rows in runs.items():
            for r in sorted(rows, key=lambda x: x["step"]):
                fh.write(name + "," + ",".join(
                    ("" if r.get(k, float("nan")) != r.get(k, float("nan"))
                     else f"{r.get(k, '')}") for k in keys) + "\n")

    con = contrast(runs)
    lines = ["# Results", ""]
    if con:
        lines += ["## Headline", ""]
        for k, v in con.items():
            mark = ""
            if k.startswith("CONTRAST"):
                mark = "   <-- PASS" if (v == v and v >= 0.20) else "   <-- below 0.20"
            lines.append(f"- {k}: " + ("n/a" if v != v else f"{v:+.3f}") + mark)
        lines.append("")
    lines += ["## Decoupling by condition", "",
             "| condition | rho drop | acc drop | faithfulness crossing |"
             " accuracy crossing | FADG | note | forgetting (NLL drift) |",
             "|---|---|---|---|---|---|---|---|"]

    def f(x):
        return "n/a" if x != x else f"{x:.0f}"

    for name, rows in runs.items():
        s = stats.get(name, {})
        rows = sorted(rows, key=lambda r: r["step"])
        n0 = rows[0].get("nll_generic", float("nan"))
        n1 = rows[-1].get("nll_generic", float("nan"))
        drift = n1 - n0 if (n0 == n0 and n1 == n1) else float("nan")
        dr, da = rho_drop(rows), rho_drop(rows, "acc")
        g = lambda x: "n/a" if x != x else f"{x:+.1%}"        # noqa: E731
        lines.append(
            f"| {name} | {g(dr)} | {g(da)} | {f(s.get('step_faith', float('nan')))} | "
            f"{f(s.get('step_acc', float('nan')))} | {f(s.get('fadg', float('nan')))} | "
            f"{s.get('fadg_censored') or '-'} | "
            + ("n/a" if drift != drift else f"{drift:+.3f}") + " |")

    if extra:
        if extra.get("dvr"):
            lines += ["", "## Differential vulnerability", ""]
            lines += [f"- {k}: {v:.4f}" if isinstance(v, float) else f"- {k}: {v}"
                      for k, v in extra["dvr"].items()]
        if extra.get("test"):
            lines += ["", "## Dissociation test", ""]
            lines += [f"- {k}: {v}" for k, v in extra["test"].items()]
        if extra.get("warnings"):
            lines += ["", "## Warnings", ""] + [f"- {w}" for w in extra["warnings"]]

    with open(os.path.join(out_dir, "results.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def build_all(root: str, delta: float = 0.20) -> Dict[str, Dict[str, float]]:
    """Regenerate every figure and table from the jsonl files already on disk."""
    import glob
    runs: Dict[str, List[Dict]] = {}
    for p in sorted(glob.glob(os.path.join(root, "*", "sweep.jsonl"))):
        name = os.path.basename(os.path.dirname(p))
        rows = []
        with open(p, encoding="utf-8") as fh:
            for ln in fh:
                ln = ln.strip()
                if ln:
                    rows.append(json.loads(ln))
        if rows:
            runs[name] = rows
    if not runs:
        raise SystemExit(f"no sweep.jsonl found under {root}")

    # `fadg` lives in rig. In the single-file build it is already in this
    # namespace; in the package build it has to be imported.
    _fadg = globals().get("fadg")
    if _fadg is None:
        from cfd.rig import fadg as _fadg

    stats = {n: _fadg([r["step"] for r in sorted(rows, key=lambda x: x["step"])],
                     [r.get("rho_cot", float("nan"))
                      for r in sorted(rows, key=lambda x: x["step"])],
                     [r.get("acc", float("nan"))
                      for r in sorted(rows, key=lambda x: x["step"])], delta)
             for n, rows in runs.items()}

    fig_silent_window(runs, os.path.join(root, "fig1_silent_window.png"), delta)
    fig_fadg(stats, os.path.join(root, "fig2_fadg_by_condition.png"))
    fig_commitment(runs, os.path.join(root, "fig4_commitment_depth.png"))
    fig_forgetting_coupling(runs, os.path.join(root, "fig5_forgetting_coupling.png"))
    primary = "ood_cot" if "ood_cot" in runs else list(runs)[0]
    fig_battery(runs[primary], os.path.join(root, "fig6_battery.png"))

    extra = {}
    f2 = os.path.join(root, primary, "figure2.json")
    if os.path.exists(f2):
        extra = json.load(open(f2, encoding="utf-8"))
        fig_dissociation(extra["dd"], os.path.join(root, "fig3_ablation.png"))
        if extra.get("dvr"):
            fig_dvr(extra["dvr"], os.path.join(root, "fig7_dvr.png"))
    write_tables(runs, stats, root, extra)
    return stats
