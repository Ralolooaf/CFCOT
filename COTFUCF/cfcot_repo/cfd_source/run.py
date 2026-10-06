#!/usr/bin/env python
"""
Single entry point for the whole pipeline.

    python selftest.py                       # ALWAYS run this first, ~20 s, no GPU
    python run.py gate    --model M          # phase-0 go/no-go   (~4-6 GPU-h total)
    python run.py train   --model M --cond ood_cot
    python run.py sweep   --model M --run runs/ood_cot     # -> Figure 1
    python run.py fig2    --model M --run runs/ood_cot     # -> Figure 2

Every command is safe to interrupt and re-run: training resumes from the last
checkpoint and the sweep skips checkpoints it has already evaluated.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Dict, List

import torch

from cfd import tasks
from cfd.engine import Builder, generate_cots
from cfd.metrics import evaluate_item, aggregate, gate_report
from cfd.rig import (load_model, write_jsonl, read_jsonl, figure1, figure2,
                     dissociation_test, d_model_of, n_layers_of)
from cfd import train as TR


def _mk_items(task: str, n: int, seed: int, hops: int, distractors: int,
              modulus: int) -> List[tasks.Item]:
    kw = ({"hops": hops, "distractors": distractors} if task == "chain"
          else {"hops": hops, "modulus": modulus})
    return tasks.build_dataset(task, n, seed=seed, **kw)


def _eval_set(model, tok, builder, items, device, max_new_tokens, gen_bs, seed):
    cots = generate_cots(model, tok, items, builder, device,
                         max_new_tokens=max_new_tokens, batch_size=gen_bs)
    recs = [evaluate_item(model, tok, builder, it, c, device, seed=seed)
            for it, c in zip(items, cots)]
    return aggregate(recs), recs, cots


# -----------------------------------------------------------------------------


def cmd_gate(a) -> int:
    model, tok, dev = load_model(a.model, a.device)
    builder = Builder(tok)
    rows, any_pass = [], False
    for task in a.tasks.split(","):
        for hops in [int(x) for x in a.hops.split(",")]:
            for dis in [int(x) for x in a.distractors.split(",")]:
                if task == "modarith" and dis != 0:
                    continue
                items = _mk_items(task, a.n, a.seed, hops, dis, a.modulus)
                agg, _, _ = _eval_set(model, tok, builder, items, dev,
                                      a.max_new_tokens, a.gen_batch, a.seed)
                rep = gate_report(agg)
                any_pass |= rep["pass"]
                rows.append({"task": task, "hops": hops, "distractors": dis,
                             **{k: v for k, v in agg.items()
                                if not k.endswith("_sem")},
                             "gate_pass": rep["pass"]})
                flag = "PASS" if rep["pass"] else "fail"
                print(f"  [{flag}] {task} hops={hops} dis={dis}  "
                      f"acc={agg.get('acc', float('nan')):.3f} "
                      f"lift={agg.get('cot_lift', float('nan')):+.3f} "
                      f"err={agg.get('err_prop_rate', float('nan')):.3f} "
                      f"shuf={agg.get('shuffled_acc_gap', float('nan')):+.3f} "
                      f"degen={agg.get('degenerate_frac', float('nan')):.2f}")
                if not rep["pass"]:
                    for c in rep["checks"]:
                        if not c["ok"]:
                            print(f"          - {c['name']}: {c['detail']}")
    write_jsonl(rows, os.path.join(a.out, "gate.jsonl"))
    print(f"\nwrote {a.out}/gate.jsonl")
    if not any_pass:
        print("\nNO CELL PASSED. Do not start training. Raise hops/distractors, "
              "or move up a model tier. This is the cheapest possible failure.")
        return 2
    best = max((r for r in rows if r["gate_pass"]),
               key=lambda r: r.get("shuffled_acc_gap", 0.0))
    print(f"\nbest cell: task={best['task']} hops={best['hops']} "
          f"distractors={best['distractors']}  (max shuffled-CoT gap)")
    return 0


def cmd_train(a) -> int:
    model, tok, dev = load_model(a.model, a.device, verify=False)
    reason = _domain_for(a.cond, a.n_train, a.seed, a.domain_jsonl)
    data, note = reason
    print(f"[data  ] {a.cond}: {len(data)} examples ({note})")
    cfg = TR.TrainCfg(steps=a.steps, ckpt_every=a.ckpt_every,
                      batch_size=a.batch_size, grad_accum=a.grad_accum,
                      lr=a.lr, seed=a.seed, r=a.rank, alpha=2 * a.rank)
    out = a.run or os.path.join("runs", a.cond)
    TR.train(model, tok, data, out, cfg, dev, max_seconds=a.max_seconds)
    print(f"[done  ] checkpoints in {out}")
    return 0


def _domain_for(cond: str, n: int, seed: int, jsonl: str):
    if jsonl:
        base = TR.load_jsonl(jsonl)
        note = f"from {jsonl}"
    else:
        base = TR.synthetic_domain(n, seed=seed,
                                   answer_only=not cond.endswith("_cot"))
        note = "synthetic clinical domain"
    if cond == "random_label":
        return TR.randomise_labels(base, seed=seed), note + ", labels shuffled"
    if cond == "replay":
        items = tasks.build_dataset("chain", n // 4, seed=seed + 99, hops=3)
        rep = [TR.Example(f"{i.rules_text}\n\n{i.question_text}\nAnswer:",
                          f" {i.gold}") for i in items]
        mixed = base + rep
        random.Random(seed).shuffle(mixed)
        return mixed, note + f", +{len(rep)} replay"
    return base, note


def cmd_sweep(a) -> int:
    model, tok, dev = load_model(a.model, a.device)
    builder = Builder(tok)
    items = _mk_items(a.task, a.n, a.seed + 1000, a.hops_one, a.dis_one, a.modulus)
    wrapped = TR.inject_lora(model, a.rank, 2 * a.rank, reset=True)
    out_path = os.path.join(a.run, "sweep.jsonl")
    done = {r["step"] for r in read_jsonl(out_path)} if os.path.exists(out_path) else set()
    rows = read_jsonl(out_path) if os.path.exists(out_path) else []
    for step, path in TR.list_checkpoints(a.run):
        if step in done:
            continue
        st = torch.load(path, map_location="cpu", weights_only=False)
        TR.load_lora_state_dict(wrapped, st["lora"])
        model.eval()
        agg, _, _ = _eval_set(model, tok, builder, items, dev,
                              a.max_new_tokens, a.gen_batch, a.seed)
        rows.append({"step": step, **agg})
        write_jsonl(sorted(rows, key=lambda r: r["step"]), out_path)   # crash-safe
        print(f"  step {step:5d}  acc={agg.get('acc', float('nan')):.3f}  "
              f"rho={agg.get('rho_cot', float('nan')):.3f}  "
              f"err={agg.get('err_prop_rate', float('nan')):.3f}  "
              f"d*={agg.get('d_star_frac', float('nan')):.3f}")
    stats = figure1({os.path.basename(a.run.rstrip('/')): rows},
                    os.path.join(a.run, "figure1.png"))
    print(json.dumps(stats, indent=2))
    print(f"wrote {a.run}/figure1.png")
    return 0


def cmd_fig2(a) -> int:
    from cfd.analysis import (collect_activations, train_sae, splice_is_identity,
                              partition_latents, double_dissociation)
    model, tok, dev = load_model(a.model, a.device)
    builder = Builder(tok)
    layer = a.layer if a.layer >= 0 else int(0.7 * n_layers_of(model))
    print(f"[layer ] {layer} of {n_layers_of(model)}  (d_model={d_model_of(model)})")

    items = _mk_items(a.task, a.n, a.seed + 7, a.hops_one, a.dis_one, a.modulus)
    cots = generate_cots(model, tok, items, builder, dev,
                         max_new_tokens=a.max_new_tokens, batch_size=a.gen_batch)
    seqs = [builder.full(i, c) for i, c in zip(items, cots)]

    acts = collect_activations(model, seqs, layer, dev, max_tokens=a.sae_tokens)
    print(f"[sae   ] training on {acts.shape[0]} activations")
    sae, st = train_sae(acts, n_latents=a.n_latents, k=a.topk, steps=a.sae_steps,
                        device=dev, seed=a.seed)
    print("[sae   ]", {k: round(v, 4) for k, v in st.items()})
    ident = max(splice_is_identity(model, sae, layer, s, dev) for s in seqs[:8])
    print(f"[check ] splice identity max|dlogit| = {ident:.2e}")
    if ident > 1e-2:
        print("ABORT: the splice is not an identity; every downstream number "
              "would be confounded by reconstruction error.")
        return 2

    dom = [builder.full(i, c) for i, c in zip(items[len(items)//2:],
                                              cots[len(items)//2:])]
    part = partition_latents(model, sae, layer, builder, items[:a.n_partition],
                             cots[:a.n_partition], dev, dom, top_k=a.top_k_latents)
    print("[part  ]", {k: round(v, 4) for k, v in part.summary().items()})
    for w in part.check():
        print(f"  WARNING: {w}")

    dd = double_dissociation(model, sae, layer, builder, items[:a.n_dd],
                             cots[:a.n_dd], part, dev, seed=a.seed)
    for k, v in dd.items():
        print(f"  {k:14s} faith={v['faithfulness']:+.4f} "
              f"flu={v['fluency']:+.4f} acc={v['accuracy']:.3f}")
    test = dissociation_test(dd)
    print("[result]", json.dumps(test, indent=2, default=float))
    figure2(dd, os.path.join(a.run, "figure2.png"))
    with open(os.path.join(a.run, "figure2.json"), "w") as fh:
        json.dump({"dd": dd, "test": test, "partition": part.summary(),
                   "sae": st, "layer": layer}, fh, indent=2, default=float)
    torch.save({"sae": sae.state_dict(), "med": part.med, "gen": part.gen,
                "dom": part.dom, "layer": layer},
               os.path.join(a.run, "sae_partition.pt"))
    print(f"wrote {a.run}/figure2.png")
    return 0


# -----------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(q):
        q.add_argument("--model", required=True)
        q.add_argument("--device", default="cuda")
        q.add_argument("--seed", type=int, default=0)
        q.add_argument("--n", type=int, default=200)
        q.add_argument("--max-new-tokens", type=int, default=160)
        q.add_argument("--gen-batch", type=int, default=8)
        q.add_argument("--modulus", type=int, default=11)

    g = sub.add_parser("gate"); common(g)
    g.add_argument("--tasks", default="chain,modarith")
    g.add_argument("--hops", default="2,3,4,5")
    g.add_argument("--distractors", default="0,3,6")
    g.add_argument("--out", default="runs/gate")
    g.set_defaults(fn=cmd_gate)

    t = sub.add_parser("train"); common(t)
    t.add_argument("--cond", default="ood_answer",
                   choices=["ood_answer", "ood_cot", "replay", "random_label"])
    t.add_argument("--run", default="")
    t.add_argument("--domain-jsonl", default="")
    t.add_argument("--n-train", type=int, default=4000)
    t.add_argument("--steps", type=int, default=600)
    t.add_argument("--ckpt-every", type=int, default=25)
    t.add_argument("--batch-size", type=int, default=4)
    t.add_argument("--grad-accum", type=int, default=2)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--rank", type=int, default=16)
    t.add_argument("--max-seconds", type=float, default=None)
    t.set_defaults(fn=cmd_train)

    s = sub.add_parser("sweep"); common(s)
    s.add_argument("--run", required=True)
    s.add_argument("--task", default="chain")
    s.add_argument("--hops-one", type=int, default=3)
    s.add_argument("--dis-one", type=int, default=3)
    s.add_argument("--rank", type=int, default=16)
    s.set_defaults(fn=cmd_sweep)

    f = sub.add_parser("fig2"); common(f)
    f.add_argument("--run", required=True)
    f.add_argument("--task", default="chain")
    f.add_argument("--hops-one", type=int, default=3)
    f.add_argument("--dis-one", type=int, default=3)
    f.add_argument("--layer", type=int, default=-1)
    f.add_argument("--n-latents", type=int, default=8192)
    f.add_argument("--topk", type=int, default=32)
    f.add_argument("--sae-steps", type=int, default=3000)
    f.add_argument("--sae-tokens", type=int, default=400000)
    f.add_argument("--top-k-latents", type=int, default=64)
    f.add_argument("--n-partition", type=int, default=48)
    f.add_argument("--n-dd", type=int, default=64)
    f.set_defaults(fn=cmd_fig2)

    a = p.parse_args()
    os.makedirs(getattr(a, "run", None) or getattr(a, "out", "runs"), exist_ok=True)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
