# cfd — catastrophic forgetting × CoT faithfulness

Tested code for Figure 1 (the silent window) and Figure 2 (the double dissociation).

## Before you spend a single GPU-hour

```bash
python selftest.py          # ~20 s, CPU only, no downloads
```

21 checks. It builds a tiny random model and a stub tokeniser and verifies the
things that would otherwise waste a night: that spans tile the sequence exactly,
that batched scoring equals naive scoring, that batch size does not change greedy
generation, that resuming is bit-identical to an uninterrupted run, that the SAE
splice is a numerical identity, and that 4-D attention masks are not ignored by
your transformers version. **If it is not green, nothing downstream is meaningful.**

Run it again after every edit to `engine.py` or `analysis.py`.

## The four commands

```bash
# 0. go / no-go. ~4-6 GPU-h. Never skip.
python run.py gate  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
                    --n 200 --hops 2,3,4,5 --distractors 0,3,6

# 1. fine-tune with dense checkpoints (resumable)
python run.py train --model <M> --cond ood_answer --run runs/ood_answer \
                    --steps 600 --ckpt-every 25 --max-seconds 12000

# 2. evaluate every checkpoint  -> figure1.png
python run.py sweep --model <M> --run runs/ood_answer --n 200

# 3. SAE + partition + double dissociation -> figure2.png
python run.py fig2  --model <M> --run runs/ood_answer
```

Conditions: `ood_answer` (OOD + format pressure), `ood_cot` (OOD, CoT format kept —
**the crux cell**), `replay` (forgetting suppressed), `random_label` (weights move,
nothing learned). Run `ood_cot` and `replay` or the result is not identified: they
are what separate forgetting from formatting.

## The gate is the whole ballgame

Do not train on a cell that fails it. Thresholds in `metrics.GATE`:

| metric | need | why |
|---|---|---|
| accuracy | 0.60–0.90 | headroom in both directions |
| CoT lift | ≥ 0.25 | the chain does real work |
| error propagation | ≥ 0.60 | corrupting a step changes the answer |
| **shuffled-CoT gap** | **≥ 0.20** | **content is read, not just position** |
| degenerate fraction | ≤ 0.10 | looping chains are trivially unfaithful |

The shuffled gap is non-negotiable. It is exactly the metric on which Qwen3-0.6B
scores ~0 on GSM8K. If no cell clears it at 1.5B, stop — you have no dynamic range
and there is nothing to measure a fall in.

## Interrupted sessions

`--max-seconds` checkpoints and exits cleanly before the session wall; re-run the
identical command to continue. Resume is bit-exact (tested). Two guards:

- resuming with a changed `steps`/`lr`/`seed`/`rank` **refuses to run** rather than
  silently altering the LR schedule
- `sweep` skips checkpoints it already evaluated and rewrites `sweep.jsonl` after
  every step, so a kill costs one checkpoint, not the run

Set `--max-seconds 12000` on Kaggle (12 h wall) and `--max-seconds 13000` on
Lightning if you leave a Studio running.

## GPU notes

Precision is chosen automatically: bf16 on Ampere+, fp16 base with **float32 LoRA**
on Turing (T4) and Pascal (P100). That combination is what stops the NaN at ~step
200; the loop raises with a pointed message if it ever sees a non-finite loss. No
FlashAttention on T4 — `sdpa` is set for you.

**fp16 and the SAE.** Attribution takes gradients back through every layer above
the splice, and in fp16 those underflow to exactly zero after ~20 layers. The
objective is loss-scaled (`analysis.GRAD_SCALE`, default 2^14) and the code raises
if the gradient still collapses — it never returns a silent zero vector. On a 28-layer
model splice at `--layer 22` or later.

## Running on Kaggle only

See **KAGGLE.md** for the full run book. The headline: Kaggle bills session hours,
not per-GPU hours, so with T4 ×2 you should always run two conditions in parallel
via `CUDA_VISIBLE_DEVICES`. Both figures fit in one 30-hour week.

## What to check in the output

`fig2` prints `Partition.check()` warnings. Take them seriously:
an empty `M_gen`, a Jaccard above 0.2, or mediators that are mostly domain latents
each mean the double dissociation cannot show what H4 predicts. The code raises
rather than running an arm with nothing in it — an empty arm would be silently
identical to the control.

## Files

```
selftest.py        21 offline checks — run first
run.py             CLI: gate | train | sweep | fig2
cfd/tasks.py       synthetic tasks, ground-truth intermediates, perturbations
cfd/engine.py      span-exact tokenising, scoring, rho_CoT, logit-lens d*
cfd/metrics.py     faithfulness battery, aggregation, the gate
cfd/train.py       LoRA (no peft), subspace gradient mask, resumable checkpoints
cfd/analysis.py    TopK SAE, splice, attribution, partition, double dissociation
cfd/rig.py         model loading, io, FADG, figures
```

`--domain-jsonl` takes `{"prompt": ..., "target": ...}` per line when you swap the
built-in synthetic domain for MedMCQA or your own data.
