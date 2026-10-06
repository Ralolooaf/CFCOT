# Kaggle: one file, one cell

## Setup

Notebook settings: **Accelerator = GPU T4 ×2**, **Internet = On**.

Upload `cfd_all.py` (one file — drag it into *Add Data → Upload*, or just paste it
into a cell prefixed with `%%writefile cfd_all.py`).

## Two ways to run it. Both work.

### A — upload the file (recommended)

Drag `cfd_all.py` in as a Kaggle Dataset, then one cell:

```python
!cp /kaggle/input/<your-dataset>/cfd_all.py /kaggle/working/
!cd /kaggle/working && python cfd_all.py pipeline
```

### B — paste the file straight into a cell and run it

Nothing else. It notices it is in a notebook, **writes a copy of itself to disk**
(recovered from IPython's input history, since a pasted cell has no `__file__`),
and then **starts the pipeline**. The saved copy is what it hands to its per-GPU
worker processes.

Interrupt the cell to stop; re-run it to continue where it stopped.

To load the functions without starting a run, set `CFD_NO_AUTORUN=1` in a cell
before the paste; then call `cfd('check')`, `cfd('eta')` or `cfd('pipeline')`
yourself. `CFD_ARGS` overrides what auto-run launches, e.g.

```python
import os; os.environ['CFD_ARGS'] = 'pipeline --n 100'
```

Option A is still slightly better on Kaggle: a shell cell keeps worker logs
streaming where you can `tail` them, and you avoid the notebook editor mangling a
3,400-line paste.

That is the whole run. It installs what is missing, runs 26 self-checks, runs the
phase-0 gate on two models across both GPUs, picks the best task cell, trains and
sweeps `ood_cot` and `replay`, builds the SAE, runs the double dissociation, and
writes `FIGURE1.png` and `FIGURE2.png` to `/kaggle/working/cfd_runs/`.

Run it with **Save & Run All (Commit)** so it keeps going after you close the tab.

## If the session dies

Add this notebook's output as a data source to a new notebook, then:

```python
!cp -r /kaggle/input/<previous-notebook>/cfd_runs /kaggle/working/
!cp /kaggle/input/<your-dataset>/cfd_all.py /kaggle/working/
!cd /kaggle/working && python cfd_all.py pipeline
```

Identical command. It resumes bit-exactly: finished checkpoints are not retrained,
evaluated checkpoints are not re-evaluated, and a finished Figure 2 is not rebuilt.

## Looking at the results

Everything lands in `/kaggle/working/cfd_runs/`:

| file | what it is |
|---|---|
| `fig1_silent_window.png` | CoT causal share and accuracy vs step, per condition |
| `fig2_fadg_by_condition.png` | the decoupling gap, with the controls |
| `fig3_ablation.png` | ablation effects on faithfulness and fluency |
| `fig4_commitment_depth.png` | depth at which the answer stabilises |
| `fig5_forgetting_coupling.png` | faithfulness loss against how much was forgotten |
| `fig6_battery.png` | every faithfulness measure over training |
| `fig7_dvr.png` | which latent group fine-tuning disturbed |
| `results.csv` | every checkpoint, every metric |
| `results.md` | the summary table, ready to paste |

Black, red and grey only, 300 dpi, serif, no gridlines. Rebuild any time without a
GPU:

```bash
python cfd_all.py figures --root /kaggle/working/cfd_runs
```

```python
from IPython.display import Image, display
d = '/kaggle/working/cfd_runs/'
for f in ['fig1_silent_window','fig2_fadg_by_condition','fig3_ablation',
          'fig5_forgetting_coupling','fig6_battery','fig7_dvr']:
    display(Image(d + f + '.png'))
print(open(d + 'results.md').read())
```

### Figure 1 — the primary claim, and the one genuinely unclaimed

A **contrast**, not a single number:

- `fadg > 0` on `ood_cot` (faithfulness crosses before accuracy), and
- `fadg` near zero or undefined on `replay`

Format pressure is identical in both; only forgetting differs. If the window opens
when forgetting is allowed and closes when replay suppresses it, forgetting is the
driver. With `--ckpt-every 40`, treat anything under ~80 steps as noise.

Check the `[forgetting] generic NLL` line first. A flat faithfulness curve with no
NLL drift means the treatment was never applied, not that the hypothesis is false.

### Figure 2 — read this before believing it

`figure2.json` holds two results. They are not equally safe.

**`dvr`** — which latents fine-tuning actually disturbed. `DVR > 1` with
`med_beats_random: true` means the mediators moved more than the generators. That
is a claim about *damage*, not about separability, and it is the one to lead with.

**`test.dissociation`** — the stronger claim that mediators and generators are
functionally separable. Prior work (arXiv 2608.08168) reports the opposite: that
reasoning and formatting share representations rather than sitting in separable
modules. A `true` here **contradicts published work** and must be airtight before
you write it down; a `false` **replicates** them and is not novel. Report it as a
test of someone else's claim, not as your headline.

Do not ignore the `WARNING:` lines from `Partition.check()`.

## Knobs worth touching

```bash
# the 6 h default: gate, ood_cot + replay, both figures, one seed
python cfd_all.py pipeline

# add the prevention condition (C3). Re-runs ood_cot with the mediator subspace
# projected out of the gradient. This is what turns an analysis into a method:
# if a small targeted mask closes the silent window at a lower target-task cost
# than replay, the mediators were carrying faithfulness, not merely correlated
# with it. ~+2 h.
python cfd_all.py pipeline --protect

# the full 2x2 identification: format pressure and "any weight movement" ruled out
python cfd_all.py pipeline --conditions ood_cot,replay,ood_answer,random_label

# error bars: run the same command per seed into separate roots
python cfd_all.py pipeline --seed 1
python cfd_all.py pipeline --seed 2

# your own fine-tuning domain instead of the built-in synthetic one
python cfd_all.py pipeline --domain-jsonl medmcqa.jsonl   # {"prompt":..,"target":..}
```

## Stages separately, if you want them

```bash
python cfd_all.py check                              # 26 self-checks, no GPU
python cfd_all.py gate  --model M --out runs/gate
python cfd_all.py train --model M --cond ood_cot --run runs/ood_cot
python cfd_all.py sweep --model M --run runs/ood_cot
python cfd_all.py fig2  --model M --run runs/ood_cot
```

## How long it takes, per accelerator

Ask the code — it measures your hardware and knows how many cards it has:

```bash
python cfd_all.py eta
```

Measured on a Kaggle T4 with a 1.5B model: **~8 s per item** (2.6 s to generate the
chain, 5.5 s for the whole faithfulness battery). Projections at the defaults
(`--n 100 --ckpt-every 40`, 2 gate models, 2 conditions, 16 checkpoints):

| accelerator | per item | gate | sweep | total |
|---|---|---|---|---|
| **GPU T4 x2** — the right choice | 8.2 s | 2.0 h | 3.6 h | **~6.1 h** |
| GPU P100 (one card) | 9.4 s | 4.7 h | 8.4 h | ~13.5 h |
| L4 x1, if your account offers it | 4.3 s | 2.2 h | 3.9 h | ~6.5 h |
| L4 x4, if your account offers it | 4.3 s | 1.1 h | 1.9 h | ~3.4 h |

**TPU v5e-8: do not use it.** This pipeline is forward hooks, custom 4-D attention
masks, activation patching and per-latent ablation. Every one of those changes the
graph, and torch_xla recompiles on each change — the run would be slower than CPU.
TPUs are for training loops with fixed shapes, which this is not.

**With four cards, run four conditions.** Jobs are dispatched one per GPU, so with
only two conditions two cards sit idle:

```bash
python cfd_all.py pipeline --conditions ood_cot,replay,ood_answer,random_label
```

All four for the wall-clock price of two, and you get the full 2x2 identification
instead of just the crux pair.

**Do not drop below about 12 checkpoints.** FADG is the gap between two crossing
points; with too few samples it cannot locate either, and the headline number is
undefined. At 600 steps that means `--ckpt-every 50` is the floor.

If `eta` says you are over budget it prints a concrete command that fits.

## T4 notes, handled for you

No bf16 on T4, so the base loads in fp16 with **float32 LoRA** — the combination
that stops the NaN around step 200. The `torch_dtype` → `dtype` rename between
transformers 4.x and 5.x is resolved by checking the dtype the model actually came
back with, because guessing wrong loads it in fp32 and OOMs with an unrelated error.
SAE attribution gradients are loss-scaled, and if they still collapse in fp16 the
run stops loudly instead of returning a silent zero vector.
