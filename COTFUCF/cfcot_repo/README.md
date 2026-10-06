# Chain-of-thought faithfulness under catastrophic forgetting

Studies A and B of *Chain-of-Thought Faithfulness Under Catastrophic Forgetting
and in Multi-agent Coordination Using Causal Interventions* (ICEMIR 2026).

Companion repository: **Study C (multi-agent coordination)** — https://github.com/Ralolooaf/multi-cfcot

The question: does ordinary fine-tuning change how load-bearing a model's
chain-of-thought is, and how would anyone know before deployment? Study A is
exploratory. Study B is a pre-registered replication of it that **did not
reproduce the comparison**, and identified the conditions under which the
comparison is measurable at all.

---

## Running it

```bash
python cfd_all.py check                                  # 21 self-checks, no GPU
python cfd_all.py gate    --model <M>                    # phase-0 go/no-go
python cfd_all.py pipeline --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
python cfd_all.py suite                                  # pre-registered, resumable
```

`suite` writes `prereg.json` **before** any data exists, with a UTC timestamp and
a hash of the code, and refuses to overwrite it. Everything resumes: re-running
any command continues where it stopped. See `CFD_SUITE_KAGGLE.md` for the
notebook setup. All computation for the paper used the free tier of a public
notebook service (2× T4).

---

## Study A — exploratory

DeepSeek-R1-Distill-Qwen-1.5B. Task: modular arithmetic, 2 hops, mod 11, six
options (chance 0.167). 600 steps, 13 checkpoints, 60 items per checkpoint, LoRA
rank 32 on attention and MLP, lr 4e-4, two T4s. **Seeds 0 and 1.**

Two arms, one variable: `ood_cot` (forgetting left free) and `replay` (mixes in
examples from a separate 3-hop symbolic rule-chaining task).

### Preconditions

| | `ood_cot` | `replay` |
|---|---|---|
| generic NLL | 4.462 → 4.898 (**+0.436**) | 4.462 → 4.430 (**−0.032**) |
| degeneracy, last checkpoint | 0.00 | 0.00 |
| items scored, last | 60/60 | 60/60 |

Chain-free accuracy **0.183 → 0.150** against chance **0.167** — the task admits
no shortcut.

### The four length-invariant measures (settled tail)

| measure | step 0 | `ood_cot` | `replay` |
|---|---|---|---|
| error propagation | 0.500 | 0.486 | 0.422 |
| shuffled-chain gap | 0.133 | 0.229 | 0.031 |
| chain contribution to accuracy | 0.783 | 0.762 | 0.575 |
| premise masking | +0.017 | −0.201 | +0.254 |

All four move the same way: the forgetting arm is **more** dependent on its
chain than the control, on every one.

**Aggregate contrast: seed 0 +0.030, seed 1 +0.026.** Same sign at both seeds.

### Length-sensitive measures — recorded, support no claim

Reported for comparability with Lobo et al., **not as findings.** Both cut the
chain at a fixed fraction of its length, so the cut point moves when length
changes — and chain length rose 19.5 → 24.1 lines in `ood_cot` and fell to 9.0
in `replay`.

`early_50` 0.900 / 0.213 / 0.778 · `early_75` 0.750 / 0.330 / 0.947 ·
`filler_50` 1.000 / 0.213 / 0.856 · `filler_75` 1.000 / 0.305 / 0.936
(step 0 / `ood_cot` / `replay`)

### Known defects of Study A

Stated here as they are stated in the paper:

- The replay arm was **answer-only**, so it also taught direct answering. This
  works *in favour* of the reported difference.
- Its replay data came from a **separate 3-hop symbolic chain task**, not from
  the evaluation task. The Apart sprint report's Methods says otherwise; that is
  an error in the sprint report, corrected here and in the paper.
- The out-of-domain clinical labels are drawn at random, independent of the note,
  so the domain data carries **no learnable input-label relation**. It acts as
  pure memorisation pressure.
- **No pre-registration.** One gate condition was demoted after early data.
- The ratio metric nominated as primary moved only **0.021** and was abandoned.

Study A is reported as observation, not as a finding, for these reasons.

---

## Study B — pre-registered replication

**Registered 2026-09-22T12:53:19Z**, before any of its data existed. Code hash
identical at registration and at evaluation (`f1b1822f6a5e...`).

Fresh seeds **10, 11, 12** — none from Study A. Same model and task. 600 steps,
checkpoints evaluated at 0 / 400 / 500 / 600, 40 items per checkpoint,
settled tail = steps ≥ 350.

**Three arms:**

| arm | what it is |
|---|---|
| `ood_cot` | forgetting left free |
| `replay_cot` | chained-format replay — fixes Study A's format confound: same source task, items and count, **only the format changes** to a reference chain followed by the answer |
| `replay_generic` | plain English replay, disjoint from the corpus used to measure forgetting — suppresses drift with **no reasoning supervision** |

### T1 — was the treatment applied, did the controls suppress it

Registered rule: `ood_cot` drift ≥ **0.15** in every seed; a control suppresses
if its drift is **≤ 0.5 ×** `ood_cot`'s drift in that seed. Drift is the
settled-tail mean minus the step-0 value, as registered.

| run | s10 | s11 | s12 |
|---|---|---|---|
| `ood_cot` | **+0.475** | **+0.310** | **+0.507** |
| `replay_cot` | +0.443 | +0.300 | +0.125 |
| `replay_generic` | +0.541 | +0.898 | +0.566 |

Treatment applied in all three seeds. **No control suppressed forgetting.**
`replay_generic` forgot **more** than the treatment arm in every seed.

*(This is Table 2 of the paper.)*

### Accuracy trajectories — the second failure

| run | 0 | 400 | 500 | 600 |
|---|---|---|---|---|
| `ood_cot` s10 | 0.975 | **0.154** | **0.154** | **0.125** |
| `ood_cot` s11 | 0.950 | 0.750 | 0.600 | 0.625 |
| `ood_cot` s12 | 0.975 | 0.825 | 0.875 | 0.800 |
| `replay_cot` s10 | 0.975 | **0.150** | **0.150** | **0.150** |
| `replay_cot` s11 | 0.950 | 0.700 | 0.650 | 0.650 |
| `replay_cot` s12 | 0.975 | 0.575 | 0.875 | 0.875 |
| `replay_generic` s10 | 0.975 | 0.971 | 1.000 | 1.000 |
| `replay_generic` s11 | 0.950 | 0.925 | 0.950 | 0.950 |
| `replay_generic` s12 | 0.975 | 0.950 | 0.950 | 0.950 |

Chance is **0.167**. In **seed 10** both `ood_cot` and `replay_cot` collapsed to
chance and stayed there. There `err_prop` reads **1.000** — its maximum, i.e.
maximal apparent dependence on a chain whose answers are noise.

Study A's calibration checked chain length and degeneracy but **not task
accuracy**. That is the gap it left: a capability floor belongs in the gate.

*(This is Table 3 of the paper.)*

### Verdicts

Registered rule: a comparison passes if ≥ 3 of the 4 measures differ in the
predicted direction in **every** seed. Null probability (5/16)³ = 0.031 under
independent coins — reported for calibration, not relied upon, since the
measures are correlated.

| | comparison | verdict | why |
|---|---|---|---|
| **P1** | `ood_cot` vs `replay_cot` | **UNINTERPRETABLE** | control did not suppress. Per seed, measures in direction: 3/4, 4/4, 4/4 |
| **P2** | `ood_cot` vs `replay_generic` | **UNINTERPRETABLE** | control did not suppress. Per seed: 1/4, 1/4, 1/4 |
| **R1** | `err_prop` regime test | **UNDEFINED** | seed 10 values not computable |
| **R2** | `cot_lift` regime test | **UNDEFINED** | same reason |

The registered rule reports the mechanism test as *undefined* rather than
resolving it on partial data.

### Regime selection — measured, not chosen by hand

Registered rule: among the candidates, the one with the highest chain-free
accuracy at step 0, provided that is at least chance + 0.25.

| candidate | chain-free acc | chance | qualifies |
|---|---|---|---|
| modarith, 1 hop, 0 distractors | 0.150 | 0.167 | no |
| **chain, 1 hop, 0 distractors** | **0.425** | 0.167 | **yes** |
| chain, 2 hops, 0 distractors | 0.250 | 0.167 | no |

A task being easier does not make it shortcut-prone — which is why the
registration made the choice a measured one.

Difference-in-differences where computable (seed 10 missing):

- **R1** `err_prop`: s11 eval +0.105, shortcut +0.145, DiD **+0.040**;
  s12 eval +0.013, shortcut −0.058, DiD **−0.071**
- **R2** `cot_lift`: s11 eval −0.392, shortcut −0.170, DiD **+0.222**;
  s12 eval −0.208, shortcut +0.208, DiD **+0.417**

### Descriptives — reported, not tested

Tail means over seeds. **Do not read these as support.**

| | err_prop | shuf_gap | cot_lift | delta_rules | acc | acc_direct | n_tokens | early_50 | filler_50 | nll_drift |
|---|---|---|---|---|---|---|---|---|---|---|
| `ood_cot` | 0.702 | 0.147 | 0.344 | −0.043 | 0.545 | 0.201 | **84.3** | 0.658 | 0.731 | 0.431 |
| `replay_cot` | 0.512 | 0.044 | 0.300 | +0.473 | 0.531 | 0.231 | **42.1** | 0.264 | 0.269 | 0.289 |
| `replay_generic` | 0.508 | 0.098 | 0.798 | −0.219 | 0.961 | 0.163 | **141.4** | 0.500 | 0.922 | 0.668 |

Two reasons they are unusable. First, no arm held forgetting down, so the
contrast the study exists to measure was never created. Second, mean chain
length differs by more than a factor of three across arms — **84 / 42 / 141
tokens** — so the arms are not comparable in the very quantity the measures were
designed to be invariant to.

Without the registration, `err_prop` 0.702 against 0.512 and 0.508 would have
been read as support.

*(This is Table 4 of the paper.)*

---

## What Study B establishes

The replication neither confirms nor refutes Study A. What it establishes is the
set of conditions under which that comparison is measurable at all — and both
failures were caught by rules written before the data existed, and both would
have been easy to miss afterwards.

**Two gates for any faithfulness study:**

1. A control must be **shown** to suppress the effect it controls for.
2. The model must be **shown** to still perform the task whose reasoning is
   being measured.

The most likely explanation for the difference: replay in answer-only form
presents short targets and small gradients, and Study A's arm did suppress drift
(−0.032). Presenting the same items with a reference chain lengthens the
targets, and that arm no longer suppressed it. If so, the protection in Study A
came in part from the **format** of its control rather than from replay as such
— which is the confound Study B was built to remove. Removing it removed the
suppression too.

The immediate next experiment is a search over replay proportion and source,
measuring only the final-checkpoint drift: minutes per configuration, no
evaluation sweep.

---

## Contents

| | |
|---|---|
| `cfd_all.py` | the whole thing — harness, pre-registration, 21 self-checks |
| `cfd_source/` | the same code split into modules (`_cli.py`, `run.py`, `selftest.py`, `preflight.py`, `build_single.py`) |
| `CFD_SUITE_KAGGLE.md` | notebook setup |
| `PAPER_DATA.md` | every number in the paper with its provenance |
| `figures/` | `fig1_studyA.png`, `fig2_studyB.png`, `figure1.png` |
| `suite/verdict.md` | the Study B verdict as the harness printed it — both code hashes, T1 drift, P1/P2 per seed and per measure, regime test, R1/R2, descriptives |
| `suite/run_manifest.txt` | the confirmatory run itself: job scheduling across both T4s, per-job log paths, wall time per seed (2.81 h / 4.96 h / 7.73 h) |
| `suite/calibration_best.json` | the calibration sweep winner that then collapsed: `lr 5e-4`, rank 32, `nll_drift 0.925` |
| `logs/gate_detail_both_models.txt` | the phase-0 gate in full, both 1.5B models, per cell, with the timing probe and `[check ] attention knockout verified`. **No cell passed.** For DeepSeek the blocker is truncation (up to 39%); for Qwen2.5 the chains are complete and the gate reports that the model does not use its chain on these tasks |
| `logs/gate_earlier_band_0.95.txt` | the same gate, earlier — accuracy band reads `need 0.6-0.95` where the later run reads `0.6-0.9`. This is the gate condition the paper discloses as changed after early data |
| `logs/gate_no_cell_passed.txt` | a third gate session |
| `logs/gate_exit2_truncation.txt` | a fourth |
| `logs/session_disk_full.txt` | a session that died mid-run (torch checkpoint corruption, then disk full) and resumed without re-running seed 10 |

### Still to add from the notebook

The files above are the **session transcripts**, which carry every number and
every verdict. The primary artifacts live inside `suite_results.zip` in the
Kaggle working directory and are not here yet:

- `prereg.json` — the registration as a file, with its UTC timestamp and code hash
- the per-checkpoint result rows the tail means were computed from
- the individual `logs/*.log` files the manifest points at
- Study A's seed 0 and seed 1 sweep outputs

Download `suite_results.zip` and commit its contents under `suite/`. The
numbers are already independently checkable against `verdict.md`; the zip makes
the rows checkable too.

---

Limitations are stated in the paper: 1.5B parameters in both settings, chosen
because the interventions need weights and attention, not because the size is
representative; synthetic, non-agentic tasks; Study A on two seeds and
exploratory; Study B on three, producing no usable comparison.

No funding was received.
