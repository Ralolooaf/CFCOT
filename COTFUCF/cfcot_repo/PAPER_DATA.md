# Consolidated data for the ICEMIR 2026 full paper

Title: Chain-of-Thought Faithfulness Under Catastrophic Forgetting and in
Multi-agent Coordination Using Causal Interventions

Everything below is measured, not remembered. Three bodies of evidence:
an exploratory study (the Apart sprint), a pre-registered replication of it,
and a pre-registered multi-agent study.

---

## 1. Study A (exploratory) — Apart sprint, September 2026

Model DeepSeek-R1-Distill-Qwen-1.5B. Task: modular arithmetic, 2 hops, mod 11,
six options (chance 0.167). 600 steps, 13 checkpoints, 60 items per checkpoint,
LoRA rank 32, lr 4e-4, attention + MLP, two T4s. Seeds 0 and 1.

Forgetting check (generic NLL on a fixed English corpus, disjoint from training):

| | ood_cot | replay |
|---|---|---|
| general NLL | 4.462 -> 4.898 (+0.436) | 4.462 -> 4.430 (-0.032) |
| degeneracy, last checkpoint | 0.00 | 0.00 |
| items scored, last | 60/60 | 60/60 |

Chain-free accuracy 0.183 -> 0.150 against chance 0.167.

Four length-invariant measures (tail):

| measure | step 0 | ood_cot | replay |
|---|---|---|---|
| error propagation | 0.500 | 0.486 | 0.422 |
| shuffled-chain gap | 0.133 | 0.229 | 0.031 |
| chain contribution to accuracy | 0.783 | 0.762 | 0.575 |
| premise masking | +0.017 | -0.201 | +0.254 |

Aggregate contrast: seed 0 +0.030, seed 1 +0.026.

Length-sensitive measures, reported for comparability with Lobo et al., NOT as
findings (chain length rose 19.5 -> 24.1 in ood_cot, fell to 9.0 in replay):
early_50 0.900/0.213/0.778, early_75 0.750/0.330/0.947,
filler_50 1.000/0.213/0.856, filler_75 1.000/0.305/0.936.

Known defects of Study A, to state in the paper:
- The replay arm was answer-only, so it also taught direct answering.
- Its replay data came from a separate 3-hop symbolic chain task, NOT from the
  evaluation task. The sprint report's Methods says otherwise; correct this.
- The out-of-domain clinical labels are drawn at random, independent of the
  note, so the domain data carries no learnable input-label relation. State it.
- No pre-registration; one gate condition was demoted after seeing early data.
- The ratio metric defined as primary moved only 0.021 and was abandoned.

---

## 2. Study B — pre-registered replication of A

Registered 2026-09-22T12:53:19Z, before any of its data existed. Code hash
identical at registration and evaluation (f1b1822f6a5e...). Fresh seeds 10, 11,
12. Same model and task. 600 steps, checkpoints evaluated at 0/400/500/600,
40 items per checkpoint, tail = steps >= 350.

Three arms: ood_cot (forgetting free), replay_cot (chained-format replay,
fixes the format confound), replay_generic (plain English replay, no reasoning
supervision).

### T1 — was the treatment applied, did controls suppress it

Drift is the settled-tail mean minus the value at step 0, as registered.

Registered rule: ood_cot drift >= 0.15 in every seed; a control suppresses if
its drift <= 0.5 x ood_cot's drift in that seed.

| run | generic-NLL drift |
|---|---|
| ood_cot s10 / s11 / s12 | +0.475 / +0.310 / +0.507 |
| replay_cot s10 / s11 / s12 | +0.443 / +0.300 / +0.125 |
| replay_generic s10 / s11 / s12 | +0.541 / +0.898 / +0.566 |

Treatment applied in all three seeds. **No control suppressed forgetting.**
replay_generic forgot MORE than the treatment arm in every seed.

### Accuracy trajectories (eval task, acc at 0/400/500/600)

| run | 0 | 400 | 500 | 600 |
|---|---|---|---|---|
| ood_cot s10 | 0.975 | 0.154 | 0.154 | 0.125 |
| ood_cot s11 | 0.950 | 0.750 | 0.600 | 0.625 |
| ood_cot s12 | 0.975 | 0.825 | 0.875 | 0.800 |
| replay_cot s10 | 0.975 | 0.150 | 0.150 | 0.150 |
| replay_cot s11 | 0.950 | 0.700 | 0.650 | 0.650 |
| replay_cot s12 | 0.975 | 0.575 | 0.875 | 0.875 |
| replay_generic s10 | 0.975 | 0.971 | 1.000 | 1.000 |
| replay_generic s11 | 0.950 | 0.925 | 0.950 | 0.950 |
| replay_generic s12 | 0.975 | 0.950 | 0.950 | 0.950 |

Chance is 0.167. **In seed 10 both ood_cot and replay_cot collapsed to chance.**
There, err_prop reads 1.000 — maximal apparent dependence on a chain whose
answers are noise. This is the second reason the comparison is uninterpretable,
and it is a gate that Study A did not have: its calibration checked chain
length and degeneracy, not task accuracy.

### Verdicts (registered rule: >= 3 of 4 measures in the predicted direction,
in every seed; null probability (5/16)^3 = 0.031 under independent coins)

- **P1** (ood_cot vs replay_cot): UNINTERPRETABLE — control did not suppress.
  Per seed, measures in direction: 3/4, 4/4, 4/4.
- **P2** (ood_cot vs replay_generic): UNINTERPRETABLE — control did not suppress.
  Per seed: 1/4, 1/4, 1/4.
- **R1** (err_prop regime test): UNDEFINED — seed 10 values not computable.
- **R2** (cot_lift regime test): UNDEFINED, same reason.

### Regime selection (registered rule: highest chain-free accuracy, if it is at
least chance + 0.25)

| candidate | chain-free acc | chance | qualifies |
|---|---|---|---|
| modarith h1 d0 | 0.150 | 0.167 | no |
| chain h1 d0 | 0.425 | 0.167 | yes |
| chain h2 d0 | 0.250 | 0.167 | no |

Shortcut task selected: chain, 1 hop, 0 distractors.

DiD values that were computed (seed 10 missing):
- R1 err_prop: s11 eval +0.105, shortcut +0.145, DiD +0.040; s12 eval +0.013,
  shortcut -0.058, DiD -0.071.
- R2 cot_lift: s11 eval -0.392, shortcut -0.170, DiD +0.222; s12 eval -0.208,
  shortcut +0.208, DiD +0.417.

### Descriptives (tail means over seeds; reported, not tested)

| | err_prop | shuf_gap | cot_lift | delta_rules | acc | acc_direct | n_tokens | early_50 | filler_50 | nll_drift |
|---|---|---|---|---|---|---|---|---|---|---|
| ood_cot | 0.702 | 0.147 | 0.344 | -0.043 | 0.545 | 0.201 | 84.3 | 0.658 | 0.731 | 0.431 |
| replay_cot | 0.512 | 0.044 | 0.300 | +0.473 | 0.531 | 0.231 | 42.1 | 0.264 | 0.269 | 0.289 |
| replay_generic | 0.508 | 0.098 | 0.798 | -0.219 | 0.961 | 0.163 | 141.4 | 0.500 | 0.922 | 0.668 |

Note the chain lengths: 84 / 42 / 141 tokens. The arms are not length-matched,
which is a further reason not to read the descriptives as a result.

---

## 3. Study C — pre-registered multi-agent coordination

Registered 2026-09-21T17:16:24Z before its data existed; code hash identical at
registration and evaluation (ff0473c9242d...). Qwen2.5-1.5B-Instruct, self-play,
Bertrand duopoly with logit demand (Calvano-style), unit cost 1.00, 60 rounds,
11-price menu, temperature 0.7, fresh seeds 100-104.

Five arms differing only in what each agent is shown about the other:
full (price + chain), action_only (price), corrupted (price + chain with every
number changed), shuffled (price + chain with lines reordered), blind (nothing).

Two readouts: LEVEL = collusion index (0 = one-shot Nash 1.470, 1 = joint
monopoly 1.925); GAP = mean absolute difference between the two agents' prices
in the settled tail.

| arm | s100 | s101 | s102 | s103 | s104 | mean gap |
|---|---|---|---|---|---|---|
| full | +0.637 / 0.0090 | +0.408 / 0.0315 | +0.629 / 0.0080 | +0.638 / 0.0005 | +0.799 / 0.0170 | 0.0132 |
| action_only | +0.397 / 0.0620 | +0.598 / 0.0605 | +0.597 / 0.0295 | +0.094 / 0.0515 | +0.726 / 0.0700 | 0.0547 |
| corrupted | +0.298 / 0.2185 | +0.503 / 0.2000 | +0.439 / 0.1785 | +0.712 / 0.2400 | +0.419 / 0.2425 | 0.2159 |
| shuffled | +0.705 / 0.0380 | +0.520 / 0.0495 | +0.798 / 0.0210 | +0.619 / 0.0130 | +0.631 / 0.0060 | 0.0255 |
| blind | +0.440 / 0.1085 | +0.573 / 0.0705 | +0.610 / 0.0795 | +0.577 / 0.1230 | +0.705 / 0.0990 | 0.0961 |

All four registered predictions passed, each 5/5 seeds, one-sided sign test
p = 1/32 = 0.031:

- **P1** gap(corrupted) > gap(full): mean difference +0.2027, 95% bootstrap CI
  [+0.1775, +0.2275], ratio 16.4.
- **P2** gap(corrupted) > gap(action_only): +0.1612 [+0.1467, +0.1774].
- **P3** gap(action_only) > gap(shuffled): +0.0292 [+0.0126, +0.0483].
- **P4** gap(action_only) > gap(full): +0.0415 [+0.0293, +0.0526].

**M1 menu-position test — CENTRALITY** (registered: centrality if slope >= 0.5):

| menu | midpoint level | blind level s150 | s151 |
|---|---|---|---|
| low (centred on Nash) | +0.000 | +0.027 | +0.041 |
| default | +0.495 | +0.567 | +0.661 |
| high (centred on monopoly) | +1.000 | +0.954 | +0.790 |

OLS slope +0.837. The blind arm's price level tracks the menu, not a belief
about prices. Levels therefore do not separate the arms; the gap does.

Caveats to state: one model, self-play, 60 rounds, an 11-price menu,
corruption is blunt (+-35%), 8-14% of rounds needed a one-turn follow-up to
produce a readable price (corrupted 14%, full 10%), total unparsed 1 of 3000.

---

## 4. What the paper claims, and what it does not

Claim 1 (Study A, exploratory): where no shortcut exists, forgetting did not
reduce the chain's causal role; it increased it relative to the control.

Claim 2 (Study B): that comparison does not replicate under controls that hold
up, because no control suppressed forgetting and one seed lost the task. The
pre-registered rules caught both. Do not report the descriptives as support.

Claim 3 (Study C): coordination between agents runs through the content of the
shared chain, not its order, and not through price observation alone. Price
level is not evidence of coordination: with no channel at all, agents price at
the menu's midpoint.

Claim 4 (method): chain-free accuracy is a one-number pre-deployment test of
which regime a task is in, and it needs only model weights.

---

## 5. Related work to cite, with arXiv identifiers

- Lobo, Agarwal, Lakkaraju. On the Impact of Fine-Tuning on Chain-of-Thought
  Reasoning. NAACL 2025, arXiv:2411.15382. The conjecture is in Appendix B.
  It also predicts the answer-only mechanism that defeated Study A's control.
- Korbak et al. Chain of Thought Monitorability. arXiv:2507.11473.
- Lanham et al. arXiv:2307.13702. Turpin et al. arXiv:2305.04388.
- Mechanistic Evidence for Faithfulness Decay. arXiv:2602.11201. Nearest
  methodological neighbour: corrupts reasoning steps while keeping surface
  coherence, varies chain length. Must be distinguished explicitly.
- From Concept Alignment to Causal Grounding. arXiv:2609.23065. SAE ablation.
- Breaking the Chain. arXiv:2603.16475. Causal intervention on structured
  mediators. (Also: avoid this title.)
- CASE. arXiv:2607.18820. Trains CoT-to-answer dependence.
- Kotha and Liang. Replaying pre-training data improves fine-tuning.
  arXiv:2603.04964. Directly relevant to fixing Study B's control.
- Han et al. 2026: RLVR reduces faithfulness by 10-14 points.
- Multi-agent side: Fish et al. (LLM algorithmic collusion; prompt wording
  drives supracompetitive pricing), On the Fragility of AI Agent Collusion
  (arXiv:2603.20281, price convergence + elevation dual criterion, asymmetric
  information), Faithful yet Collusive (arXiv:2609.18346, CoT monitoring of
  collusive pricing agents). No published work uses a symmetric no-observation
  control to separate prior from coordination.

---

## 6. Code and data

- Study A and B: github.com/Ralolooaf/CFCOT — cfd_all.py, runs/, suite/
  (prereg.json, regime.json, verdict.md, per-checkpoint jsonl, logs)
- Study C: separate repository — col_all.py, suite/ (prereg.json, arms.jsonl,
  menu.jsonl, verdict.md)

## 7. Next runs, in priority order

1. Find a replay ratio that suppresses forgetting: train only, measure generic
   NLL at the last checkpoint. ~15 min per configuration, no sweep needed.
2. Add an accuracy floor to the gate (eval accuracy must stay above chance +
   margin), then repeat Study B.
3. Study C at 3B, and a dose-response curve for corruption magnitude.
