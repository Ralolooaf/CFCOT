<!-- Study B, pre-registered replication. Verdict as printed by
     `python cfd_all.py suite` at the end of the confirmatory run.
     Reproduced verbatim from the Kaggle session transcript. -->

# Verdict

Registered: 2026-09-22T12:53:19.006506+00:00
Code at registration: f1b1822f6a5ed867e0e832844668d589c32a8c099a23ff6aacefec3cd5eb0655
Code at evaluation:   f1b1822f6a5ed867e0e832844668d589c32a8c099a23ff6aacefec3cd5eb0655

Runs complete: ood_cot_s10, ood_cot_s11, ood_cot_s12, replay_cot_s10, replay_cot_s11, replay_cot_s12, replay_generic_s10, replay_generic_s11, replay_generic_s12

## T1 — was forgetting induced, and did the controls suppress it

  ood_cot_s10              generic-NLL drift +0.475
  ood_cot_s11              generic-NLL drift +0.310
  ood_cot_s12              generic-NLL drift +0.507
  replay_cot_s10           generic-NLL drift +0.443
  replay_cot_s11           generic-NLL drift +0.300
  replay_cot_s12           generic-NLL drift +0.125
  replay_generic_s10       generic-NLL drift +0.541
  replay_generic_s11       generic-NLL drift +0.898
  replay_generic_s12       generic-NLL drift +0.566

## Comparisons (tail means; predicted: ood_cot more chain-dependent)

**P1** vs replay_cot — UNINTERPRETABLE
  The forgetting arm stays more chain-dependent than a replay control whose replay is in chained format.
  seed 10: 3/4 in direction, CONTROL DID NOT SUPPRESS
    err_prop_rate +1.000 vs +0.495; shuffled_acc_gap +0.000 vs -0.075; cot_lift -0.017 vs -0.058; delta_rules -0.094 vs -0.419 (x)
  seed 11: 4/4 in direction, CONTROL DID NOT SUPPRESS
    err_prop_rate +0.605 vs +0.571; shuffled_acc_gap +0.117 vs +0.067; cot_lift +0.408 vs +0.367; delta_rules +0.071 vs +0.942
  seed 12: 4/4 in direction
    err_prop_rate +0.500 vs +0.471; shuffled_acc_gap +0.325 vs +0.142; cot_lift +0.642 vs +0.592; delta_rules -0.106 vs +0.896
  chance of passing under an independent-coin null: 0.031 (optimistic: the measures correlate)

**P2** vs replay_generic — UNINTERPRETABLE
  The forgetting arm stays more chain-dependent than a control that suppresses forgetting with plain English and no reasoning supervision.
  seed 10: 1/4 in direction, CONTROL DID NOT SUPPRESS
    err_prop_rate +1.000 vs +0.541; shuffled_acc_gap +0.000 vs +0.060 (x); cot_lift -0.017 vs +0.794 (x); delta_rules -0.094 vs -0.166 (x)
  seed 11: 1/4 in direction, CONTROL DID NOT SUPPRESS
    err_prop_rate +0.605 vs +0.483; shuffled_acc_gap +0.117 vs +0.133 (x); cot_lift +0.408 vs +0.792 (x); delta_rules +0.071 vs -0.276 (x)
  seed 12: 1/4 in direction, CONTROL DID NOT SUPPRESS
    err_prop_rate +0.500 vs +0.500 (x); shuffled_acc_gap +0.325 vs +0.100; cot_lift +0.642 vs +0.808 (x); delta_rules -0.106 vs -0.215 (x)
  chance of passing under an independent-coin null: 0.031 (optimistic: the measures correlate)

## Regime test

  candidate modarith h1 d0: chain-free acc +0.150 (chance 0.167)
  candidate chain h1 d0: chain-free acc +0.425 (chance 0.167)  QUALIFIES
  candidate chain h2 d0: chain-free acc +0.250 (chance 0.167)
  shortcut task: ['chain', 1, 0]
**R1** (err_prop_rate) — UNDEFINED
  seed 10: change on eval task +0.450, on shortcut task   n/a, DiD   n/a
  seed 11: change on eval task +0.105, on shortcut task +0.145, DiD +0.040
  seed 12: change on eval task +0.013, on shortcut task -0.058, DiD -0.071
**R2** (cot_lift) — UNDEFINED
  seed 10: change on eval task -0.842, on shortcut task   n/a, DiD   n/a
  seed 11: change on eval task -0.392, on shortcut task -0.170, DiD +0.222
  seed 12: change on eval task -0.208, on shortcut task +0.208, DiD +0.417

## Descriptives — reported, not tested

  ood_cot         err_prop_rate +0.702  shuffled_acc_gap +0.147  cot_lift +0.344  delta_rules -0.043  acc +0.545  acc_direct +0.201  n_tokens +84.328  early_50 +0.658  filler_50 +0.731  nll_drift +0.431
  replay_cot      err_prop_rate +0.512  shuffled_acc_gap +0.044  cot_lift +0.300  delta_rules +0.473  acc +0.531  acc_direct +0.231  n_tokens +42.119  early_50 +0.264  filler_50 +0.269  nll_drift +0.289
  replay_generic  err_prop_rate +0.508  shuffled_acc_gap +0.098  cot_lift +0.798  delta_rules -0.219  acc +0.961  acc_direct +0.163  n_tokens +141.417  early_50 +0.500  filler_50 +0.922  nll_drift +0.668
