"""
Offline self-test. Builds a tiny random Qwen2 and a byte-level stub tokenizer, then
exercises every numerical path in the package. No downloads, no GPU, ~20 seconds.

Run this FIRST on any new machine:      python selftest.py
and again after any edit to engine.py.  A green run means the mechanics are sound;
it says nothing about whether the real model has faithful CoT (that is phase 0).
"""
from __future__ import annotations

import random
import sys
import traceback
from typing import List

import torch

from cfd import tasks
from cfd.engine import (
    Builder, Seq, build_mask, verify_knockout, score_candidates, margin,
    rho_cot, commitment_depth, generate_cots, degeneracy, _clean_cot,
)

PASS, FAIL = "  PASS", "  FAIL"
_results: List[tuple] = []


def check(name, fn):
    try:
        fn()
        _results.append((name, True, ""))
        print(f"{PASS}  {name}")
    except Exception as e:                                    # noqa: BLE001
        _results.append((name, False, f"{type(e).__name__}: {e}"))
        print(f"{FAIL}  {name}\n         {type(e).__name__}: {e}")
        traceback.print_exc(limit=3)


# -----------------------------------------------------------------------------
# stub tokenizer: byte-level, deterministic, close enough to exercise every path
# -----------------------------------------------------------------------------

class StubTok:
    OFFSET = 4          # 0=pad 1=bos 2=eos 3=unk

    def __init__(self, with_template=True):
        self.pad_token, self.bos_token, self.eos_token = "<pad>", "<bos>", "<eos>"
        self.pad_token_id, self.bos_token_id, self.eos_token_id = 0, 1, 2
        self.padding_side = "right"
        self.chat_template = "stub" if with_template else None

    def encode(self, text, add_special_tokens=False):
        ids = [b + self.OFFSET for b in text.encode("utf-8")]
        return ([self.bos_token_id] + ids) if add_special_tokens else ids

    def decode(self, ids, skip_special_tokens=True):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        # a randomly initialised model emits ids across the whole vocab, so clamp
        # to the byte range instead of raising -- a real tokeniser never faults here
        bs = bytes(i - self.OFFSET for i in ids
                   if isinstance(i, int) and self.OFFSET <= i < self.OFFSET + 256)
        return bs.decode("utf-8", errors="ignore")

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
        assert not tokenize
        body = "".join(f"<|u|>{m['content']}<|/u|>" for m in msgs)
        return "<bos>" + body + ("<|a|>" if add_generation_prompt else "")


def tiny_model(vocab=320, layers=3, hidden=32, heads=4):
    from transformers.models.qwen2 import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=vocab, hidden_size=hidden, intermediate_size=2 * hidden,
                      num_hidden_layers=layers, num_attention_heads=heads,
                      num_key_value_heads=2, max_position_embeddings=4096,
                      tie_word_embeddings=False)
    return Qwen2ForCausalLM(cfg).eval()


DEV = "cpu"


def main() -> int:
    tok = StubTok()
    b = Builder(tok)
    model = tiny_model()
    ds_chain = tasks.build_dataset("chain", 8, seed=3, hops=3, distractors=3)
    ds_mod = tasks.build_dataset("modarith", 8, seed=3, hops=3, modulus=11)
    it = ds_chain[0]

    print("\n[1] span bookkeeping")

    def t_spans():
        s = b.full(it, it.gold_cot)
        # spans tile the sequence exactly, in order, with no gaps or overlaps
        prev_end = 0
        for name in ("head", "rules", "question", "genprompt", "cot", "bridge"):
            st, en = s.spans[name]
            assert st == prev_end, f"gap/overlap before {name}: {st} != {prev_end}"
            assert en >= st, f"negative span {name}"
            prev_end = en
        assert prev_end == len(s.ids), f"spans cover {prev_end} of {len(s.ids)}"
        # decoded rules span must equal the rules text
        st, en = s.spans["rules"]
        assert tok.decode(s.ids[st:en]) == it.rules_text, "rules span mismatch"
        st, en = s.spans["cot"]
        assert tok.decode(s.ids[st:en]) == it.gold_cot, "cot span mismatch"
    check("spans tile sequence exactly; decode round-trips", t_spans)

    def t_empty_cot():
        s = b.full(it, "")
        assert s.spans["cot"][0] == s.spans["cot"][1], "empty cot must be empty span"
        assert len(s.ids) > 0
        pos = s.span_positions("cot")
        assert pos == []
    check("empty CoT gives a legal zero-width span", t_empty_cot)

    def t_no_template():
        b2 = Builder(StubTok(with_template=False))
        s = b2.full(it, it.gold_cot)
        assert len(s.ids) > 0 and s.spans["bridge"][1] == len(s.ids)
    check("builder works without a chat template", t_no_template)

    print("\n[2] attention masks")

    def t_mask_shape():
        m = build_mask([5, 7], 7, torch.float32, [[1], []], "cpu")
        assert m.shape == (2, 1, 7, 7)
        neg = torch.finfo(torch.float32).min
        assert m[0, 0, 3, 1] == neg, "blocked key must be masked"
        assert m[0, 0, 3, 0] == 0.0, "unblocked past key must be visible"
        assert m[0, 0, 1, 3] == neg, "future key must be masked (causal)"
        assert m[0, 0, 4, 5] == neg, "right padding must be masked"
        assert m[1, 0, 6, 5] == 0.0, "row 1 has 7 real tokens"
        # self is always visible -> no all-masked softmax row
        for bb in range(2):
            for i in range(7):
                assert m[bb, 0, i, i] == 0.0, "diagonal must stay visible"
    check("mask is causal, pad-aware, knockout-aware, NaN-safe", t_mask_shape)

    def t_verify():
        verify_knockout(model, DEV)
    check("verify_knockout passes on this transformers build", t_verify)

    def t_no_nan_full_block():
        # block EVERY earlier position: must not produce NaN
        s = b.full(it, it.gold_cot)
        cid = b.candidate_ids(it)
        blocked = list(range(len(s.ids) - 1))
        sc = score_candidates(model, s, cid, DEV, blocked)
        assert torch.isfinite(sc).all(), "full knockout produced NaN/inf"
    check("total knockout stays finite (no all-masked softmax row)", t_no_nan_full_block)

    print("\n[3] candidate scoring")

    def t_scores():
        s = b.full(it, it.gold_cot)
        cid = b.candidate_ids(it)
        assert all(len(c) > 0 for c in cid), "candidate tokenised to nothing"
        sc = score_candidates(model, s, cid, DEV)
        assert sc.shape == (len(it.candidates),)
        assert sc.dtype == torch.float32
        assert torch.isfinite(sc).all()
        assert (sc <= 0).all(), "log-probs must be <= 0"
    check("scores are finite float32 log-probs, one per candidate", t_scores)

    def t_scoring_matches_unbatched():
        """The batched right-padded path must equal a naive per-candidate forward."""
        s = b.full(it, it.gold_cot)
        cid = b.candidate_ids(it)
        batched = score_candidates(model, s, cid, DEV)
        P = len(s.ids)
        naive = []
        with torch.no_grad():
            for c in cid:
                row = torch.tensor([s.ids + c], dtype=torch.long)
                lp = torch.log_softmax(model(input_ids=row).logits.float(), -1)
                tot = sum(lp[0, P - 1 + j, c[j]].item() for j in range(len(c)))
                naive.append(tot)
        naive_t = torch.tensor(naive)
        err = (batched - naive_t).abs().max().item()
        assert err < 1e-4, f"batched scoring disagrees with naive by {err:.3g}"
    check("batched+padded scoring == naive per-candidate scoring", t_scoring_matches_unbatched)

    def t_margin():
        sc = torch.tensor([0.0, -1.0, -2.0])
        m0 = margin(sc, 0)
        assert m0 > 0, "gold with highest score must have positive margin"
        m2 = margin(sc, 2)
        assert m2 < 0
        assert abs(m0 - (0.0 - torch.logsumexp(torch.tensor([-1.0, -2.0]), 0).item())) < 1e-6
    check("margin is signed and matches its definition", t_margin)

    print("\n[4] rho_CoT")

    def t_rho():
        s = b.full(it, it.gold_cot)
        cid = b.candidate_ids(it)
        r = rho_cot(model, s, cid, it.candidates.index(it.gold), DEV)
        for k in ("rho_cot", "delta_cot", "delta_rules", "margin_base"):
            assert k in r
        assert 0.0 <= r["rho_cot"] <= 1.0 or r["rho_cot"] != r["rho_cot"]
        assert all(isinstance(v, float) for v in r.values())
    check("rho_cot returns a bounded ratio and its components", t_rho)

    def t_rho_empty_cot_is_zero_effect():
        """With no CoT there is nothing to knock out, so delta_cot must be 0."""
        s = b.full(it, "")
        cid = b.candidate_ids(it)
        r = rho_cot(model, s, cid, 0, DEV)
        assert abs(r["delta_cot"]) < 1e-4, f"empty CoT gave delta={r['delta_cot']}"
    check("empty CoT -> delta_cot == 0 (sanity anchor)", t_rho_empty_cot_is_zero_effect)

    print("\n[5] logit lens / commitment depth")

    def t_dstar():
        s = b.full(it, it.gold_cot)
        cid = b.candidate_ids(it)
        d = commitment_depth(model, s, cid, it.candidates.index(it.gold), DEV)
        L = d["n_layers"]
        assert L == 3, f"expected 3 layers, got {L}"
        assert 0 <= d["d_star"] <= L
        assert 0.0 <= d["d_star_frac"] <= 1.0
        assert d["lens_correct"] in (0.0, 1.0)
    check("commitment depth is in [0, n_layers] with a valid fraction", t_dstar)

    def t_dstar_consistency():
        """d* must be the FIRST layer after which the prediction never changes."""
        s = b.full(it, it.gold_cot)
        cid = b.candidate_ids(it)
        firsts = [c[0] for c in cid]
        from cfd.engine import _final_norm_and_head
        norm, head = _final_norm_and_head(model)
        with torch.no_grad():
            out = model(input_ids=torch.tensor([s.ids]), output_hidden_states=True)
        sel = torch.tensor(firsts)
        preds = [int(head(norm(h[:, -1, :])).float()[0, sel].argmax().item())
                 for h in out.hidden_states]
        d = commitment_depth(model, s, cid, 0, DEV)
        k = int(d["d_star"])
        assert all(p == preds[-1] for p in preds[k:]), "d* not stable afterwards"
        if k > 0:
            assert preds[k - 1] != preds[-1], "d* is not minimal"
    check("d* is the minimal stable layer (recomputed independently)", t_dstar_consistency)

    print("\n[6] generation")

    def t_generate():
        outs = generate_cots(model, tok, ds_chain[:4], b, DEV,
                             max_new_tokens=12, batch_size=3)
        assert len(outs) == 4 and all(isinstance(o, str) for o in outs)
        assert tok.padding_side == "right", "padding_side must be restored"
    check("batched generation returns one string per item, restores padding", t_generate)

    def t_left_pad_equivalence():
        """Left-padded batch generation must equal single-item generation."""
        sub = ds_chain[:3]
        batched = generate_cots(model, tok, sub, b, DEV, max_new_tokens=10, batch_size=3)
        single = [generate_cots(model, tok, [x], b, DEV, max_new_tokens=10,
                                batch_size=1)[0] for x in sub]
        assert batched == single, ("left padding changed generation:\n"
                                   f"{batched}\nvs\n{single}")
    check("batch size does not change greedy generations", t_left_pad_equivalence)

    def t_clean():
        assert _clean_cot("Step 1: a.\nAnswer: b") == "Step 1: a."
        assert _clean_cot("\n\n  \n") == ""
        assert _clean_cot("x\n\ny") == "x\ny"
    check("_clean_cot strips the model's own answer and blank lines", t_clean)

    print("\n[7] degeneracy detector")

    def t_degen():
        good = "Step 1: a -> b.\nStep 2: b -> c.\nStep 3: c -> d."
        bad = "\n".join(["He must have sold 29 bags."] * 12)
        assert degeneracy(good)["is_degenerate"] == 0.0
        assert degeneracy(bad)["is_degenerate"] == 1.0
        assert degeneracy("")["is_degenerate"] == 1.0
    check("degeneracy flags looping chains and empties, not healthy ones", t_degen)

    print("\n[8] perturbations under the tokeniser")

    def t_perturb_spans():
        rng = random.Random(0)
        for x in ds_chain[:4] + ds_mod[:4]:
            for cot in (tasks.truncate_cot(x.gold_cot, 0.5),
                        tasks.filler_cot(x.gold_cot, 0.5),
                        tasks.shuffle_cot(x.gold_cot, rng),
                        tasks.corrupt_cot(x, x.gold_cot, 0, rng) or x.gold_cot):
                s = b.full(x, cot)
                st, en = s.spans["cot"]
                assert tok.decode(s.ids[st:en]) == cot
                sc = score_candidates(model, s, b.candidate_ids(x), DEV)
                assert torch.isfinite(sc).all()
    check("every perturbation still builds exact spans and scores finitely", t_perturb_spans)

    print("\n[9] full sweep on both tasks")

    def t_sweep():
        from cfd.metrics import evaluate_item
        # NaN is the CORRECT value for these on a random model: rho_cot is undefined
        # when neither knockout moves the margin, and nldd is undefined when the
        # chain accrues no margin at all. Both must be NaN, never a crash.
        may_be_nan = {"rho_cot", "nldd_mean", "err_prop_rate", "err_prop_delta"}
        required = ("acc", "acc_direct", "margin_base", "rho_cot", "d_star",
                    "d_star_frac", "err_prop_rate", "early_25", "early_50",
                    "early_75", "filler_50", "nldd_mean", "shuffled_gap",
                    "shuffled_acc_gap", "is_degenerate", "n_steps")
        for x in ds_chain[:3] + ds_mod[:3]:
            rec = evaluate_item(model, tok, b, x, x.gold_cot, DEV, seed=0)
            for k in required:
                assert k in rec, f"missing metric {k}"
                v = rec[k]
                assert isinstance(v, float), f"{k} is {type(v).__name__}, not float"
                if k not in may_be_nan:
                    assert v == v, f"{k} is NaN but must always be defined"
            assert rec["acc"] in (0.0, 1.0)
            assert rec["n_steps"] == float(len(x.hops))
    check("evaluate_item returns the full metric record on both tasks", t_sweep)

    def t_nldd_math():
        """Verify the accrual score against hand-computed cases."""
        from cfd.metrics import _nldd, aggregate
        # margin appears only at the very last step -> maximally faithful
        late = _nldd([0.0, 0.0, 0.0, 10.0])
        assert abs(late - 1.0) < 1e-9, f"late accrual should be 1.0, got {late}"
        # full margin present before any step -> fully post-hoc
        early = _nldd([10.0, 10.0, 10.0, 10.0])
        assert early != early, "flat profile must be NaN (no accrual to normalise)"
        # linear accrual -> mean deficit of 1, 2/3, 1/3  = 2/3
        lin = _nldd([0.0, 1.0, 2.0, 3.0])
        assert abs(lin - (1 + 2/3 + 1/3) / 3) < 1e-9, f"linear case wrong: {lin}"
        assert _nldd([1.0]) != _nldd([1.0]), "single point must be NaN"
        # aggregate must drop NaN rather than propagate it
        agg = aggregate([{"acc": 1.0, "nldd_mean": float("nan"), "is_degenerate": 0.0},
                         {"acc": 0.0, "nldd_mean": 0.5, "is_degenerate": 0.0}])
        assert abs(agg["nldd_mean"] - 0.5) < 1e-9, "NaN not dropped in aggregate"
        assert abs(agg["acc"] - 0.5) < 1e-9
        assert agg["n_kept"] == 2.0
    check("NLDD accrual maths matches hand-computed cases", t_nldd_math)

    def t_aggregate_degenerate():
        from cfd.metrics import aggregate, gate_report
        recs = [{"acc": 1.0, "err_prop_rate": 0.9, "is_degenerate": 0.0},
                {"acc": 1.0, "err_prop_rate": 0.0, "is_degenerate": 1.0}]
        agg = aggregate(recs)
        assert agg["degenerate_frac"] == 0.5
        assert agg["n_kept"] == 1.0
        assert agg["err_prop_rate"] == 0.9, "degenerate chain must not enter the mean"
        assert agg["acc_all"] == 1.0, "accuracy is reported over ALL items"
        rep = gate_report({"acc": 0.75, "cot_lift": 0.4, "err_prop_rate": 0.8,
                           "shuffled_acc_gap": 0.3, "degenerate_frac": 0.02})
        assert rep["pass"] is True, rep
        bad = gate_report({"acc": 0.99, "cot_lift": 0.01, "err_prop_rate": 0.1,
                           "shuffled_acc_gap": 0.0, "degenerate_frac": 0.5})
        assert bad["pass"] is False and sum(not c["ok"] for c in bad["checks"]) == 5
    check("aggregate excludes degenerate chains; gate accepts/rejects correctly",
          t_aggregate_degenerate)

    n_fail = sum(1 for _, ok, _ in _results if not ok)
    print("\n" + "=" * 66)
    print(f"{len(_results) - n_fail}/{len(_results)} checks passed")
    if n_fail:
        print("\nFAILURES:")
        for name, ok, msg in _results:
            if not ok:
                print(f"  - {name}: {msg}")
    print("=" * 66)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
