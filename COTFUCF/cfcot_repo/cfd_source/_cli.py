
# ==========================================================================
# environment preflight
# ==========================================================================


def preflight(verbose: bool = True) -> List[str]:
    bad: List[str] = []
    import transformers
    if verbose:
        print("=" * 66)
        print(f"  torch        {torch.__version__}")
        print(f"  transformers {transformers.__version__}")
    tv = tuple(int(x) for x in transformers.__version__.split(".")[:2])
    if tv < (4, 44):
        bad.append(f"transformers {transformers.__version__} too old (need >=4.44)")

    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        if verbose:
            print(f"  GPUs         {n}")
            for i in range(n):
                p = torch.cuda.get_device_properties(i)
                print(f"    [{i}] {p.name}  {p.total_memory/1e9:.1f} GB  sm_{p.major}{p.minor}")
            print(f"  bf16         {torch.cuda.is_bf16_supported()}"
                  + ("" if torch.cuda.is_bf16_supported()
                     else "  -> fp16 base + float32 LoRA (expected on T4)"))
        free = torch.cuda.mem_get_info(0)[0] / 1e9
        if verbose:
            print(f"  free VRAM    {free:.1f} GB on GPU 0")
        if free < 7:
            bad.append(f"only {free:.1f} GB free on GPU 0; a 1.5B fp16 run needs ~8 GB")
        if n < 2 and verbose and ON_KAGGLE:
            print("  NOTE: one GPU visible. Set Accelerator = GPU T4 x2 -- two GPUs "
                  "cost the same quota as one.")
    elif verbose:
        print("  GPUs         none (self-checks still run; real jobs will not)")

    try:
        u = shutil.disk_usage(_workdir())
        if verbose:
            print(f"  disk         {u.free/1e9:.1f} GB free at {_workdir()}")
        if u.free / 1e9 < 12:
            bad.append("under 12 GB free; model (~3.5 GB) + checkpoints "
                       "(~1.7 GB/condition) will not fit")
    except OSError:
        pass
    if verbose:
        print("=" * 66)
    return bad


# ==========================================================================
# offline self-checks (tiny random model + byte-level stub tokeniser)
# ==========================================================================


class StubTok:
    OFFSET = 4

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
        bs = bytes(i - self.OFFSET for i in ids
                   if isinstance(i, int) and self.OFFSET <= i < self.OFFSET + 256)
        return bs.decode("utf-8", errors="ignore")

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
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


def run_selftest(verbose: bool = True) -> int:
    res: List[Tuple[str, bool, str]] = []

    def check(name, fn):
        try:
            fn()
            res.append((name, True, ""))
            if verbose:
                print(f"  ok    {name}")
        except Exception as e:                                    # noqa: BLE001
            res.append((name, False, f"{type(e).__name__}: {e}"))
            print(f"  FAIL  {name}\n        {type(e).__name__}: {e}")
            if verbose:
                traceback.print_exc(limit=3)

    D = "cpu"
    tok, b = StubTok(), None
    model = tiny_model()
    b = Builder(tok)
    ds = require_distinct_first_tokens(
        b, build_dataset("chain", 12, seed=3, hops=3, distractors=3))
    dm = require_distinct_first_tokens(
        b, build_dataset("modarith", 12, seed=3, hops=3, modulus=11))
    it = ds[0]

    def t_task_invariants():
        for x in ds + dm:
            assert x.gold in x.candidates
            assert len(set(x.candidates)) == len(x.candidates)
            assert x.hops[-1] == x.gold
            assert len(split_steps(x.gold_cot)) == len(x.hops)
        a = build_dataset("chain", 20, seed=7)
        c = build_dataset("chain", 20, seed=7)
        assert [x.to_dict() for x in a] == [x.to_dict() for x in c]
    check("tasks: invariants hold and generation is deterministic", t_task_invariants)

    def t_perturb():
        rng = random.Random(0)
        for x in ds:
            bad = corrupt_cot(x, x.gold_cot, 0, rng)
            assert bad is not None and bad != x.gold_cot
            assert len(split_steps(bad)) == len(split_steps(x.gold_cot))
        s = shuffle_cot(it.gold_cot, random.Random(3))
        assert sorted(split_steps(s)) == sorted(split_steps(it.gold_cot)) and s != it.gold_cot
        assert len(split_steps(filler_cot(it.gold_cot, 0.5))) == len(it.hops)
    check("tasks: corruption, shuffling and filler preserve step count", t_perturb)

    def t_spans():
        s = b.full(it, it.gold_cot)
        prev = 0
        for nm in SPAN_ORDER:
            st, en = s.spans[nm]
            assert st == prev and en >= st, f"gap/overlap at {nm}"
            prev = en
        assert prev == len(s.ids)
        st, en = s.spans["rules"]
        assert tok.decode(s.ids[st:en]) == it.rules_text
        st, en = s.spans["cot"]
        assert tok.decode(s.ids[st:en]) == it.gold_cot
    check("spans tile the sequence exactly and decode round-trips", t_spans)

    def t_empty_and_notemplate():
        s = b.full(it, "")
        assert s.spans["cot"][0] == s.spans["cot"][1]
        assert s.span_positions("cot") == []
        b2 = Builder(StubTok(with_template=False))
        s2 = b2.full(it, it.gold_cot)
        assert s2.spans["bridge"][1] == len(s2.ids)
    check("empty CoT and missing chat template both build legal sequences",
          t_empty_and_notemplate)

    def t_mask():
        m = build_mask([5, 7], 7, torch.float32, [[1], []], "cpu")
        neg = torch.finfo(torch.float32).min
        assert m.shape == (2, 1, 7, 7)
        assert m[0, 0, 3, 1] == neg and m[0, 0, 3, 0] == 0.0
        assert m[0, 0, 1, 3] == neg and m[0, 0, 4, 5] == neg
        assert m[1, 0, 6, 5] == 0.0
        for bb in range(2):
            for i in range(7):
                assert m[bb, 0, i, i] == 0.0
    check("mask is causal, pad-aware, knockout-aware and NaN-safe", t_mask)

    check("4-D attention masks are honoured by this transformers build",
          lambda: verify_knockout(model, D))

    def t_no_nan():
        s = b.full(it, it.gold_cot)
        sc = score_candidates(model, s, b.candidate_ids(it), D,
                              list(range(len(s.ids) - 1)))
        assert torch.isfinite(sc).all()
    check("total knockout stays finite (no all-masked softmax row)", t_no_nan)

    def t_scores():
        s = b.full(it, it.gold_cot)
        cid = b.candidate_ids(it)
        sc = score_candidates(model, s, cid, D)
        assert sc.shape == (len(it.candidates),) and sc.dtype == torch.float32
        assert torch.isfinite(sc).all() and (sc <= 0).all()
    check("scores are finite float32 log-probs, one per candidate", t_scores)

    def t_scoring_matches_manual():
        """One-row first-token scoring must equal a hand-computed log-softmax."""
        x = require_distinct_first_tokens(b, [it])[0]
        s = b.full(x, x.gold_cot)
        cid = b.candidate_ids(x)
        fast = score_candidates(model, s, cid, D)
        with torch.no_grad():
            lp = torch.log_softmax(
                model(input_ids=torch.tensor([s.ids])).logits[0, -1].float(), -1)
        manual = torch.tensor([lp[c[0]].item() for c in cid])
        err = (fast - manual).abs().max().item()
        assert err < 1e-5, f"scoring disagrees with manual by {err:.3g}"
        assert fast.shape == (len(x.candidates),)

    check("one-forward first-token scoring matches a manual log-softmax",
          t_scoring_matches_manual)

    def t_collision_refused():
        """Scoring must refuse colliding options rather than return nonsense."""
        bad = Item(task="t", rules_text="r", question_text="q", gold="alpha",
                   candidates=["alpha", "alphb"], hops=["alpha"], gold_cot="s")
        ids = b.candidate_ids(bad)
        if len({c[0] for c in ids}) == len(ids):
            return                       # this tokeniser happens not to collide
        try:
            score_candidates(model, b.full(bad, "s"), ids, D)
            raise AssertionError("colliding first tokens were not refused")
        except ValueError:
            pass
    check("colliding option first-tokens are refused, not silently mis-scored",
          t_collision_refused)

    def t_margin():
        sc = torch.tensor([0.0, -1.0, -2.0])
        assert margin(sc, 0) > 0 and margin(sc, 2) < 0
    check("margin is signed and matches its definition", t_margin)

    def t_rho():
        s = b.full(it, it.gold_cot)
        r = rho_cot(model, s, b.candidate_ids(it), it.candidates.index(it.gold), D)
        assert all(isinstance(v, float) for v in r.values())
        e = rho_cot(model, b.full(it, ""), b.candidate_ids(it), 0, D)
        assert abs(e["delta_cot"]) < 1e-4, "empty CoT must have zero knockout effect"
    check("rho_cot is well formed; empty CoT anchors delta to zero", t_rho)

    def t_dstar():
        s = b.full(it, it.gold_cot)
        cid = b.candidate_ids(it)
        d = commitment_depth(model, s, cid, it.candidates.index(it.gold), D)
        L = int(d["n_layers"])
        assert L == 3 and 0 <= d["d_star"] <= L and 0.0 <= d["d_star_frac"] <= 1.0
        norm, head = _final_norm_and_head(model)
        with torch.no_grad():
            out = model(input_ids=torch.tensor([s.ids]), output_hidden_states=True)
        sel = torch.tensor([c[0] for c in cid])
        preds = [int(head(norm(h[:, -1, :])).float()[0, sel].argmax().item())
                 for h in out.hidden_states]
        k = int(d["d_star"])
        assert all(p == preds[-1] for p in preds[k:])
        if k > 0:
            assert preds[k - 1] != preds[-1], "d* is not minimal"
    check("d* is the minimal stable logit-lens layer (recomputed independently)",
          t_dstar)

    def t_gen():
        outs = generate_cots(model, tok, ds[:4], b, D, max_new_tokens=12, batch_size=3)
        assert len(outs) == 4 and tok.padding_side == "right"
        sub = ds[:3]
        bt = generate_cots(model, tok, sub, b, D, max_new_tokens=10, batch_size=3)
        sg = [generate_cots(model, tok, [x], b, D, max_new_tokens=10, batch_size=1)[0]
              for x in sub]
        assert bt == sg, "left padding changed greedy generation"
    check("batch size does not change greedy generation", t_gen)

    def t_degen():
        assert degeneracy("Step 1: a.\nStep 2: b.\nStep 3: c.")["is_degenerate"] == 0.0
        assert degeneracy("\n".join(["He sold 29 bags."] * 12))["is_degenerate"] == 1.0
        assert degeneracy("")["is_degenerate"] == 1.0
        assert _clean_cot("Step 1: a.\nAnswer: b") == "Step 1: a."
    check("degeneracy flags loops and empties but not healthy chains", t_degen)

    def t_eval():
        may_nan = {"rho_cot", "nldd_mean", "err_prop_rate", "err_prop_delta"}
        req = ("acc", "acc_direct", "margin_base", "rho_cot", "d_star", "d_star_frac",
               "err_prop_rate", "early_25", "early_50", "early_75", "filler_50",
               "nldd_mean", "shuffled_gap", "shuffled_acc_gap", "is_degenerate")
        for x in ds[:3] + dm[:3]:
            rec = evaluate_item(model, tok, b, x, x.gold_cot, D, seed=0)
            for k in req:
                assert k in rec and isinstance(rec[k], float), k
                if k not in may_nan:
                    assert rec[k] == rec[k], f"{k} is NaN"
    check("evaluate_item returns the full record on both task families", t_eval)

    def t_nldd():
        assert abs(_nldd([0., 0., 0., 10.]) - 1.0) < 1e-9
        assert _nldd([10., 10., 10., 10.]) != _nldd([10., 10., 10., 10.])
        assert abs(_nldd([0., 1., 2., 3.]) - (1 + 2/3 + 1/3) / 3) < 1e-9
        agg = aggregate([{"acc": 1., "nldd_mean": float("nan"), "is_degenerate": 0.},
                         {"acc": 0., "nldd_mean": .5, "is_degenerate": 0.}])
        assert abs(agg["nldd_mean"] - .5) < 1e-9 and agg["n_kept"] == 2.
    check("NLDD accrual maths matches hand-computed cases", t_nldd)

    def t_agg_gate():
        agg = aggregate([{"acc": 1., "err_prop_rate": .9, "is_degenerate": 0.},
                         {"acc": 1., "err_prop_rate": 0., "is_degenerate": 1.}])
        assert agg["degenerate_frac"] == .5 and agg["n_kept"] == 1.
        assert agg["err_prop_rate"] == .9 and agg["acc_all"] == 1.
        assert gate_report({"acc": .75, "cot_lift": .4, "err_prop_rate": .8,
                            "shuffled_acc_gap": .3, "degenerate_frac": .02,
                            "rho_cot": .5})["pass"]
        bad = gate_report({"acc": .99, "cot_lift": .01, "err_prop_rate": .1,
                           "shuffled_acc_gap": 0., "degenerate_frac": .5,
                           "rho_cot": .02, "truncated_frac": .9})
        assert not bad["pass"]
        failed = {c["name"] for c in bad["checks"] if not c["ok"]}
        assert failed == {"chains not truncated", "accuracy in band", "CoT lift",
                          "error propagation", "CoT causal share",
                          "not degenerate"}, failed
        # the shuffled-CoT gap is reported as a warning, never as a gate condition
        assert bad["warnings"], "low shuffle gap should still be flagged"
        assert not any(c["name"] == "shuffled-CoT gap" for c in bad["checks"])
        # everything healthy except the primary metric -> must still fail
        thin = gate_report({"acc": .75, "cot_lift": .4, "err_prop_rate": .8,
                            "shuffled_acc_gap": .3, "degenerate_frac": .02,
                            "rho_cot": .05})
        assert not thin["pass"], "gate passed with no headroom in rho_CoT"
    check("aggregate drops degenerate chains; gate accepts and rejects correctly",
          t_agg_gate)

    def t_lora():
        m2 = tiny_model(vocab=320)
        ids = torch.randint(1, 300, (1, 16))
        with torch.no_grad():
            base = m2(input_ids=ids).logits.clone()
        w = inject_lora(m2, r=4, alpha=8)
        with torch.no_grad():
            assert (base - m2(input_ids=ids).logits).abs().max() < 1e-6
        assert all(p.dtype == torch.float32 for p in lora_parameters(w))
        with torch.no_grad():
            w[list(w)[0]].B.normal_(0, .05)
            trained = m2(input_ids=ids).logits.clone()
        w2 = inject_lora(m2, r=4, alpha=8)
        assert all(w[k] is w2[k] for k in w), "re-wrapped: nested LoRA"
        assert sum(1 for _ in m2.modules() if isinstance(_, LoRALinear)) == len(w)
        with torch.no_grad():
            assert (m2(input_ids=ids).logits - trained).abs().max() < 1e-6
        inject_lora(m2, r=4, alpha=8, reset=True)
        with torch.no_grad():
            assert (m2(input_ids=ids).logits - base).abs().max() < 1e-6
        try:
            inject_lora(m2, r=8, alpha=16)
            raise AssertionError("rank-change guard did not fire")
        except RuntimeError:
            pass
    check("LoRA: zero-init no-op, idempotent injection, rank guard", t_lora)

    def t_resume():
        data = synthetic_domain(64, seed=0)
        cfg = TrainCfg(steps=20, ckpt_every=5, batch_size=2, grad_accum=1,
                       warmup=3, seed=0, r=4, alpha=8)
        a_dir, b_dir = "/tmp/_cfd_a", "/tmp/_cfd_b"
        for d in (a_dir, b_dir):
            shutil.rmtree(d, ignore_errors=True)
        ma = tiny_model(vocab=320)
        train(ma, tok, data, a_dir, cfg, D, verbose=False)
        mb = tiny_model(vocab=320)
        wb, _ = train(mb, tok, data, b_dir, cfg, D, verbose=False,
                      max_steps_this_session=10)
        train(mb, tok, data, b_dir, cfg, D, wrapped=wb, resume=True, verbose=False)
        sa = torch.load(a_dir + "/ckpt_000020.pt", map_location="cpu",
                        weights_only=False)["lora"]
        sb = torch.load(b_dir + "/ckpt_000020.pt", map_location="cpu",
                        weights_only=False)["lora"]
        err = max((sa[k] - sb[k]).abs().max().item() for k in sa)
        assert err == 0.0, f"resume is not bit-exact (max diff {err:.3g})"
        try:
            train(mb, tok, data, b_dir, TrainCfg(steps=99, ckpt_every=5, batch_size=2,
                  grad_accum=1, warmup=3, seed=0, r=4, alpha=8), D,
                  wrapped=wb, resume=True, verbose=False)
            raise AssertionError("cfg guard did not fire")
        except RuntimeError as e:
            assert "refusing to resume" in str(e)
    check("training resume is bit-exact and the config guard fires", t_resume)

    def t_sae():
        seqs = [b.full(x, x.gold_cot) for x in ds[:6]]
        acts = collect_activations(model, seqs, 1, D)
        sae, st = train_sae(acts, n_latents=128, k=8, steps=250, batch=64,
                            device=D, verbose=False)
        assert st["fvu"] < 1.0
        assert (sae.W_dec.norm(dim=0) - 1).abs().max().item() < 1e-4
        worst = max(splice_is_identity(model, sae, 1, s, D) for s in seqs)
        assert worst < 1e-3, f"splice is not an identity ({worst:.3g})"
        with torch.no_grad():
            with Splice(model, sae, 1) as sp:
                model(input_ids=torch.tensor([seqs[0].ids]))
            assert int((sp.z[0] > 0).sum(-1).max()) <= 8
    check("SAE trains, decoder is unit-norm, splice is a numerical identity", t_sae)

    def t_attr():
        seqs = [b.full(x, x.gold_cot) for x in ds[:6]]
        acts = collect_activations(model, seqs, 1, D)
        sae, _ = train_sae(acts, n_latents=128, k=8, steps=200, batch=64,
                           device=D, verbose=False)
        s = seqs[0]
        cid, gi = b.candidate_ids(ds[0]), ds[0].candidates.index(ds[0].gold)
        pos = s.span_positions("cot")
        a1 = attribution_ie(model, sae, 1, s, cid, gi, D, pos, scale=1.0)
        a2 = attribution_ie(model, sae, 1, s, cid, gi, D, pos, scale=2.0 ** 14)
        assert (a1 - a2).abs().max().item() < 1e-6, "grad scaling changed the result"
        assert int((a1 != 0).sum()) > 0, "no gradient reached the latents"
        try:
            attribution_ie(model, sae, 1, s, cid, gi, D, pos, scale=1e-300)
            raise AssertionError("underflow guard did not fire")
        except RuntimeError:
            pass
        top = torch.topk(a1.abs(), 5).indices.tolist()
        ex = exact_ie(model, sae, 1, s, cid, gi, D, top, pos)
        assert any(abs(v) > 1e-8 for v in ex.values()), "exact ablation did nothing"
    check("attribution is grad-scale invariant; fp16 underflow raises, not zeros",
          t_attr)

    def t_dd():
        seqs = [b.full(x, x.gold_cot) for x in ds[:6]]
        acts = collect_activations(model, seqs, 1, D)
        sae, _ = train_sae(acts, n_latents=128, k=8, steps=200, batch=64,
                           device=D, verbose=False)
        cots = [x.gold_cot for x in ds]
        part = partition_latents(model, sae, 1, b, ds[:3], cots[:3], D,
                                 seqs[3:], top_k=12)
        assert len(part.med) == 12 and len(part.gen) == 12
        assert not (set(part.med) & set(part.gen)), "med and gen must be disjoint"
        dd = double_dissociation(model, sae, 1, b, ds[:3], cots[:3], part, D)
        assert set(dd) == {"control", "ablate_med", "ablate_gen", "ablate_random"}
        assert all(v["fluency"] == v["fluency"] for v in dd.values())
        import copy
        bad = copy.copy(part)
        bad.gen = []
        try:
            double_dissociation(model, sae, 1, b, ds[:3], cots[:3], bad, D)
            raise AssertionError("empty-arm guard did not fire")
        except ValueError:
            pass
    check("partition is non-empty and disjoint; empty ablation arms are refused",
          t_dd)

    def t_retention():
        v = retention_nll(model, tok, D)
        assert isinstance(v, float) and v == v and v > 0, v
        g = cot_format_nll(model, tok, b, ds[:4], D, limit=4)
        assert isinstance(g, float) and g == g and g > 0, g
    check("retention and CoT-format NLL are finite positive per-token values",
          t_retention)

    def t_dissoc_random():
        base = {"faithfulness": 1.0, "fluency": -1.0, "accuracy": 1.0, "n_ablated": 0.}
        good = {"control": base,
                "ablate_med": {**base, "faithfulness": 0.1},
                "ablate_gen": {**base, "fluency": -3.0},
                "ablate_random": {**base, "faithfulness": .95, "fluency": -1.05}}
        assert dissociation_test(good)["dissociation"] is True
        # med no better than a matched random set -> not a dissociation
        weak = {**good, "ablate_random": {**base, "faithfulness": 0.05,
                                          "fluency": -4.0}}
        t = dissociation_test(weak)
        assert t["dissociation"] is False and t["beats_random"] is False
    check("dissociation requires beating a count-matched random ablation",
          t_dissoc_random)

    def t_fadg():
        st = fadg([0, 10, 20, 30, 40], [1., .9, .7, .6, .5], [1., 1., .95, .9, .75], .2)
        assert st["step_faith"] == 20, st
        assert st["step_acc"] == 40, st
        assert st["fadg"] == 20 and st["monitorability_half_life"] == 40
        n = fadg([0, 10], [1., 1.], [1., 1.], .2)
        assert n["fadg"] != n["fadg"], "no crossing must be NaN, not 0"
        # a single spurious dip must NOT be reported as the crossing point
        noisy = crossing([0, 10, 20, 30, 40], [1., .5, 1., 1., 1.], 0.8)
        assert noisy != noisy, f"one-point dip counted as a crossing ({noisy})"
        real = crossing([0, 10, 20, 30, 40], [1., .5, .5, 1., 1.], 0.8)
        assert real == 10, f"sustained crossing missed ({real})"
        # a crossing at the very last checkpoint is still a crossing
        assert crossing([0, 10, 20], [1., 1., .5], 0.8) == 20
        # THE STRONGEST CASE: faithfulness collapses, accuracy never moves.
        # This must not report NaN -- that would make the best possible result
        # indistinguishable from nothing having happened.
        S = [0, 40, 80, 120, 160]
        best = fadg(S, [1., .9, .6, .5, .45], [1., 1., .99, .98, .97], .2)
        assert best["fadg"] == best["fadg"], "strongest result reported as NaN"
        assert best["fadg"] > 0 and best["fadg_censored"], best
        assert best["step_faith"] == 80
        # both flat -> genuinely nothing, still NaN
        none = fadg(S, [1.] * 5, [1.] * 5, .2)
        assert none["fadg"] != none["fadg"]
    check("FADG ignores single-point noise but catches sustained crossings", t_fadg)

    def t_dedupe():
        x = ds[0]
        fixed = dedupe_candidates(b, x)
        assert fixed is None or fixed.gold in fixed.candidates
        if fixed is not None:
            ids = b.candidate_ids(fixed)
            assert len({c[0] for c in ids}) == len(ids), "collisions survived"
            assert len(fixed.candidates) >= 2
            assert fixed.hops == x.hops and fixed.gold_cot == x.gold_cot
    check("candidate deduplication keeps gold and removes first-token collisions",
          t_dedupe)

    nf = sum(1 for _, ok, _ in res if not ok)
    print("=" * 66)
    print(f"  {len(res)-nf}/{len(res)} self-checks passed")
    if nf:
        for n, ok, m in res:
            if not ok:
                print(f"    - {n}: {m}")
    print("=" * 66)
    return 1 if nf else 0


# ==========================================================================
# commands
# ==========================================================================


def _items(task, n, seed, hops, dis, modulus, builder=None):
    kw = ({"hops": hops, "distractors": dis} if task == "chain"
          else {"hops": hops, "modulus": modulus})
    items = build_dataset(task, n, seed=seed, **kw)
    # Scoring reads one position and compares option first-tokens, so options that
    # collide there are not scoreable. Repair or drop them once, up front.
    return require_distinct_first_tokens(builder, items) if builder else items


def _eval_set(model, tok, builder, items, device, mnt, gb, seed,
              with_retention: bool = False):
    cots = generate_cots(model, tok, items, builder, device,
                         max_new_tokens=mnt, batch_size=gb)
    recs = [evaluate_item(model, tok, builder, it, c, device, seed=seed)
            for it, c in zip(items, cots)]
    agg = aggregate(recs)
    agg["truncated_frac"] = (sum(TRUNCATED) / len(TRUNCATED)) if TRUNCATED else 0.0
    if with_retention:
        agg["nll_generic"] = retention_nll(model, tok, device)
        agg["nll_gold_cot"] = cot_format_nll(model, tok, builder, items, device)
    return agg, cots


def estimate_runtime(model, tok, builder, a, n_probe: int = 6) -> Dict[str, float]:
    """Time a handful of items on THIS machine and project the whole pipeline.

    Guessing from FLOPs is unreliable on a T4 (low MFU, Python overhead per call),
    so measure instead. Costs about a minute and prevents starting a run that
    cannot finish inside the session wall.
    """
    items = _items("chain", n_probe, a.seed, 3, 3, 11, builder)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    cots = generate_cots(model, tok, items, builder, dev,
                         max_new_tokens=a.max_new_tokens, batch_size=a.gen_batch)
    t_gen = (time.time() - t0) / n_probe
    t0 = time.time()
    for it, c in zip(items, cots):
        evaluate_item(model, tok, builder, it, c, dev, seed=0)
    t_eval = (time.time() - t0) / n_probe
    per_item = t_gen + t_eval
    n = a.n
    n_ckpt = a.steps // a.ckpt_every + 1
    n_gate_cells = len(a.hops.split(",")) * len(a.distractors.split(",")) + \
        len(a.hops.split(","))                      # chain cells + modarith cells
    # NOT torch.cuda.device_count(): the probe runs as a child spawned with
    # CUDA_VISIBLE_DEVICES pinned to one card, so it would always see 1 and the
    # estimate would be pessimistic by exactly the parallelism factor.
    ngpu = max(1, int(getattr(a, "ngpu", 0)) or torch.cuda.device_count())
    n_models = len(a.gate_models.split(","))
    n_cond = len(a.conditions.split(","))

    gate = n_gate_cells * n * per_item * math.ceil(n_models / ngpu)
    sweep = n_ckpt * n * per_item * math.ceil(n_cond / ngpu)
    train_s = a.steps * 0.45 * math.ceil(n_cond / ngpu)
    fig2 = a.sae_items * 0.03 + 120 + a.n_dd * 12 * per_item / 16
    total = gate + train_s + sweep + fig2
    return {"per_item": per_item, "t_gen": t_gen, "t_eval": t_eval,
            "gate_h": gate / 3600, "train_h": train_s / 3600,
            "sweep_h": sweep / 3600, "fig2_h": fig2 / 3600,
            "total_h": total / 3600, "n_ckpt": float(n_ckpt),
            "n_gate_cells": float(n_gate_cells)}


def cmd_eta(a) -> int:
    model, tok, dev = load_model(a.model or a.gate_models.split(",")[0], "cuda",
                                 verify=False)
    e = estimate_runtime(model, tok, Builder(tok), a)
    print("\n" + "=" * 66)
    print(f"  measured   {e['t_gen']:.2f}s generate + {e['t_eval']:.2f}s evaluate "
          f"= {e['per_item']:.2f}s per item")
    print(f"  gate       {e['gate_h']:5.2f} h   "
          f"({int(e['n_gate_cells'])} cells x {a.n} items)")
    print(f"  train      {e['train_h']:5.2f} h")
    print(f"  sweep      {e['sweep_h']:5.2f} h   "
          f"({int(e['n_ckpt'])} checkpoints x {a.n} items)")
    print(f"  figure 2   {e['fig2_h']:5.2f} h")
    ngpu = max(1, int(getattr(a, "ngpu", 0)) or torch.cuda.device_count())
    print(f"  TOTAL      {e['total_h']:5.2f} h on {ngpu} GPU(s)")
    print("=" * 66)
    if ngpu < 2:
        print("\n  ONE GPU. Kaggle bills session hours, not per-GPU hours, so extra")
        print("  cards are free throughput. Pick a multi-GPU accelerator (T4 x2 or")
        print("  L4 x4) in the notebook settings and this number divides by that.")
    budget = 8.0
    if e["total_h"] > budget:
        # solve for the item count that fits, keeping at least 12 checkpoints so
        # FADG still has the resolution to locate a crossing
        scale = budget / e["total_h"]
        n_fit = max(40, int(a.n * scale ** 0.5))
        ck_fit = a.ckpt_every
        while a.steps // ck_fit + 1 > 12 and (n_fit / a.n) * (a.ckpt_every / ck_fit) > scale:
            ck_fit += 10
        print(f"\n  To fit roughly {budget:.0f} h, run:")
        print(f"    python cfd_all.py pipeline --n {n_fit} --ckpt-every {ck_fit}")
        print(f"    ({a.steps // ck_fit + 1} checkpoints -- do not go below 12, or")
        print("     FADG cannot locate where the two curves cross)")
    return 0


def cmd_figures(a) -> int:
    """Rebuild every figure and table from results already on disk. No GPU."""
    stats = build_all(a.root, delta=a.delta)
    print(json.dumps(stats, indent=2))
    for f in sorted(os.listdir(a.root)):
        if f.endswith((".png", ".csv", ".md")):
            print("  " + f)
    return 0


def cmd_check(a) -> int:
    bad = preflight()
    rc = run_selftest()
    for x in bad:
        print(f"  WARN  {x}")
    return rc


def cmd_gate(a) -> int:
    model, tok, dev = load_model(a.model, a.device)
    builder = Builder(tok)
    rows, ok = [], False
    for task in a.tasks.split(","):
        for hops in [int(x) for x in a.hops.split(",")]:
            for dis in [int(x) for x in a.distractors.split(",")]:
                if task == "modarith" and dis != 0:
                    continue
                agg, _ = _eval_set(model, tok, builder,
                                   _items(task, a.n, a.seed, hops, dis, a.modulus, builder),
                                   dev, a.max_new_tokens, a.gen_batch, a.seed)
                rep = gate_report(agg)
                ok |= rep["pass"]
                rows.append({"model": a.model, "task": task, "hops": hops,
                             "distractors": dis, "gate_pass": rep["pass"],
                             **{k: v for k, v in agg.items() if not k.endswith("_sem")}})
                print(f"  [{'PASS' if rep['pass'] else 'fail'}] {task} h={hops} d={dis}"
                      f"  acc={agg.get('acc', float('nan')):.3f}"
                      f"  rho={agg.get('rho_cot', float('nan')):.3f}"
                      f"  lift={agg.get('cot_lift', float('nan')):+.3f}"
                      f"  err={agg.get('err_prop_rate', float('nan')):.3f}"
                      f"/{agg.get('err_prop_line', float('nan')):.2f}"
                      f"  shuf={agg.get('shuffled_acc_gap', float('nan')):+.3f}"
                      f"  degen={agg.get('degenerate_frac', float('nan')):.2f}"
                      f"  steps={agg.get('n_steps', float('nan')):.1f}"
                      f"  len={agg.get('n_tokens', float('nan')):.0f}"
                      f"  TRUNC={agg.get('truncated_frac', 0.0):.0%}",
                      flush=True)
                for w in rep.get("warnings", []):
                    print(f"           ~ {w}")
                if not rep["pass"]:
                    trunc = agg.get("truncated_frac", 0.0)
                    if trunc > 0.15:
                        print(f"           - TRUNCATED ({trunc:.0%}). Every other "
                              "number in this row is measured on a fragment and "
                              "means nothing. Re-run with a larger "
                              "--max-new-tokens; do not change anything else.")
                    else:
                        for ch in rep["checks"]:
                            if not ch["ok"]:
                                print(f"           - {ch['name']}: {ch['detail']}")
                write_jsonl(rows, os.path.join(a.out, "gate.jsonl"))
    if not ok:
        tr = [r.get("truncated_frac", 0.0) for r in rows]
        worst_trunc = max(tr) if tr else 0.0
        print("\nNO CELL PASSED -- do not start training.")
        if worst_trunc > 0.15:
            print(f"  The dominant problem is truncation (up to {worst_trunc:.0%} of "
                  "chains hit the token budget).\n"
                  "  Re-run with a larger --max-new-tokens and change NOTHING else. "
                  "Every\n  other number is measured on fragments and cannot be "
                  "interpreted yet.")
        else:
            best = max(rows, key=lambda r: r.get("shuffled_acc_gap", -9))
            print(f"  Chains are complete, so the numbers are real: this model does "
                  f"not use\n  its chain on these tasks. Closest cell was "
                  f"{best['task']} h={best['hops']} d={best['distractors']} "
                  f"(shuf={best.get('shuffled_acc_gap', float('nan')):+.2f}).\n"
                  "  Try an easier setting (lower hops, --modulus 7) before a "
                  "larger model.")
        return 2
    best = max((r for r in rows if r["gate_pass"]),
               key=lambda r: r.get("shuffled_acc_gap", 0.0))
    print(f"\nBEST CELL  task={best['task']} hops={best['hops']} "
          f"distractors={best['distractors']}")
    with open(os.path.join(a.out, "best_cell.json"), "w") as fh:
        json.dump(best, fh, indent=2)
    return 0


def _domain_for(cond, n, seed, jsonl):
    # the protected run is ood_cot with a masked gradient: same data, same schedule
    cond = "ood_cot" if cond == "protected" else cond
    if jsonl:
        base, note = load_jsonl(jsonl), f"from {jsonl}"
    else:
        base = synthetic_domain(n, seed=seed, answer_only=not cond.endswith("_cot"))
        note = "synthetic clinical domain"
    if cond == "random_label":
        return randomise_labels(base, seed=seed), note + ", labels shuffled"
    if cond == "replay_generic":
        # Suppress general drift WITHOUT any reasoning-task supervision. This is
        # the control that answers the obvious objection to `replay`: that mixing
        # the task back in preserves the reasoning circuits directly rather than
        # by preventing forgetting. If faithfulness survives here too, the
        # mechanism is general forgetting, not task-specific retraining.
        rep = generic_replay(max(1, n // 4), seed=seed)
        mixed = base + rep
        random.Random(seed).shuffle(mixed)
        return mixed, note + f", +{len(rep)} generic-text replay"
    if cond == "replay":
        its = build_dataset("chain", max(1, n // 4), seed=seed + 99, hops=3)
        rep = [Example(f"{i.rules_text}\n\n{i.question_text}\nAnswer:", f" {i.gold}")
               for i in its]
        mixed = base + rep
        random.Random(seed).shuffle(mixed)
        return mixed, note + f", +{len(rep)} replay"
    return base, note


def cmd_train(a) -> int:
    model, tok, dev = load_model(a.model, a.device, verify=False)
    base_cond = "ood_cot" if a.cond == "protected" else a.cond
    data, note = _domain_for(base_cond, a.n_train, a.seed, a.domain_jsonl)
    print(f"[data  ] {a.cond}: {len(data)} examples ({note})")
    cfg = TrainCfg(steps=a.steps, ckpt_every=a.ckpt_every, batch_size=a.batch_size,
                   grad_accum=a.grad_accum, lr=a.lr, seed=a.seed, r=a.rank,
                   alpha=2 * a.rank)

    gm = None
    if a.cond == "protected":
        # C3: identical data and schedule to ood_cot, but the mediator subspace is
        # projected out of every LoRA update that writes to the residual stream.
        # This is the condition that turns an analysis into a method: if a small
        # targeted mask closes the silent window at a lower target-task cost than
        # replay, the mediators were not just correlated with faithfulness, they
        # were carrying it.
        if not a.protect_from or not os.path.exists(a.protect_from):
            raise SystemExit(
                "--cond protected needs --protect-from <run>/sae_partition.pt.\n"
                "Run `fig2` on the ood_cot run first; it writes that file.")
        blob = torch.load(a.protect_from, map_location="cpu", weights_only=False)
        sd, med = blob["sae"], blob["med"]
        basis = sd["W_dec"][:, torch.tensor(med, dtype=torch.long)].to(torch.float32)
        wrapped = inject_lora(model, a.rank, 2 * a.rank)
        gm = SubspaceGradMask(wrapped, basis, d_model_of(model))
        print(f"[protect] masking a {basis.shape[1]}-dim mediator subspace across "
              f"{len(gm.targets)} residual-writing LoRA modules")
        train(model, tok, data, a.run, cfg, dev, wrapped=wrapped, grad_mask=gm,
              max_seconds=a.max_seconds)
        return 0

    train(model, tok, data, a.run, cfg, dev, max_seconds=a.max_seconds)
    return 0


def cmd_sweep(a) -> int:
    model, tok, dev = load_model(a.model, a.device)
    builder = Builder(tok)
    items = _items(a.task, a.n, a.seed + 1000, a.hops_one, a.dis_one, a.modulus, builder)
    cks = list_checkpoints(a.run)
    if not cks:
        raise SystemExit(f"no checkpoints in {a.run}; run `train` first")
    # Read the rank from the checkpoint. Trusting --rank means a mismatch shows up
    # as a shape error after the model is already loaded and the eval set built.
    _cfg0 = torch.load(cks[0][1], map_location="cpu", weights_only=False).get("cfg", {})
    rank = int(_cfg0.get("r", a.rank))
    alpha = int(_cfg0.get("alpha", 2 * rank))
    if rank != a.rank:
        print(f"[rank  ] using r={rank} from the checkpoint (--rank said {a.rank})")
    wrapped = inject_lora(model, rank, alpha, reset=True)
    out = os.path.join(a.run, "sweep.jsonl")
    rows = read_jsonl(out) if os.path.exists(out) else []
    done = {r["step"] for r in rows}
    for step, path in cks:
        if step in done:
            continue
        st = torch.load(path, map_location="cpu", weights_only=False)
        load_lora_state_dict(wrapped, st["lora"])
        model.eval()
        agg, _ = _eval_set(model, tok, builder, items, dev,
                           a.max_new_tokens, a.gen_batch, a.seed,
                           with_retention=True)
        rows.append({"step": step, **agg})
        write_jsonl(sorted(rows, key=lambda r: r["step"]), out)
        print(f"  step {step:5d}  acc={agg.get('acc', float('nan')):.3f}"
              f"  rho={agg.get('rho_cot', float('nan')):.3f}"
              f"  err={agg.get('err_prop_rate', float('nan')):.3f}"
              f"  d*={agg.get('d_star_frac', float('nan')):.3f}"
              f"  nll={agg.get('nll_generic', float('nan')):.3f}", flush=True)
    stats = figure1({os.path.basename(a.run.rstrip("/")): rows},
                    os.path.join(a.run, "figure1.png"))
    print(json.dumps(stats, indent=2))

    # Did the treatment actually get applied? Without this, a flat faithfulness
    # curve is uninterpretable: it could mean forgetting does not hurt
    # faithfulness, or that no forgetting happened.
    ordered = sorted(rows, key=lambda r: r["step"])
    n0 = ordered[0].get("nll_generic", float("nan"))
    n1 = ordered[-1].get("nll_generic", float("nan"))
    if n0 == n0 and n1 == n1:
        drift = n1 - n0
        print(f"\n[forgetting] generic NLL {n0:.3f} -> {n1:.3f}  (drift {drift:+.3f})")
        if drift < 0.05:
            print("  WARNING: general capability barely moved, so catastrophic "
                  "forgetting did not occur. A null faithfulness result here means "
                  "the treatment was never applied, not that the hypothesis is "
                  "false. Raise --lr, --steps or --rank and re-run.")
    return 0


def cmd_fig2(a) -> int:
    model, tok, dev = load_model(a.model, a.device)
    builder = Builder(tok)
    nl = n_layers_of(model)
    layer = a.layer if a.layer >= 0 else int(0.79 * nl)
    print(f"[layer ] {layer} of {nl}   d_model={d_model_of(model)}")
    items = _items(a.task, a.n, a.seed + 7, a.hops_one, a.dis_one, a.modulus, builder)
    cots = generate_cots(model, tok, items, builder, dev,
                         max_new_tokens=a.max_new_tokens, batch_size=a.gen_batch)
    seqs = [builder.full(i, c) for i, c in zip(items, cots)]

    # The SAE needs far more activations than the analysis set provides. 160 items
    # is ~29k token activations; for 4096 latents that is 7 tokens per latent and
    # the dictionary is hopelessly under-determined -- most latents die and the
    # partition becomes noise. Collect from a much larger corpus of *unanswered*
    # prompts, which costs one cheap single-row forward each.
    sae_items = _items(a.task, a.sae_items, a.seed + 4242,
                       a.hops_one, a.dis_one, a.modulus, builder)
    sae_seqs = [builder.full(i, i.gold_cot) for i in sae_items]
    acts = collect_activations(model, sae_seqs, layer, dev, max_tokens=a.sae_tokens)
    per_latent = acts.shape[0] / max(1, a.n_latents)
    print(f"[sae   ] {acts.shape[0]} activations  ({per_latent:.0f} per latent)")
    if per_latent < 50:
        print(f"  WARNING: only {per_latent:.0f} activations per latent. Raise "
              "--sae-items or lower --n-latents, or the dictionary will be noise.")
    sae, st = train_sae(acts, n_latents=a.n_latents, k=a.topk, steps=a.sae_steps,
                        device=dev, seed=a.seed)
    print("[sae   ]", {k: round(v, 4) for k, v in st.items()})
    if st["dead_frac"] > 0.5:
        print(f"  WARNING: {st['dead_frac']:.0%} of latents never fired. The "
              "partition is drawn from a mostly-dead dictionary -- lower "
              "--n-latents or raise --sae-items before trusting Figure 2.")
    ident = max(splice_is_identity(model, sae, layer, s, dev) for s in seqs[:8])
    print(f"[check ] splice identity max|dlogit| = {ident:.2e}")
    if ident > 1e-2:
        print("ABORT: splice is not an identity; every result would be confounded.")
        return 2
    # Domain latents must come from the FINE-TUNING distribution, not from more
    # reasoning items. Using held-out reasoning prompts here would make M_dom
    # "latents that fire on reasoning", which is what M_med already is -- the
    # domain-overlap diagnostic and the rescue/cost frontier would both be
    # measuring nothing.
    dom_ex, dom_note = _domain_for(a.dom_cond, a.n_domain, a.seed, a.domain_jsonl)
    dom_seqs = [builder.raw(e.prompt + e.target) for e in dom_ex[:a.n_domain]]
    print(f"[domain] {len(dom_seqs)} sequences for M_dom ({dom_note})")
    part = partition_latents(model, sae, layer, builder, items[:a.n_partition],
                             cots[:a.n_partition], dev, dom_seqs,
                             top_k=a.top_k_latents)
    print("[part  ]", {k: round(v, 4) for k, v in part.summary().items()})
    for w in part.check():
        print(f"  WARNING: {w}")
    dd = double_dissociation(model, sae, layer, builder, items[:a.n_dd],
                             cots[:a.n_dd], part, dev, seed=a.seed)
    for k, v in dd.items():
        print(f"  {k:14s} faith={v['faithfulness']:+.4f} flu={v['fluency']:+.4f} "
              f"acc={v['accuracy']:.3f}")
    # Which latents did fine-tuning actually overwrite? The partition above is
    # built on the base model; this compares it against the final checkpoint.
    dvr: Dict[str, float] = {}
    ck = list_checkpoints(a.run) if os.path.isdir(a.run) else []
    if ck:
        base_act = group_activation(model, sae, layer, seqs[:64], dev)
        st_ck = torch.load(ck[-1][1], map_location="cpu", weights_only=False)
        # Read the rank from the checkpoint rather than trusting a flag: a
        # mismatch here is a shape error deep inside the load, after the SAE and
        # the whole dissociation have already been paid for.
        rank = int(st_ck.get("cfg", {}).get("r", a.rank))
        alpha = int(st_ck.get("cfg", {}).get("alpha", 2 * rank))
        w = inject_lora(model, rank, alpha, reset=True)
        load_lora_state_dict(w, st_ck["lora"])
        model.eval()
        ft_act = group_activation(model, sae, layer, seqs[:64], dev)
        dvr = differential_vulnerability(base_act, ft_act, part, seed=a.seed)
        inject_lora(model, rank, alpha, reset=True)
        print(f"[dvr   ] step {ck[-1][0]}: " +
              " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                       for k, v in dvr.items()))
        if dvr.get("DVR", 0) > 1 and dvr.get("med_beats_random"):
            print("  >>> mediators were disturbed more than generators (DVR > 1)")

    test = dissociation_test(dd)
    print("[result]", json.dumps(test, indent=2, default=float))
    figure2(dd, os.path.join(a.run, "figure2.png"))
    with open(os.path.join(a.run, "figure2.json"), "w") as fh:
        json.dump({"dd": dd, "test": test, "dvr": dvr, "partition": part.summary(),
                   "sae": st, "layer": layer, "warnings": part.check()},
                  fh, indent=2, default=float)
    torch.save({"sae": sae.state_dict(), "med": part.med, "gen": part.gen,
                "dom": part.dom, "layer": layer},
               os.path.join(a.run, "sae_partition.pt"))
    return 0


# ==========================================================================
# pipeline: one command, everything, both GPUs
# ==========================================================================


def _spawn(gpu: int, argv: List[str], log: str) -> subprocess.Popen:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONUNBUFFERED"] = "1"
    fh = open(log, "a", buffering=1)
    fh.write(f"\n{'='*66}\n$ {' '.join(argv)}\n{'='*66}\n")
    return subprocess.Popen([sys.executable, script_path(), *argv],
                            env=env, stdout=fh, stderr=subprocess.STDOUT)


def _wait(jobs, root) -> None:
    for name, p in jobs:
        rc = p.wait()
        tag = "ok" if rc == 0 else f"EXIT {rc}"
        print(f"  [{tag}] {name}   (log: {root}/logs/{name}.log)", flush=True)
        if rc != 0:
            print(f"  ---- last lines of {name}.log ----", flush=True)
            try:
                print("".join(open(f"{root}/logs/{name}.log").readlines()[-25:]))
            except OSError:
                pass


def cmd_pipeline(a) -> int:
    root = _workdir()
    os.makedirs(f"{root}/logs", exist_ok=True)
    ngpu = max(1, torch.cuda.device_count())
    print(f"\n### workdir {root}   GPUs {ngpu}", flush=True)
    print(f"### plan: {a.conditions.replace(',', ' + ')} on "
          f"{a.force_cell or 'the best gate cell'}, {a.steps} steps, "
          f"{a.steps // a.ckpt_every + 1} checkpoints, {a.n} items", flush=True)
    print("### the number that decides the study is CONTRAST, printed at the end\n",
          flush=True)

    common = ["--n", str(a.n), "--max-new-tokens", str(a.max_new_tokens),
              "--gen-batch", str(a.gen_batch), "--seed", str(a.seed)]

    print("### STAGE 0  environment + self-checks", flush=True)
    if run_selftest(verbose=False) != 0:
        print("SELF-CHECKS FAILED -- stopping before any GPU time is spent.")
        return 1
    for w in preflight(verbose=True):
        print(f"  WARN  {w}")

    if not a.no_eta:
        # Run the probe in a SUBPROCESS. Loading the model here would leave ~3.5 GB
        # of weights plus a CUDA context resident on GPU 0 for the rest of the
        # pipeline, and the stage-1 child would then OOM on the same card.
        print("\n### timing probe (about a minute)", flush=True)
        _wait([("eta", _spawn(0, [
            "eta", "--model", a.model or a.gate_models.split(",")[0],
            "--gate-models", a.gate_models, "--conditions", a.conditions,
            "--hops", a.hops, "--distractors", a.distractors,
            "--steps", str(a.steps), "--ckpt-every", str(a.ckpt_every),
            "--sae-items", str(a.sae_items), "--n-dd", str(a.n_dd),
            "--ngpu", str(ngpu), *common],
            f"{root}/logs/eta.log"))], root)
        try:
            for ln in open(f"{root}/logs/eta.log"):
                if any(k in ln for k in ("measured", "TOTAL", "gate ", "sweep ")):
                    print("  " + ln.rstrip(), flush=True)
        except OSError:
            pass

    # ---- stage 1: gate --------------------------------------------------
    best_path = f"{root}/gate/best_cell.json"
    if a.force_cell and not os.path.exists(best_path):
        # Deliberate override. The gate is a guard, not an oracle: its thresholds
        # were fixed before any data existed. When the primary metric (rho_cot) has
        # headroom and the chain is demonstrably load-bearing, a supporting check
        # falling just short is a limitation to report, not a reason to stop.
        try:
            _t, _h, _d = a.force_cell.split(":")
        except ValueError:
            raise SystemExit("--force-cell wants task:hops:distractors, "
                             "e.g. modarith:2:0")
        os.makedirs(f"{root}/gate", exist_ok=True)
        json.dump({"model": a.model or a.gate_models.split(",")[0], "task": _t,
                   "hops": int(_h), "distractors": int(_d), "gate_pass": False,
                   "forced": True}, open(best_path, "w"), indent=2)
        print(f"\n### gate OVERRIDDEN: using {_t} h={_h} d={_d} without passing.\n"
              "    Record in the paper which gate conditions were not met.\n",
              flush=True)
    if not os.path.exists(best_path):
        print("\n### STAGE 1  phase-0 gate", flush=True)
        models = a.gate_models.split(",")
        for i in range(0, len(models), ngpu):
            jobs = []
            for g, m in enumerate(models[i:i + ngpu]):
                tag = m.split("/")[-1]
                jobs.append((f"gate_{tag}", _spawn(g, [
                    "gate", "--model", m, "--device", "cuda",
                    "--hops", a.hops, "--distractors", a.distractors,
                    "--out", f"{root}/gate/{tag}", *common],
                    f"{root}/logs/gate_{tag}.log")))
            _wait(jobs, root)
        cells = []
        for f in glob.glob(f"{root}/gate/*/best_cell.json"):
            cells.append(json.load(open(f)))
        if not cells:
            print("\nNO MODEL PASSED THE GATE. Nothing downstream is measurable.\n"
                  "Raise --hops / --distractors, or use a larger model.")
            return 2
        best = max(cells, key=lambda c: c.get("shuffled_acc_gap", 0.0))
        os.makedirs(f"{root}/gate", exist_ok=True)
        json.dump(best, open(best_path, "w"), indent=2)
    best = json.load(open(best_path))
    model = a.model or best["model"]
    print(f"\n### using model={model} task={best['task']} "
          f"hops={best['hops']} distractors={best['distractors']}", flush=True)

    tcfg = ["--task", best["task"], "--hops-one", str(best["hops"]),
            "--dis-one", str(best["distractors"])]

    # ---- stage 2: train + sweep, two conditions per pass ----------------
    conds = a.conditions.split(",")
    for i in range(0, len(conds), ngpu):
        chunk = conds[i:i + ngpu]
        print(f"\n### STAGE 2  train: {', '.join(chunk)}", flush=True)
        jobs = []
        for g, c in enumerate(chunk):
            fin = os.path.join(root, c, f"ckpt_{a.steps:06d}.pt")
            if os.path.exists(fin):
                print(f"  [skip] train_{c} already complete", flush=True)
                continue
            jobs.append((f"train_{c}", _spawn(g, [
                "train", "--model", model, "--device", "cuda", "--cond", c,
                "--run", f"{root}/{c}", "--steps", str(a.steps),
                "--ckpt-every", str(a.ckpt_every), "--rank", str(a.rank),
                "--domain-jsonl", a.domain_jsonl,
                "--max-seconds", str(a.max_seconds), "--seed", str(a.seed)],
                f"{root}/logs/train_{c}.log")))
        _wait(jobs, root)

        print(f"### STAGE 2  sweep: {', '.join(chunk)}", flush=True)
        jobs = []
        for g, c in enumerate(chunk):
            jobs.append((f"sweep_{c}", _spawn(g, [
                "sweep", "--model", model, "--device", "cuda",
                "--run", f"{root}/{c}", "--rank", str(a.rank), *tcfg, *common],
                f"{root}/logs/sweep_{c}.log")))
        _wait(jobs, root)

    # ---- stage 3: figure 2 ----------------------------------------------
    primary = conds[0]
    if not os.path.exists(f"{root}/{primary}/figure2.json"):
        print(f"\n### STAGE 3  figure 2 on {primary}", flush=True)
        _wait([("fig2", _spawn(0, [
            "fig2", "--model", model, "--device", "cuda",
            "--run", f"{root}/{primary}", "--layer", str(a.layer),
            "--n-latents", str(a.n_latents), "--sae-steps", str(a.sae_steps),
            "--sae-items", str(a.sae_items), "--n-dd", str(a.n_dd),
            "--dom-cond", primary, "--n-domain", str(a.n_domain),
            "--n-partition", str(a.n_partition),
            "--domain-jsonl", a.domain_jsonl,
            "--top-k-latents", str(a.top_k_latents), *tcfg, *common],
            f"{root}/logs/fig2.log"))], root)

    # ---- stage 3b: prevention (C3) --------------------------------------
    part_file = f"{root}/{primary}/sae_partition.pt"
    if a.protect and os.path.exists(part_file):
        if not os.path.exists(os.path.join(root, "protected",
                                           f"ckpt_{a.steps:06d}.pt")):
            print("\n### STAGE 3b  prevention: gradient-masked re-run", flush=True)
            _wait([("train_protected", _spawn(0, [
                "train", "--model", model, "--device", "cuda", "--cond", "protected",
                "--run", f"{root}/protected", "--protect-from", part_file,
                "--steps", str(a.steps), "--ckpt-every", str(a.ckpt_every),
                "--rank", str(a.rank), "--domain-jsonl", a.domain_jsonl,
                "--max-seconds", str(a.max_seconds), "--seed", str(a.seed)],
                f"{root}/logs/train_protected.log"))], root)
        _wait([("sweep_protected", _spawn(0, [
            "sweep", "--model", model, "--device", "cuda",
            "--run", f"{root}/protected", "--rank", str(a.rank), *tcfg, *common],
            f"{root}/logs/sweep_protected.log"))], root)

    # ---- stage 4: collect -----------------------------------------------
    print("\n### STAGE 4  results", flush=True)
    stats = build_all(root)
    _runs = {}
    for _p in sorted(glob.glob(f"{root}/*/sweep.jsonl")):
        _runs[os.path.basename(os.path.dirname(_p))] = read_jsonl(_p)
    _con = contrast(_runs)
    if _con:
        print("\n  ---- HEADLINE ----", flush=True)
        for k, v in _con.items():
            tag = ""
            if k.startswith("CONTRAST"):
                tag = "   PASS" if (v == v and v >= 0.20) else "   below 0.20"
            print(f"  {k:34s} " + ("n/a" if v != v else f"{v:+.3f}") + tag, flush=True)
        print("  ------------------\n", flush=True)
    print(json.dumps(stats, indent=2))
    for name, st in stats.items():
        if st["fadg"] == st["fadg"] and st["fadg"] > 0:
            print(f"  >>> SILENT WINDOW in {name}: faithfulness fell "
                  f"{st['fadg']:.0f} steps before accuracy")
    f2 = f"{root}/{primary}/figure2.json"
    if os.path.exists(f2):
        d = json.load(open(f2))
        print(f"  dissociation: {d['test']['dissociation']}  "
              f"(beats random: {d['test'].get('beats_random')})")
        if d.get("dvr"):
            print(f"  DVR: {d['dvr'].get('DVR', float('nan')):.3f}  "
                  f"med>random: {d['dvr'].get('med_beats_random')}")
        for w in d.get("warnings", []):
            print(f"  WARNING: {w}")
    print("\nwritten to " + root + ":")
    for f in sorted(os.listdir(root)):
        if f.endswith((".png", ".csv", ".md")):
            print("  " + f)
    return 0


# ==========================================================================


_SCRIPT_PATH: Optional[str] = None


def _cell_source() -> Optional[str]:
    """This file's source, recovered from IPython's input history.

    When the file is pasted into a notebook cell there is no __file__ and no file
    on disk, so worker subprocesses cannot be launched. IPython keeps every
    executed cell in `In`, so we can find the cell that contained this source and
    write it back out.
    """
    try:
        ip = get_ipython()                      # type: ignore[name-defined]
        hist = ip.user_ns.get("In") or []
    except Exception:                           # noqa: BLE001
        return None
    for src in reversed(hist):
        if isinstance(src, str) and "def cmd_pipeline" in src and "def main(" in src:
            return src
    return None


def script_path() -> str:
    """A real file containing this source, for spawning one worker per GPU."""
    global _SCRIPT_PATH
    if _SCRIPT_PATH and os.path.exists(_SCRIPT_PATH):
        return _SCRIPT_PATH
    f = globals().get("__file__")
    if f and os.path.exists(f) and "ipykernel" not in f:
        _SCRIPT_PATH = os.path.abspath(f)
        return _SCRIPT_PATH
    src = _cell_source()
    if src:
        dst = os.path.abspath(os.path.join(_workdir(), os.pardir, "cfd_all.py"))
        with open(dst, "w", encoding="utf-8") as fh:
            fh.write(src)
        _SCRIPT_PATH = dst
        return dst
    raise RuntimeError(
        "cannot locate this file on disk, so worker processes cannot be started.\n"
        "Save it first:  put  %%writefile cfd_all.py  as the FIRST line of the cell\n"
        "containing this code, run it, then use:  !python cfd_all.py pipeline")


def in_notebook() -> bool:
    """True inside Jupyter/Colab/Kaggle, where sys.argv belongs to the kernel."""
    if "ipykernel" in sys.modules or "google.colab" in sys.modules:
        return True
    try:
        return get_ipython().__class__.__name__ in (        # type: ignore[name-defined]
            "ZMQInteractiveShell", "Shell")
    except NameError:
        return False


def cfd(cmdline: str = "check") -> int:
    """Notebook entry point.  cfd("pipeline --n 100")"""
    import shlex
    return main(shlex.split(cmdline))


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="catastrophic forgetting x CoT faithfulness",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(q, need_model=True):
        q.add_argument("--model", required=need_model, default="")
        q.add_argument("--device", default="cuda")
        q.add_argument("--seed", type=int, default=0)
        q.add_argument("--n", type=int, default=100)
        q.add_argument("--max-new-tokens", type=int, default=512)
        q.add_argument("--gen-batch", type=int, default=16)
        q.add_argument("--modulus", type=int, default=11)

    c = sub.add_parser("check"); c.set_defaults(fn=cmd_check)

    fg = sub.add_parser("figures")
    fg.add_argument("--root", default=_workdir())
    fg.add_argument("--delta", type=float, default=0.20)
    fg.set_defaults(fn=cmd_figures)

    g = sub.add_parser("gate"); common(g)
    g.add_argument("--tasks", default="chain,modarith")
    g.add_argument("--hops", default="2,3,4,5")
    g.add_argument("--distractors", default="0,3,6")
    g.add_argument("--out", default=_workdir() + "/gate")
    g.set_defaults(fn=cmd_gate)

    t = sub.add_parser("train"); common(t)
    t.add_argument("--cond", default="ood_cot",
                   choices=["ood_answer", "ood_cot", "replay", "replay_generic",
                            "random_label", "protected"])
    t.add_argument("--protect-from", default="",
                   help="sae_partition.pt from a fig2 run; required for --cond protected")
    t.add_argument("--run", required=True)
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
    f.add_argument("--n-latents", type=int, default=4096)
    f.add_argument("--topk", type=int, default=32)
    f.add_argument("--sae-steps", type=int, default=3000)
    f.add_argument("--sae-items", type=int, default=3000)
    f.add_argument("--sae-tokens", type=int, default=500000)
    f.add_argument("--top-k-latents", type=int, default=64)
    f.add_argument("--n-partition", type=int, default=48)
    f.add_argument("--n-dd", type=int, default=64)
    f.add_argument("--n-domain", type=int, default=128)
    f.add_argument("--dom-cond", default="ood_cot")
    f.add_argument("--rank", type=int, default=16)
    f.add_argument("--domain-jsonl", default="")
    f.set_defaults(fn=cmd_fig2)

    z = sub.add_parser("pipeline"); common(z, need_model=False)
    # Bare `pipeline` IS the recommended run: the gate has already been measured
    # four times, so it is skipped, and the two conditions that carry the headline
    # contrast run in parallel on the two GPUs.
    z.set_defaults(n=80)
    z.add_argument("--gate-models",
                   default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,"
                           "Qwen/Qwen2.5-1.5B-Instruct")
    z.add_argument("--conditions", default="ood_cot,replay")
    z.add_argument("--hops", default="2,3")
    z.add_argument("--distractors", default="0")
    z.add_argument("--steps", type=int, default=600)
    z.add_argument("--ckpt-every", type=int, default=50)
    z.add_argument("--rank", type=int, default=16)
    z.add_argument("--max-seconds", type=float, default=39000)
    z.add_argument("--layer", type=int, default=-1)
    z.add_argument("--n-latents", type=int, default=4096)
    z.add_argument("--sae-steps", type=int, default=3000)
    z.add_argument("--sae-items", type=int, default=3000)
    z.add_argument("--n-dd", type=int, default=64)
    z.add_argument("--n-domain", type=int, default=128)
    z.add_argument("--n-partition", type=int, default=48)
    z.add_argument("--domain-jsonl", default="")
    z.add_argument("--no-eta", action="store_true")
    z.add_argument("--force-cell", default="modarith:2:0",
                   help="skip the gate and use this cell; '' runs the gate")
    z.add_argument("--no-protect", dest="protect", action="store_false",
                   help="skip the gradient-mask prevention condition")
    z.set_defaults(protect=False)
    z.add_argument("--top-k-latents", type=int, default=64)
    z.set_defaults(fn=cmd_pipeline)

    e = sub.add_parser("eta"); common(e, need_model=False)
    e.add_argument("--gate-models",
                   default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,"
                           "Qwen/Qwen2.5-1.5B-Instruct")
    e.add_argument("--conditions",
                   default="ood_cot,replay,replay_generic,random_label")
    e.add_argument("--hops", default="2,3")
    e.add_argument("--distractors", default="0")
    e.add_argument("--steps", type=int, default=600)
    e.add_argument("--ckpt-every", type=int, default=50)
    e.add_argument("--sae-items", type=int, default=3000)
    e.add_argument("--n-dd", type=int, default=64)
    e.add_argument("--ngpu", type=int, default=0,
                   help="true GPU count; the probe is pinned to one card")
    e.set_defaults(fn=cmd_eta)

    if argv is None:
        argv = sys.argv[1:]
    a = p.parse_args(argv)
    for attr in ("run", "out"):
        v = getattr(a, attr, None)
        if v:
            os.makedirs(v, exist_ok=True)
    return a.fn(a)


if __name__ == "__main__" and not in_notebook():
    sys.exit(main())
elif in_notebook():
    try:
        _p = script_path()
        print(f"cfd_all saved to {_p}")
    except RuntimeError as _e:
        print(f"[warn] could not save a copy to disk: {_e}")
    if os.environ.get("CFD_NO_AUTORUN"):
        print("\nCFD_NO_AUTORUN is set, so nothing was started. Entry points:\n"
              "    cfd('pipeline')              everything: gate -> both figures\n"
              "    cfd('check')                 26 self-checks, no GPU, ~1 min\n"
              "    cfd('eta')                   how long a full run takes here\n"
              "    cfd('pipeline --n 100')      smaller and faster")
    else:
        print("Starting the full pipeline. It self-checks first, then measures how\n"
              "long the run will take on this machine, then does the work.\n"
              "Interrupt the cell to stop; re-run it to continue where it stopped.\n"
              "To load without running, set CFD_NO_AUTORUN=1 before the paste.\n",
              flush=True)
        try:
            _rc = cfd(os.environ.get("CFD_ARGS", "pipeline"))
            print(f"\n[pipeline finished with code {_rc}]")
        except KeyboardInterrupt:
            print("\n[interrupted -- re-run this cell to continue where it stopped]")
        except Exception as _e:                                  # noqa: BLE001
            traceback.print_exc()
            print(f"\n[pipeline failed: {type(_e).__name__}: {_e}]")
