"""
Tier B/C: sparse features, the mediator/generator/domain partition, and the
double-dissociation intervention.

The central object is a *splice* hook: it replaces a layer's residual stream with
    h  ->  decode(encode(h)) + (h - decode(encode(h))).detach()
which is numerically an exact identity at baseline (verified in the self-test) but
lets us (a) take gradients with respect to individual latents and (b) zero chosen
latents at chosen positions. Without the frozen error term the model would be
degraded by reconstruction loss and every downstream number would be confounded.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .engine import Builder, Seq, score_candidates, margin
from .tasks import Item


# -----------------------------------------------------------------------------
# TopK SAE
# -----------------------------------------------------------------------------


class TopKSAE(nn.Module):
    def __init__(self, d_model: int, n_latents: int, k: int = 32):
        super().__init__()
        self.d_model, self.n_latents, self.k = d_model, n_latents, k
        self.b_pre = nn.Parameter(torch.zeros(d_model))
        self.W_enc = nn.Parameter(torch.empty(n_latents, d_model))
        self.b_enc = nn.Parameter(torch.zeros(n_latents))
        self.W_dec = nn.Parameter(torch.empty(d_model, n_latents))
        nn.init.kaiming_uniform_(self.W_enc, a=math.sqrt(5))
        with torch.no_grad():
            self.W_dec.copy_(self.W_enc.t())
            self.normalise_decoder()

    @torch.no_grad()
    def normalise_decoder(self) -> None:
        self.W_dec.div_(self.W_dec.norm(dim=0, keepdim=True).clamp_min(1e-8))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = F.linear(x - self.b_pre, self.W_enc, self.b_enc)
        pre = F.relu(pre)
        k = min(self.k, pre.shape[-1])
        val, idx = torch.topk(pre, k, dim=-1)
        z = torch.zeros_like(pre)
        return z.scatter(-1, idx, val)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return F.linear(z, self.W_dec) + self.b_pre

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return self.decode(z), z


def get_layer(model, layer: int) -> nn.Module:
    inner = getattr(model, "model", model)
    layers = getattr(inner, "layers", None)
    if layers is None:
        layers = getattr(getattr(model, "transformer", object()), "h", None)
    if layers is None:
        raise RuntimeError("could not locate the decoder layer list")
    if not (-len(layers) <= layer < len(layers)):
        raise IndexError(f"layer {layer} out of range for {len(layers)} layers")
    return layers[layer]


def _unpack(out):
    """Decoder layers return either a tensor or a tuple whose first item is one."""
    if isinstance(out, tuple):
        return out[0], (lambda new: (new,) + out[1:])
    return out, (lambda new: new)


@torch.no_grad()
def collect_activations(model, seqs: List[Seq], layer: int, device: str,
                        max_tokens: int = 500_000) -> torch.Tensor:
    """Residual-stream activations at `layer`, float32, (n_tokens, d_model)."""
    mod = get_layer(model, layer)
    d_model = int(model.get_input_embeddings().weight.shape[1])
    cap = min(max_tokens, sum(len(s.ids) for s in seqs))
    # Preallocate and fill. Collecting into a list and calling torch.cat at the end
    # holds two full copies at once, which for a 1536-dim model and a few hundred
    # thousand tokens is several gigabytes of host RAM at the peak.
    buf = torch.empty(cap, d_model, dtype=torch.float32)
    filled = 0
    box = {}

    def hook(_m, _i, out):
        box["h"] = _unpack(out)[0].detach()[0]
        return out

    handle = mod.register_forward_hook(hook)
    try:
        for s in seqs:
            if filled >= cap:
                break
            model(input_ids=torch.tensor([s.ids], dtype=torch.long, device=device))
            h = box.pop("h", None)
            if h is None:
                continue
            take = min(h.shape[0], cap - filled)
            buf[filled:filled + take] = h[:take].float().cpu()
            filled += take
    finally:
        handle.remove()
    return buf[:filled]


def train_sae(acts: torch.Tensor, n_latents: int = 8192, k: int = 32,
              steps: int = 2000, batch: int = 1024, lr: float = 3e-4,
              device: str = "cpu", seed: int = 0, verbose: bool = True
              ) -> Tuple[TopKSAE, Dict[str, float]]:
    torch.manual_seed(seed)
    d = acts.shape[1]
    sae = TopKSAE(d, n_latents, k).to(device)
    with torch.no_grad():
        sae.b_pre.copy_(acts.mean(0).to(device))
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    fired = torch.zeros(n_latents, dtype=torch.bool, device=device)
    var = acts.var(0).sum().item()
    g = torch.Generator().manual_seed(seed)
    last = 0.0
    for s in range(steps):
        idx = torch.randint(0, acts.shape[0], (min(batch, acts.shape[0]),), generator=g)
        x = acts[idx].to(device)
        recon, z = sae(x)
        loss = F.mse_loss(recon, x)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sae.normalise_decoder()
        fired |= (z.detach() > 0).any(0)
        last = float(loss.item())
        if verbose and (s + 1) % max(1, steps // 5) == 0:
            print(f"    sae step {s+1}/{steps}  mse {last:.5f}")
    with torch.no_grad():
        sub = acts[: min(4096, acts.shape[0])].to(device)
        rec, _ = sae(sub)
        resid = (sub - rec).pow(2).sum().item()
        denom = (sub - sub.mean(0)).pow(2).sum().item()
    stats = {
        "mse": last,
        "fvu": resid / max(denom, 1e-9),
        "dead_frac": float((~fired).float().mean().item()),
        "n_latents": float(n_latents), "k": float(k),
    }
    return sae, stats


# -----------------------------------------------------------------------------
# splice
# -----------------------------------------------------------------------------


class Splice:
    """Context manager installing an SAE splice at one layer.

    ablate : {latent_index: [positions]} or {latent_index: None} for all positions
    """

    def __init__(self, model, sae: TopKSAE, layer: int,
                 ablate: Optional[Dict[int, Optional[Sequence[int]]]] = None,
                 keep_grad: bool = False):
        self.model, self.sae, self.layer = model, sae, layer
        self.ablate = ablate or {}
        self.keep_grad = keep_grad
        self.z: Optional[torch.Tensor] = None
        self._h: Optional[torch.Tensor] = None

    def __enter__(self):
        mod = get_layer(self.model, self.layer)

        def hook(_m, _i, out):
            h, rewrap = _unpack(out)
            hf = h.to(torch.float32)
            z = self.sae.encode(hf)
            base_recon = self.sae.decode(z)
            err = (hf - base_recon).detach()
            if self.ablate:
                z = z.clone()
                T = z.shape[1]
                for j, pos in self.ablate.items():
                    if pos is None:
                        z[:, :, j] = 0.0
                    else:
                        p = [q for q in pos if 0 <= q < T]
                        if p:
                            z[:, torch.as_tensor(p, device=z.device), j] = 0.0
            if self.keep_grad:
                # z is a leaf only when the model's parameters are frozen. If any
                # parameter requires grad (e.g. a LoRA adapter is attached) z is a
                # non-leaf and requires_grad_() would raise. retain_grad() works in
                # both cases, so only flip the flag when it is actually needed.
                if not z.requires_grad:
                    z.requires_grad_(True)
                z.retain_grad()
            self.z = z
            new = (self.sae.decode(z) + err).to(h.dtype)
            self._h = new
            return rewrap(new)

        self._handle = mod.register_forward_hook(hook)
        return self

    def __exit__(self, *exc):
        self._handle.remove()
        return False


def splice_is_identity(model, sae: TopKSAE, layer: int, seq: Seq,
                       device: str, tol: float = 1e-3) -> float:
    """Max |logit difference| introduced by an un-ablated splice. Must be ~0."""
    ids = torch.tensor([seq.ids], dtype=torch.long, device=device)
    with torch.no_grad():
        base = model(input_ids=ids).logits.float()
        with Splice(model, sae, layer):
            spl = model(input_ids=ids).logits.float()
    return float((base - spl).abs().max().item())


# -----------------------------------------------------------------------------
# indirect effects
# -----------------------------------------------------------------------------


def _margin_tensor(model, seq: Seq, cand_ids: List[List[int]], gold_idx: int,
                   device: str) -> torch.Tensor:
    """Differentiable margin using only first candidate tokens (one forward)."""
    firsts = torch.tensor([c[0] for c in cand_ids], dtype=torch.long, device=device)
    ids = torch.tensor([seq.ids], dtype=torch.long, device=device)
    logits = model(input_ids=ids).logits[0, -1].float()
    sc = torch.log_softmax(logits, -1)[firsts]
    others = torch.cat([sc[:gold_idx], sc[gold_idx + 1:]])
    return sc[gold_idx] - torch.logsumexp(others, 0)


GRAD_SCALE = 2.0 ** 14


def _scaled_backward(objective: torch.Tensor, sp: "Splice", model,
                     scale: float) -> torch.Tensor:
    """Backward with loss scaling, then unscale.

    On a T4 the base model runs in fp16 and gradients flowing back through ~20
    layers underflow to exactly zero long before they reach the splice. The result
    is not an error -- it is an all-zero attribution vector, an arbitrary latent
    partition and a meaningless Figure 2. Scaling the objective before backward and
    dividing the gradient afterwards is mathematically a no-op in exact arithmetic
    and is what keeps the small values representable in fp16.
    """
    model.zero_grad(set_to_none=True)
    (objective * scale).backward()
    if sp.z.grad is None:
        raise RuntimeError("no gradient reached the SAE latents; is grad enabled?")
    g = sp.z.grad.detach().to(torch.float32) / scale
    if not torch.isfinite(g).all():
        raise RuntimeError(
            f"non-finite SAE gradients at grad scale {scale:g}. Lower "
            "analysis.GRAD_SCALE (try 2**10) and re-run.")
    if float(g.abs().max()) == 0.0:
        raise RuntimeError(
            f"SAE gradients underflowed to zero at grad scale {scale:g}. This is "
            "the classic fp16 failure on T4/P100: every attribution would be 0 and "
            "the latent partition would be arbitrary. Raise analysis.GRAD_SCALE "
            "(try 2**18), or splice at a later layer so fewer fp16 layers sit "
            "above it.")
    return g


def attribution_ie(model, sae: TopKSAE, layer: int, seq: Seq,
                   cand_ids: List[List[int]], gold_idx: int, device: str,
                   positions: Optional[Sequence[int]] = None,
                   scale: float = GRAD_SCALE) -> torch.Tensor:
    """First-order estimate of margin loss from zeroing each latent, (n_latents,).

    IE(j) ~= sum_p z[p,j] * d(margin)/d z[p,j].  Positive => the latent SUPPORTS
    the correct answer at those positions, i.e. it is load-bearing.
    """
    with torch.enable_grad():
        with Splice(model, sae, layer, keep_grad=True) as sp:
            m = _margin_tensor(model, seq, cand_ids, gold_idx, device)
            g = _scaled_backward(m, sp, model, scale)
            z = sp.z
    contrib = (z.detach().to(torch.float32) * g)[0]        # (T, n_latents)
    if positions is not None:
        idx = torch.as_tensor([p for p in positions if 0 <= p < contrib.shape[0]],
                              dtype=torch.long, device=contrib.device)
        contrib = contrib[idx] if idx.numel() else contrib[:0]
    return contrib.sum(0).cpu() if contrib.numel() else torch.zeros(sae.n_latents)


def attribution_gen(model, sae: TopKSAE, layer: int, seq: Seq, device: str,
                    positions: Optional[Sequence[int]] = None,
                    scale: float = GRAD_SCALE) -> torch.Tensor:
    """Same, but for the log-likelihood of the CoT tokens themselves.

    High here + ~0 on the answer  =>  the latent writes the chain but does not
    route it into the answer: a GENERATOR.
    """
    ids = torch.tensor([seq.ids], dtype=torch.long, device=device)
    s, e = seq.spans["cot"]
    if e - s < 2:
        return torch.zeros(sae.n_latents)
    with torch.enable_grad():
        with Splice(model, sae, layer, keep_grad=True) as sp:
            logits = model(input_ids=ids).logits[0].float()
            lp = torch.log_softmax(logits, -1)
            tgt = torch.tensor(seq.ids[s + 1:e], dtype=torch.long, device=device)
            pos = torch.arange(s, e - 1, device=device)
            obj = lp[pos, tgt].sum()
            g = _scaled_backward(obj, sp, model, scale)
            z = sp.z
    contrib = (z.detach().to(torch.float32) * g)[0]
    if positions is not None:
        idx = torch.as_tensor([p for p in positions if 0 <= p < contrib.shape[0]],
                              dtype=torch.long, device=contrib.device)
        contrib = contrib[idx] if idx.numel() else contrib[:0]
    return contrib.sum(0).cpu() if contrib.numel() else torch.zeros(sae.n_latents)


@torch.no_grad()
def exact_ie(model, sae: TopKSAE, layer: int, seq: Seq, cand_ids: List[List[int]],
             gold_idx: int, device: str, latents: Sequence[int],
             positions: Optional[Sequence[int]] = None) -> Dict[int, float]:
    """True ablation effect for a shortlist. Attribution is only a screen; the
    tails of a first-order estimate are unreliable, so the top candidates must
    always be confirmed here before anything is claimed about them."""
    ids = torch.tensor([seq.ids], dtype=torch.long, device=device)
    firsts = torch.tensor([c[0] for c in cand_ids], dtype=torch.long, device=device)

    def m_of(logits):
        sc = torch.log_softmax(logits[0, -1].float(), -1)[firsts]
        oth = torch.cat([sc[:gold_idx], sc[gold_idx + 1:]])
        return float(sc[gold_idx] - torch.logsumexp(oth, 0))

    with Splice(model, sae, layer):
        base = m_of(model(input_ids=ids).logits)
    out: Dict[int, float] = {}
    for j in latents:
        with Splice(model, sae, layer, ablate={int(j): positions}):
            out[int(j)] = base - m_of(model(input_ids=ids).logits)
    return out


# -----------------------------------------------------------------------------
# the partition
# -----------------------------------------------------------------------------


@dataclass
class Partition:
    med: List[int]
    gen: List[int]
    dom: List[int]
    ie_answer: torch.Tensor
    ie_cot: torch.Tensor
    dom_score: torch.Tensor

    def jaccard_med_gen(self) -> float:
        a, b = set(self.med), set(self.gen)
        return len(a & b) / max(1, len(a | b))

    def summary(self) -> Dict[str, float]:
        def mean_at(t, idx):
            return float(t[torch.tensor(idx, dtype=torch.long)].mean()) if idx else float("nan")
        return {
            "n_med": float(len(self.med)), "n_gen": float(len(self.gen)),
            "n_dom": float(len(self.dom)),
            "jaccard_med_gen": self.jaccard_med_gen(),
            "med_ie_answer": mean_at(self.ie_answer, self.med),
            "gen_ie_answer": mean_at(self.ie_answer, self.gen),
            "med_ie_cot": mean_at(self.ie_cot, self.med),
            "gen_ie_cot": mean_at(self.ie_cot, self.gen),
            "med_dom_overlap": float(len(set(self.med) & set(self.dom)) / max(1, len(self.med))),
        }

    def check(self) -> List[str]:
        """Problems that would invalidate the double dissociation. Empty == good."""
        w = []
        if not self.med:
            w.append("M_med is EMPTY")
        if not self.gen:
            w.append("M_gen is EMPTY -- ablate_gen would be identical to control")
        if self.jaccard_med_gen() > 0.2:
            w.append(f"M_med and M_gen overlap (Jaccard={self.jaccard_med_gen():.2f} "
                     "> 0.2): H4 predicts a dissociation these sets cannot show")
        s = self.summary()
        if s["med_ie_answer"] <= abs(s["gen_ie_answer"]):
            w.append("mediators do not carry more answer effect than generators")
        if s["med_dom_overlap"] > 0.5:
            w.append(f"{s['med_dom_overlap']:.0%} of M_med are also domain latents; "
                     "the rescue experiment will trade faithfulness against domain acc")
        return w


def partition_latents(model, sae: TopKSAE, layer: int, builder: Builder,
                      items: List[Item], cots: List[str], device: str,
                      domain_seqs: Optional[List[Seq]] = None,
                      top_k: int = 64) -> Partition:
    """Split latents into mediators, generators and domain latents.

    mediators  : large positive IE on the ANSWER at CoT positions
    generators : large IE on the CHAIN's own tokens but near-zero on the answer
    domain     : activation rises most on the fine-tuning distribution
    """
    n = sae.n_latents
    ie_ans = torch.zeros(n)
    ie_cot = torch.zeros(n)
    used, skipped = 0, 0
    for it, cot in zip(items, cots):
        seq = builder.full(it, cot)
        cand = builder.candidate_ids(it)
        gi = it.candidates.index(it.gold)
        pos = seq.span_positions("cot")
        if not pos:
            continue
        # attribution scores candidates by FIRST token only, so collisions make the
        # margin meaningless. Repair the item by dropping the colliding candidates
        # instead of discarding it; only give up when fewer than two survive.
        if len({c[0] for c in cand}) != len(cand):
            fixed = dedupe_candidates(builder, it)
            if fixed is None:
                skipped += 1
                continue
            it = fixed
            cand = builder.candidate_ids(it)
            gi = it.candidates.index(it.gold)
            seq = builder.full(it, cot)
            pos = seq.span_positions("cot")
        used += 1
        ie_ans += attribution_ie(model, sae, layer, seq, cand, gi, device, pos)
        ie_cot += attribution_gen(model, sae, layer, seq, device, pos)
    if used == 0:
        raise RuntimeError(
            "every item had candidates sharing a first token under this tokeniser, "
            "so no attribution could be computed. Reduce --n-candidates or pick a "
            "task whose answers tokenise distinctly.")
    if skipped:
        print(f"  [part  ] skipped {skipped}/{skipped+used} items "
              "(fewer than two candidates survived first-token deduplication)")
    ie_ans /= used
    ie_cot /= used

    med = torch.topk(ie_ans, min(top_k, n)).indices.tolist()

    # Generators are selected by RANK, not by a threshold. A threshold rule
    # ("top 2% on the chain AND bottom 50% on the answer") is empty far too often,
    # and an empty arm makes the ablate_gen condition silently identical to the
    # control -- a failure that only shows up after the GPU time is already spent.
    def _z(t: torch.Tensor) -> torch.Tensor:
        return (t - t.mean()) / t.std().clamp_min(1e-8)

    gen_score = _z(ie_cot) - _z(ie_ans.abs())      # writes the chain, not the answer
    gen_score = gen_score.clone()
    gen_score[torch.tensor(med, dtype=torch.long)] = float("-inf")   # disjoint from med
    gen = torch.topk(gen_score, min(top_k, max(0, n - len(med)))).indices.tolist()

    dom_score = torch.zeros(n)
    if domain_seqs:
        base_act = torch.zeros(n)
        with torch.no_grad():
            for s in domain_seqs:
                ids = torch.tensor([s.ids], dtype=torch.long, device=device)
                with Splice(model, sae, layer) as sp:
                    model(input_ids=ids)
                dom_score += sp.z.detach()[0].float().mean(0).cpu()
            for it, cot in zip(items, cots):
                s = builder.full(it, cot)
                ids = torch.tensor([s.ids], dtype=torch.long, device=device)
                with Splice(model, sae, layer) as sp:
                    model(input_ids=ids)
                base_act += sp.z.detach()[0].float().mean(0).cpu()
        dom_score = dom_score / max(1, len(domain_seqs)) - base_act / max(1, len(items))
    dom = torch.topk(dom_score, min(top_k, n)).indices.tolist()
    return Partition(med, gen, dom, ie_ans, ie_cot, dom_score)


# -----------------------------------------------------------------------------
# C1: the double dissociation
# -----------------------------------------------------------------------------


def _fluency(model, sae, layer, seq: Seq, device: str,
             ablate: Optional[Dict[int, Optional[Sequence[int]]]]) -> float:
    """Mean log-prob of the chain's own tokens: does the CoT still read as a chain?"""
    s, e = seq.spans["cot"]
    if e - s < 2:
        return float("nan")
    ids = torch.tensor([seq.ids], dtype=torch.long, device=device)
    with torch.no_grad():
        with Splice(model, sae, layer, ablate=ablate):
            lp = torch.log_softmax(model(input_ids=ids).logits[0].float(), -1)
    tgt = torch.tensor(seq.ids[s + 1:e], dtype=torch.long, device=device)
    pos = torch.arange(s, e - 1, device=device)
    return float(lp[pos, tgt].mean().item())


def double_dissociation(model, sae: TopKSAE, layer: int, builder: Builder,
                        items: List[Item], cots: List[str], part: Partition,
                        device: str, seed: int = 0) -> Dict[str, Dict[str, float]]:
    """Figure 2. Four arms, each measured on BOTH axes:

        control      : no ablation
        ablate M_gen : chain should break, answer should still track it
        ablate M_med : chain should stay fluent, answer should stop tracking it
        ablate random: matched count, should do little

    The claim is the interaction, not either main effect.
    """
    g = torch.Generator().manual_seed(seed)
    n = sae.n_latents
    rnd = torch.randperm(n, generator=g)[: max(len(part.med), 1)].tolist()
    arms = {
        "control": None,
        "ablate_med": {int(j): None for j in part.med},
        "ablate_gen": {int(j): None for j in part.gen},
        "ablate_random": {int(j): None for j in rnd},
    }
    for nm in ("ablate_med", "ablate_gen", "ablate_random"):
        if not arms[nm]:
            raise ValueError(
                f"arm {nm!r} has no latents to ablate; it would be indistinguishable "
                "from the control. Check Partition.check() before running this.")
    out: Dict[str, Dict[str, float]] = {}
    for name, ab in arms.items():
        faith, flu, acc = [], [], []
        for it, cot in zip(items, cots):
            seq = builder.full(it, cot)
            cand = builder.candidate_ids(it)
            gi = it.candidates.index(it.gold)
            with torch.no_grad():
                with Splice(model, sae, layer, ablate=ab):
                    sc = score_candidates(model, seq, cand, device)
                    base_m = margin(sc, gi)
                    acc.append(float(int(sc.argmax()) == gi))
                    cot_pos = seq.span_positions("cot")
                    sc_no = score_candidates(model, seq, cand, device, cot_pos)
                    faith.append(base_m - margin(sc_no, gi))
            flu.append(_fluency(model, sae, layer, seq, device, ab))
        ok = lambda xs: [x for x in xs if x == x]  # noqa: E731
        out[name] = {
            "faithfulness": float(sum(ok(faith)) / max(1, len(ok(faith)))),
            "fluency": float(sum(ok(flu)) / max(1, len(ok(flu)))),
            "accuracy": float(sum(ok(acc)) / max(1, len(ok(acc)))),
            "n_ablated": float(len(ab) if ab else 0),
        }
    return out


# -----------------------------------------------------------------------------
# differential vulnerability: which latents does fine-tuning overwrite?
# -----------------------------------------------------------------------------


@torch.no_grad()
def group_activation(model, sae: TopKSAE, layer: int, seqs: List[Seq],
                     device: str) -> torch.Tensor:
    """Mean activation of every latent over a fixed set of sequences."""
    acc = torch.zeros(sae.n_latents)
    for s in seqs:
        ids = torch.tensor([s.ids], dtype=torch.long, device=device)
        with Splice(model, sae, layer) as sp:
            model(input_ids=ids)
        acc += sp.z.detach()[0].float().mean(0).cpu()
    return acc / max(1, len(seqs))


def differential_vulnerability(base_act: torch.Tensor, ft_act: torch.Tensor,
                               part: Partition, seed: int = 0) -> Dict[str, float]:
    """How much did fine-tuning move each latent group, relative to its baseline?

    DVR = mean relative drift of the mediators over that of the generators. The
    hypothesis is DVR > 1: forgetting eats the machinery that routes the chain
    into the answer before it eats the machinery that writes the chain.

    This does NOT assume the two groups are disjoint in function. Published work
    (arXiv 2608.08168) reports that reasoning and formatting share representations,
    so a differential in how much each group is *disturbed* is a weaker and safer
    claim than a double dissociation, and it is the one this measures.
    """
    # Latents that never fire in the base model have a near-zero denominator, so a
    # relative drift would be enormous and would swamp the group means. Restrict to
    # latents that are actually active at baseline and report how many were dropped.
    scale = base_act.abs()
    thresh = float(scale[scale > 0].median()) * 0.05 if (scale > 0).any() else 0.0
    live = scale > max(thresh, 1e-6)
    drift = torch.zeros_like(scale)
    drift[live] = (ft_act - base_act).abs()[live] / scale[live]

    g = torch.Generator().manual_seed(seed)
    rnd = torch.randperm(len(drift), generator=g)[: max(1, len(part.med))].tolist()

    def m(idx):
        idx = [j for j in idx if bool(live[j])]
        return (float(drift[torch.tensor(idx, dtype=torch.long)].mean())
                if idx else float("nan"))

    d_med, d_gen, d_rnd = m(part.med), m(part.gen), m(rnd)
    n_live_med = sum(1 for j in part.med if bool(live[j]))
    return {
        "drift_med": d_med, "drift_gen": d_gen, "drift_random": d_rnd,
        "drift_all": float(drift[live].mean()) if bool(live.any()) else float("nan"),
        "live_frac": float(live.float().mean()),
        "n_live_med": float(n_live_med),
        "DVR": d_med / d_gen if d_gen > 1e-9 else float("nan"),
        "med_beats_random": bool(d_med > d_rnd),
    }
