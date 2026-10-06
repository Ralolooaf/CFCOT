"""
Model engine: span-exact sequence building, candidate scoring, attention knockout,
logit lens.

Design rules that keep this correct:
  * every analysis sequence is built by CONCATENATING token-id lists we control,
    so span boundaries are exact by construction -- never by searching decoded text
  * batched *generation* uses left padding; batched *scoring* uses right padding.
    They are never mixed.
  * all metric arithmetic happens in float32 even when the model runs in fp16/bf16
  * the attention-knockout mechanism is self-tested at load time (see
    `verify_knockout`); if the installed transformers version ignores 4-D masks the
    run aborts loudly instead of silently reporting zeros.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Dict, Sequence, Optional, Tuple

import torch

from .tasks import Item

INSTRUCTION = (
    "Solve the problem by reasoning one step at a time. "
    "Write one short step per line, then stop."
)
ANSWER_TAG = "Answer:"


# -----------------------------------------------------------------------------
# device / dtype
# -----------------------------------------------------------------------------


def pick_dtype(device: str) -> torch.dtype:
    """bf16 on Ampere+ (L4/A100/H100), fp16 on Turing (T4) / Pascal (P100)."""
    if device != "cuda" or not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def describe_device() -> str:
    if not torch.cuda.is_available():
        return "cpu (no CUDA)"
    p = torch.cuda.get_device_properties(0)
    return (f"{p.name} | {p.total_memory/1e9:.1f} GB | "
            f"bf16={torch.cuda.is_bf16_supported()} | sm_{p.major}{p.minor}")


# -----------------------------------------------------------------------------
# span-exact sequence building
# -----------------------------------------------------------------------------

SPAN_ORDER = ("head", "rules", "question", "genprompt", "cot", "bridge")


@dataclass
class Seq:
    ids: List[int]
    spans: Dict[str, Tuple[int, int]]   # name -> [start, end)

    def __len__(self) -> int:
        return len(self.ids)

    def span_positions(self, *names: str) -> List[int]:
        out: List[int] = []
        for n in names:
            if n not in self.spans:
                raise KeyError(f"no span {n!r}; have {sorted(self.spans)}")
            s, e = self.spans[n]
            out.extend(range(s, e))
        return out


class Builder:
    """Turns (Item, cot_text) into token ids with exact spans."""

    def __init__(self, tok):
        self.tok = tok
        self.has_template = getattr(tok, "chat_template", None) is not None

    def _enc(self, text: str) -> List[int]:
        if text == "":
            return []
        return self.tok.encode(text, add_special_tokens=False)

    def _template_parts(self, user_content: str) -> Tuple[str, str]:
        """Return (prefix, suffix) such that prefix + user_content + suffix is the
        fully rendered chat prompt with a generation prompt appended."""
        if not self.has_template:
            bos = self.tok.bos_token or ""
            return bos, "\n" + ANSWER_TAG.replace(":", "") + " reasoning:\n"
        rendered = self.tok.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False, add_generation_prompt=True,
        )
        idx = rendered.find(user_content)
        if idx < 0:
            # template mangled the content (rare); fall back to raw mode
            self.has_template = False
            bos = self.tok.bos_token or ""
            return bos, "\n" + ANSWER_TAG.replace(":", "") + " reasoning:\n"
        return rendered[:idx], rendered[idx + len(user_content):]

    def prompt(self, item: Item) -> Seq:
        """Sequence up to and including the generation prompt (no CoT yet)."""
        user_content = (
            f"{INSTRUCTION}\n\n{item.rules_text}\n\n{item.question_text}"
        )
        pre, suf = self._template_parts(user_content)
        pieces = [
            ("head", pre + INSTRUCTION + "\n\n"),
            ("rules", item.rules_text),
            ("question", "\n\n" + item.question_text),
            ("genprompt", suf),
        ]
        return self._assemble(pieces)

    def full(self, item: Item, cot_text: str) -> Seq:
        """Prompt + CoT + bridge. The next token after this sequence is the answer."""
        p = self.prompt(item)
        cot_ids = self._enc(cot_text)
        bridge_ids = self._enc("\n" + ANSWER_TAG + " ")
        ids = list(p.ids) + cot_ids + bridge_ids
        spans = dict(p.spans)
        n = len(p.ids)
        spans["cot"] = (n, n + len(cot_ids))
        spans["bridge"] = (n + len(cot_ids), n + len(cot_ids) + len(bridge_ids))
        return Seq(ids, spans)

    def _assemble(self, pieces: Sequence[Tuple[str, str]]) -> Seq:
        ids: List[int] = []
        spans: Dict[str, Tuple[int, int]] = {}
        for name, text in pieces:
            chunk = self._enc(text)
            spans[name] = (len(ids), len(ids) + len(chunk))
            ids.extend(chunk)
        return Seq(ids, spans)

    def raw(self, text: str) -> Seq:
        """A Seq from arbitrary text, for activation collection over corpora that
        are not task items (e.g. the fine-tuning domain)."""
        ids = self._enc(text) or self._enc(" ")
        return Seq(ids, {"rules": (0, len(ids))})

    def candidate_ids(self, item: Item) -> List[List[int]]:
        """Token ids for each candidate as it would appear after 'Answer: '."""
        return [self._enc(c) for c in item.candidates]

    def distinct_first_tokens(self, item: Item) -> bool:
        ids = self.candidate_ids(item)
        return len({c[0] for c in ids}) == len(ids)


def dedupe_candidates(builder: "Builder", item: Item) -> Optional[Item]:
    """Drop candidates that collide with another on their FIRST token.

    Attribution and the logit lens both score candidates by first token only (one
    forward per backward pass), so a collision makes the margin meaningless. Rather
    than discard the whole item, keep gold plus a maximal collision-free subset.
    Returns None when fewer than two candidates survive.
    """
    ids = builder.candidate_ids(item)
    gold_i = item.candidates.index(item.gold)
    keep, seen = [gold_i], {ids[gold_i][0]}
    for i, toks in enumerate(ids):
        if i == gold_i or toks[0] in seen:
            continue
        seen.add(toks[0])
        keep.append(i)
    if len(keep) < 2:
        return None
    keep.sort()
    return Item(task=item.task, rules_text=item.rules_text,
                question_text=item.question_text, gold=item.gold,
                candidates=[item.candidates[i] for i in keep],
                hops=item.hops, gold_cot=item.gold_cot, meta=dict(item.meta))


# -----------------------------------------------------------------------------
# attention masks
# -----------------------------------------------------------------------------


def build_mask(lengths: Sequence[int], total: int, dtype: torch.dtype,
               blocked: Optional[Sequence[Sequence[int]]] = None,
               device: str = "cpu") -> torch.Tensor:
    """(B, 1, T, T) additive causal mask with right padding and optional knockout.

    lengths[b] is the number of REAL tokens in row b; the rest is right padding.
    blocked[b] is a list of key positions that row b may not attend to.
    """
    B = len(lengths)
    neg = torch.finfo(dtype).min
    m = torch.full((B, 1, total, total), neg, dtype=dtype, device=device)
    causal = torch.tril(torch.ones(total, total, dtype=torch.bool, device=device))
    for b, L in enumerate(lengths):
        allow = causal.clone()
        if L < total:
            allow[:, L:] = False          # never attend to right padding
        if blocked is not None and len(blocked[b]) > 0:
            idx = torch.as_tensor(sorted(set(blocked[b])), dtype=torch.long,
                                  device=device)
            allow[:, idx] = False
        # every row must keep at least one visible key, else softmax is NaN.
        # position i always keeps itself.
        allow[torch.arange(total, device=device), torch.arange(total, device=device)] = True
        m[b, 0][allow] = 0.0
    return m


def verify_knockout(model, device: str) -> None:
    """Abort loudly if 4-D attention masks are ignored by this transformers build.

    Three checks:
      1. an explicit causal 4-D mask reproduces the default 2-D behaviour exactly
      2. blocking a middle span actually changes downstream logits
      3. positions BEFORE the blocked span are untouched (causality holds)
    """
    model.eval()
    dtype = next(model.parameters()).dtype
    V = int(model.get_input_embeddings().weight.shape[0])
    T = 12
    ids = torch.arange(T, device=device).remainder(max(V - 1, 1)).unsqueeze(0) + 1
    ids = ids.clamp(max=V - 1)
    with torch.no_grad():
        base = model(input_ids=ids).logits.float()
        m_id = build_mask([T], T, dtype, None, device)
        same = model(input_ids=ids, attention_mask=m_id).logits.float()
        m_bl = build_mask([T], T, dtype, [[4, 5, 6]], device)
        blk = model(input_ids=ids, attention_mask=m_bl).logits.float()

    # Compare RELATIVE to the logit scale, not in absolute terms. A 2-D mask takes
    # SDPA's is_causal fast path while a 4-D mask goes through the attn_mask path;
    # they are different kernels, so in fp16 they disagree by an amount that grows
    # with the logits. An absolute threshold would abort a perfectly good run on a
    # real model within the first minute.
    scale = max(float(base.abs().max().item()), 1.0)
    d_id = (base - same).abs().max().item() / scale
    d_bl = (base - blk).abs().max().item() / scale
    d_pre = (base[:, :4] - blk[:, :4]).abs().max().item() / scale
    same_argmax = bool((base.argmax(-1) == same.argmax(-1)).all())
    tol = 5e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-4

    if d_id > tol or not same_argmax:
        raise RuntimeError(
            f"an explicit causal 4-D mask does not reproduce the default causal "
            f"behaviour (relative max diff {d_id:.3g}, argmax match "
            f"{same_argmax}). Knockout results would not be trustworthy on this "
            f"transformers build. Try attn_implementation='eager'."
        )
    if d_bl <= max(tol, 1e-3):
        raise RuntimeError(
            "the 4-D attention mask appears to be IGNORED: blocking a span changed "
            "nothing. Knockout would silently report zeros for every measurement. "
            "Aborting rather than producing a flat rho_CoT curve."
        )
    if d_pre > tol:
        raise RuntimeError(
            f"blocking a later span changed EARLIER logits (relative {d_pre:.3g}); "
            "causal masking is broken on this build. Aborting."
        )


# -----------------------------------------------------------------------------
# candidate scoring
# -----------------------------------------------------------------------------


@torch.no_grad()
def score_candidates(model, seq: Seq, cand_ids: List[List[int]],
                     device: str, blocked: Optional[Sequence[int]] = None,
                     ) -> torch.Tensor:
    """Log-probability of each candidate's FIRST token at the answer position.

    ONE forward pass of ONE row, not one row per candidate. The candidates all
    share the same prefix, so replicating that prefix per candidate multiplied the
    cost by the number of options -- on a T4 with a 1.5B model that was the single
    largest cost in the whole pipeline.

    Scoring by first token is exact provided the candidates' first tokens differ,
    which `require_distinct_first_tokens` enforces when the eval set is built. It
    is the standard multiple-choice scoring rule: the model's distribution over the
    next token, restricted to the option set.

    `blocked` are key positions inside `seq` that no query may attend to.
    """
    if any(len(c) == 0 for c in cand_ids):
        raise ValueError("a candidate tokenised to zero tokens")
    firsts = [c[0] for c in cand_ids]
    if len(set(firsts)) != len(firsts):
        raise ValueError(
            "candidates share a first token; build the eval set through "
            "require_distinct_first_tokens() so scoring stays well defined")
    dtype = next(model.parameters()).dtype
    T = len(seq.ids)
    inp = torch.tensor([seq.ids], dtype=torch.long, device=device)
    mask = build_mask([T], T, dtype, [list(blocked or [])], device)
    logits = model(input_ids=inp, attention_mask=mask).logits[0, -1].float()
    lp = torch.log_softmax(logits, dim=-1)
    return lp[torch.tensor(firsts, dtype=torch.long, device=device)]


def require_distinct_first_tokens(builder: "Builder", items: List[Item]
                                  ) -> List[Item]:
    """Repair or drop items whose options collide on their first token."""
    out, dropped = [], 0
    for it in items:
        if builder.distinct_first_tokens(it):
            out.append(it)
            continue
        fixed = dedupe_candidates(builder, it)
        if fixed is None:
            dropped += 1
        else:
            out.append(fixed)
    if dropped:
        print(f"  [data  ] dropped {dropped}/{len(items)} items "
              "(fewer than two options survived first-token deduplication)")
    if not out:
        raise RuntimeError(
            "no item had two options with distinct first tokens under this "
            "tokeniser; the answer set is not scoreable")
    return out


def margin(scores: torch.Tensor, gold_idx: int) -> float:
    """gold log-prob minus logsumexp of the rest -- a signed, calibrated margin."""
    if scores.numel() < 2:
        return float(scores[gold_idx].item())
    others = torch.cat([scores[:gold_idx], scores[gold_idx + 1:]])
    return float(scores[gold_idx].item() - torch.logsumexp(others, dim=0).item())


# -----------------------------------------------------------------------------
# rho_CoT : where does the answer's causal support live?
# -----------------------------------------------------------------------------


@torch.no_grad()
def rho_cot(model, seq: Seq, cand_ids: List[List[int]], gold_idx: int,
            device: str, base_margin: Optional[float] = None) -> Dict[str, float]:
    """Share of the answer margin destroyed by knocking out the CoT, relative to
    the total destroyed by knocking out either the CoT or the rules.

    rho = |d_cot| / (|d_cot| + |d_rules|),  d_x = margin_base - margin_without_x

    rho -> 1 : the answer is computed FROM the chain (reasoning)
    rho -> 0 : the answer is recomputed from the premises at answer time (retrieval)
    """
    base = (base_margin if base_margin is not None
            else margin(score_candidates(model, seq, cand_ids, device), gold_idx))
    cot_pos = seq.span_positions("cot")
    rul_pos = seq.span_positions("rules")
    m_no_cot = margin(score_candidates(model, seq, cand_ids, device, cot_pos), gold_idx)
    m_no_rul = margin(score_candidates(model, seq, cand_ids, device, rul_pos), gold_idx)
    d_cot = base - m_no_cot
    d_rul = base - m_no_rul
    denom = abs(d_cot) + abs(d_rul)
    return {
        "margin_base": base,
        "margin_no_cot": m_no_cot,
        "margin_no_rules": m_no_rul,
        "delta_cot": d_cot,
        "delta_rules": d_rul,
        "rho_cot": (abs(d_cot) / denom) if denom > 1e-9 else float("nan"),
    }


# -----------------------------------------------------------------------------
# logit lens : commitment depth d*
# -----------------------------------------------------------------------------


def _final_norm_and_head(model):
    inner = getattr(model, "model", None)
    norm = getattr(inner, "norm", None) if inner is not None else None
    if norm is None:
        for path in ("transformer.ln_f", "gpt_neox.final_layer_norm", "model.final_layernorm"):
            obj = model
            ok = True
            for part in path.split("."):
                obj = getattr(obj, part, None)
                if obj is None:
                    ok = False
                    break
            if ok:
                norm = obj
                break
    head = getattr(model, "lm_head", None) or getattr(model, "embed_out", None)
    if norm is None or head is None:
        raise RuntimeError(
            "could not locate final norm / lm_head for logit lens; "
            "add this architecture to _final_norm_and_head()"
        )
    return norm, head


@torch.no_grad()
def commitment_depth(model, seq: Seq, cand_ids: List[List[int]], gold_idx: int,
                     device: str) -> Dict[str, float]:
    """Shallowest layer whose logit-lens argmax over candidates equals the FINAL
    layer's argmax, and stays equal for every deeper layer.

    Uses the first token of each candidate. Candidates whose first tokens collide
    are reported via `first_token_collision` so the caller can drop the item.
    """
    firsts = [c[0] for c in cand_ids]
    collision = len(set(firsts)) != len(firsts)
    norm, head = _final_norm_and_head(model)
    ids = torch.tensor([seq.ids], dtype=torch.long, device=device)
    out = model(input_ids=ids, output_hidden_states=True)
    hs = out.hidden_states                      # (L+1) tensors, [0] = embeddings
    L = len(hs) - 1
    sel = torch.tensor(firsts, dtype=torch.long, device=device)

    preds = []
    for li in range(L + 1):
        h = hs[li][:, -1, :]
        logit = head(norm(h)).float()[0, sel]
        preds.append(int(logit.argmax().item()))
    final = preds[-1]
    d_star = L
    for li in range(L + 1):
        if all(p == final for p in preds[li:]):
            d_star = li
            break
    return {
        "d_star": float(d_star),
        "d_star_frac": float(d_star) / float(L),
        "n_layers": float(L),
        "lens_final_pred": float(final),
        "lens_correct": float(final == gold_idx),
        "first_token_collision": float(collision),
    }


# -----------------------------------------------------------------------------
# generation
# -----------------------------------------------------------------------------


TRUNCATED: List[float] = []


@torch.no_grad()
def generate_cots(model, tok, items: List[Item], builder: Builder, device: str,
                  max_new_tokens: int = 160, batch_size: int = 8) -> List[str]:
    """Greedy CoT generation. Deterministic, so checkpoints stay comparable."""
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    TRUNCATED.clear()
    prev_side = tok.padding_side
    tok.padding_side = "left"
    outs: List[str] = []
    try:
        for i in range(0, len(items), batch_size):
            chunk = items[i: i + batch_size]
            seqs = [builder.prompt(it) for it in chunk]
            T = max(len(s) for s in seqs)
            pad_id = tok.pad_token_id
            inp = torch.full((len(chunk), T), pad_id, dtype=torch.long, device=device)
            att = torch.zeros((len(chunk), T), dtype=torch.long, device=device)
            for b, s in enumerate(seqs):
                inp[b, T - len(s):] = torch.tensor(s.ids, dtype=torch.long, device=device)
                att[b, T - len(s):] = 1
            gen = model.generate(
                input_ids=inp, attention_mask=att,
                max_new_tokens=max_new_tokens, do_sample=False,
                num_beams=1, pad_token_id=pad_id, use_cache=True,
            )
            new = gen[:, T:]
            eos = {tok.eos_token_id, pad_id}
            for row in new:
                text = tok.decode(row, skip_special_tokens=True)
                # A chain that ran into max_new_tokens is a fragment, not a chain.
                # Perturbing a fragment cannot change the answer, so every
                # faithfulness measure silently reads zero. Record it.
                hit_cap = (row.shape[0] >= max_new_tokens and
                           int(row[-1].item()) not in eos)
                TRUNCATED.append(float(hit_cap))
                outs.append(_clean_cot(text))
    finally:
        tok.padding_side = prev_side
    return outs


def _clean_cot(text: str) -> str:
    """Keep the reasoning, drop anything from the model's own answer onward.

    Reasoning-tuned models (R1-Distill, Qwen3 thinking) wrap their chain in
    <think>...</think>. Left in place those tags become part of the CoT span, and
    a truncated chain would end mid-tag -- so unwrap them: keep the contents, drop
    the markers, and if the block never closed keep what there is.
    """
    if "<think>" in text:
        after = text.split("<think>", 1)[1]
        text = after.split("</think>", 1)[0] if "</think>" in after else after
    for tag in ("</think>", "<think>", "<|thinking|>", "</|thinking|>"):
        text = text.replace(tag, " ")
    for stop in (ANSWER_TAG, "answer:", "ANSWER:", "\\boxed{"):
        j = text.find(stop)
        if j >= 0:
            text = text[:j]
    lines = [ln.rstrip() for ln in text.split("\n")]
    lines = [ln for ln in lines if ln.strip()]
    return "\n".join(lines).strip()


def degeneracy(text: str) -> Dict[str, float]:
    """Repetition diagnostics. Degenerate chains are trivially unfaithful and must
    be excluded from faithfulness statistics (and reported separately)."""
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    toks = text.split()
    n = len(toks)
    d1 = len(set(toks)) / n if n else 0.0
    tri = [tuple(toks[i:i + 3]) for i in range(max(0, n - 2))]
    d3 = len(set(tri)) / len(tri) if tri else 0.0
    dup_lines = 1.0 - (len(set(lines)) / len(lines)) if lines else 0.0
    # A reasoning model legitimately revisits ("wait, let me check"). What matters
    # is a genuine LOOP: the same line repeating back to back.
    run, worst_run = 1, 1
    for a, b in zip(lines, lines[1:]):
        run = run + 1 if a == b else 1
        worst_run = max(worst_run, run)
    return {
        "n_lines": float(len(lines)),
        "n_tokens": float(n),
        "distinct_1": d1,
        "distinct_3": d3,
        "dup_line_frac": dup_lines,
        "max_consecutive_repeat": float(worst_run),
        "is_degenerate": float(
            worst_run >= 4 or dup_lines > 0.60 or
            (n >= 12 and d3 < 0.35) or len(lines) == 0),
    }
