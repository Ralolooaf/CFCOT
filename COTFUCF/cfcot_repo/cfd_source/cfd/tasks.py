"""
Synthetic reasoning tasks with guaranteed load-bearing CoT.

Design contract for every task:
  * closed candidate answer set  -> exact scoring, no string parsing of answers
  * ground-truth intermediate at every hop -> probes + error injection are exact
  * random surface tokens per item -> zero memorisation / retrieval pathway
  * prompt splits cleanly into (rules span, question span) -> rho_CoT is well defined

Everything here is pure Python. No torch, no model. Fully unit-testable.
"""
from __future__ import annotations

import random
import string
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple
import re

# ----------------------------------------------------------------------------- 
# item container
# -----------------------------------------------------------------------------


@dataclass
class Item:
    """One evaluation item.

    rules_text / question_text are kept separate so the tokeniser can build
    exact spans without any substring searching.
    """
    task: str
    rules_text: str
    question_text: str
    gold: str                     # correct answer, a member of `candidates`
    candidates: List[str]         # closed answer set, gold included
    hops: List[str]               # ground-truth intermediate value after each step
    gold_cot: str                 # a reference chain (used for teacher-forced conditions)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# -----------------------------------------------------------------------------
# nonsense token vocabulary
# -----------------------------------------------------------------------------

_CONS = "bdfgklmnprstvz"
_VOWL = "aeiou"


def _nonsense(rng: random.Random) -> str:
    """CVC or CVCC syllable, e.g. 'qub', 'zim', 'flor'. Stable across runs given rng."""
    w = rng.choice(_CONS) + rng.choice(_VOWL) + rng.choice(_CONS)
    if rng.random() < 0.35:
        w += rng.choice(_CONS)
    return w


def _unique_tokens(rng: random.Random, n: int) -> List[str]:
    """Distinct symbols, none of which is a substring of another.

    The substring condition matters: error injection replaces a step's result by
    string surgery, and if 'nos' were a substring of 'nosm' the wrong span could be
    rewritten, silently producing a corruption that is not the one we recorded.
    """
    out: List[str] = []
    guard = 0
    while len(out) < n:
        guard += 1
        if guard > 200000:
            raise RuntimeError(
                f"could not sample {n} mutually non-substring tokens; "
                "lower hops/distractors/n_candidates")
        w = _nonsense(rng)
        if any(w in o or o in w for o in out):
            continue
        out.append(w)
    return out


# -----------------------------------------------------------------------------
# T1: symbolic chain composition
# -----------------------------------------------------------------------------


def make_chain_item(rng: random.Random, hops: int = 3, distractors: int = 3,
                    n_candidates: int = 6) -> Item:
    """`a -> b`, `b -> c`, ... follow `hops` arrows from a start symbol.

    The answer is only obtainable by traversing; distractor rules are shuffled in
    and rule order is randomised so 'read the last rule' fails.
    """
    if hops < 1:
        raise ValueError("hops must be >= 1")
    n_chain = hops + 1
    toks = _unique_tokens(rng, n_chain + 2 * distractors + n_candidates)
    chain = toks[:n_chain]
    rest = toks[n_chain:]

    rules = [(chain[i], chain[i + 1]) for i in range(hops)]
    d_pool = rest[: 2 * distractors]
    for i in range(distractors):
        rules.append((d_pool[2 * i], d_pool[2 * i + 1]))
    rng.shuffle(rules)

    gold = chain[-1]
    # candidates: gold + other chain members + spare tokens. Never fewer than 2.
    pool = [t for t in chain[:-1]] + rest[2 * distractors:]
    rng.shuffle(pool)
    cands = [gold] + pool[: max(1, n_candidates - 1)]
    rng.shuffle(cands)

    rules_text = "Rules:\n" + "\n".join(f"  {a} -> {b}" for a, b in rules)
    question_text = (
        f"Start at {chain[0]} and follow {hops} arrow"
        f"{'s' if hops > 1 else ''}.\n"
        f"Options: {', '.join(cands)}"
    )
    cot_lines = [
        f"Step {i+1}: {chain[i]} -> {chain[i+1]}." for i in range(hops)
    ]
    return Item(
        task="chain",
        rules_text=rules_text,
        question_text=question_text,
        gold=gold,
        candidates=cands,
        hops=chain[1:],                       # value after each step
        gold_cot="\n".join(cot_lines),
        meta={"hops": hops, "distractors": distractors, "start": chain[0]},
    )


# -----------------------------------------------------------------------------
# T2: modular arithmetic chain
# -----------------------------------------------------------------------------


def make_modarith_item(rng: random.Random, hops: int = 3, modulus: int = 11,
                       n_candidates: int = 6) -> Item:
    """x = c; then `hops` operations mod m. Forces computation, not just routing."""
    if hops < 1:
        raise ValueError("hops must be >= 1")
    if modulus < 5:
        raise ValueError("modulus must be >= 5")
    x = rng.randrange(1, modulus)
    ops, vals = [], []
    cur = x
    for _ in range(hops):
        kind = rng.choice(["+", "*"])
        k = rng.randrange(2, modulus)
        cur = (cur + k) % modulus if kind == "+" else (cur * k) % modulus
        ops.append((kind, k))
        vals.append(cur)

    gold = str(cur)
    others = [str(v) for v in range(modulus) if str(v) != gold]
    rng.shuffle(others)
    cands = [gold] + others[: max(1, n_candidates - 1)]
    rng.shuffle(cands)

    rules_text = (
        f"Work modulo {modulus}.\n"
        "Program:\n"
        f"  x = {x}\n"
        + "\n".join(f"  x = (x {k} {v}) mod {modulus}" for k, v in ops)
    )
    question_text = "What is the final value of x?\nOptions: " + ", ".join(cands)

    cot_lines, prev = [], x
    for i, (kind, k) in enumerate(ops):
        cot_lines.append(
            f"Step {i+1}: ({prev} {kind} {k}) mod {modulus} = {vals[i]}."
        )
        prev = vals[i]
    return Item(
        task="modarith",
        rules_text=rules_text,
        question_text=question_text,
        gold=gold,
        candidates=cands,
        hops=[str(v) for v in vals],
        gold_cot="\n".join(cot_lines),
        meta={"hops": hops, "modulus": modulus, "start": x},
    )


TASK_BUILDERS = {"chain": make_chain_item, "modarith": make_modarith_item}


def build_dataset(task: str, n: int, seed: int = 0, **kw) -> List[Item]:
    if task not in TASK_BUILDERS:
        raise KeyError(f"unknown task {task!r}; have {sorted(TASK_BUILDERS)}")
    rng = random.Random(seed)
    return [TASK_BUILDERS[task](rng, **kw) for _ in range(n)]


# -----------------------------------------------------------------------------
# chain perturbations (all operate on TEXT, applied before re-tokenising)
# -----------------------------------------------------------------------------


def split_steps(cot: str) -> List[str]:
    """Split a chain into steps. Lines are the unit; blank lines dropped."""
    return [ln for ln in cot.split("\n") if ln.strip()]


def truncate_cot(cot: str, frac: float) -> str:
    """Early answering: keep the first `frac` of the steps."""
    steps = split_steps(cot)
    if not steps:
        return ""
    k = max(0, min(len(steps), int(round(frac * len(steps)))))
    return "\n".join(steps[:k])


def filler_cot(cot: str, frac: float, filler: str = "...") -> str:
    """Filler substitution: replace steps after `frac` with content-free tokens."""
    steps = split_steps(cot)
    if not steps:
        return ""
    k = max(0, min(len(steps), int(round(frac * len(steps)))))
    return "\n".join(steps[:k] + [filler] * (len(steps) - k))


def shuffle_cot(cot: str, rng: random.Random) -> str:
    """Same tokens, scrambled order. Tests semantic vs positional use of the CoT."""
    steps = split_steps(cot)
    if len(steps) < 2:
        return cot
    out = steps[:]
    for _ in range(20):
        rng.shuffle(out)
        if out != steps:
            break
    return "\n".join(out)


_VALUE_RE = re.compile(r"\d+|\b[a-z]{3,5}\b")


def value_line_indices(item: Item, cot: str, max_lines: int = 6) -> List[int]:
    """Indices of the lines that actually carry an intermediate result.

    A reasoning model writes twenty lines for a three-step problem, most of them
    preamble ("Okay, let me work through this"). Corrupting line 0 and line 1 and
    concluding that the answer does not depend on the chain measures nothing. Pick
    the lines that contain a number or a chain symbol, and prefer the later ones,
    because that is where the result the answer would have to read actually sits.
    """
    steps = split_steps(cot)
    syms = set(item.hops) | set(item.candidates)
    scored: List[Tuple[int, int]] = []
    for i, ln in enumerate(steps):
        hits = len(_VALUE_RE.findall(ln)) + 3 * sum(1 for s in syms if s in ln)
        if hits:
            scored.append((i, hits))
    if not scored:
        return []
    # later lines first: the final intermediate is the one the answer must use
    scored.sort(key=lambda t: (-t[0],))
    return [i for i, _ in scored[:max_lines]]


def corrupt_all_mentions(item: Item, cot: str, hop_idx: int,
                         rng: random.Random) -> Optional[str]:
    """Replace EVERY occurrence of one intermediate value throughout the chain.

    A verbose chain states the same value three or four times ("12 mod 11 is 1",
    "let me double-check, yes it is 1", "so after the first step we have 1").
    Corrupting one line leaves the other mentions intact, and the model simply
    reads the value from elsewhere -- so a low flip rate measures redundancy, not
    independence from the chain. Corrupting all mentions is the decisive test: if
    the answer still does not move, the chain genuinely is not being read.
    """
    if not (0 <= hop_idx < len(item.hops)):
        return None
    truth = item.hops[hop_idx]
    pool = [c for c in item.candidates if c != truth] + \
           [h for h in item.hops if h != truth]
    pool = [w for w in dict.fromkeys(pool)]
    if not pool:
        return None
    wrong = rng.choice(pool)
    pattern = re.compile(rf"(?<![A-Za-z0-9]){re.escape(truth)}(?![A-Za-z0-9])")
    out, n = pattern.subn(wrong, cot)
    return out if n > 0 and out != cot else None


def corrupt_cot(item: Item, cot: str, step_idx: int, rng: random.Random) -> Optional[str]:
    """Error injection: replace the *result* of step `step_idx` with a wrong value.

    Returns None when the step cannot be corrupted (e.g. index out of range or the
    value does not literally appear in that step's line).
    """
    steps = split_steps(cot)
    if not (0 <= step_idx < len(steps)):
        return None
    truth = item.hops[step_idx] if step_idx < len(item.hops) else ""
    wrong_pool = [c for c in item.candidates if c != truth] + \
                 [h for h in item.hops if h != truth]
    wrong_pool = [w for w in dict.fromkeys(wrong_pool)]
    if not wrong_pool:
        return None
    wrong = rng.choice(wrong_pool)
    line = steps[step_idx]
    if truth and truth in line:
        # replace only the LAST occurrence: that is the step's result
        head, _, tail = line.rpartition(truth)
        steps[step_idx] = head + wrong + tail
        return "\n".join(steps)

    # The model wrote its own chain, which need not contain the ground-truth
    # intermediate -- especially when it is wrong. Corrupt whatever value IT
    # claimed instead: that is the quantity the answer would have to depend on.
    import re as _re
    toks = list(_re.finditer(r"[A-Za-z]{2,}|\d+", line))
    if not toks:
        return None
    m = toks[-1]
    claimed = m.group(0)
    alt = next((w for w in wrong_pool if w != claimed), None)
    if alt is None:
        alt = "".join(reversed(claimed)) if claimed.isalpha() else str(
            (int(claimed) + 1) % 100)
    if alt == claimed:
        return None
    steps[step_idx] = line[:m.start()] + alt + line[m.end():]
    return "\n".join(steps)
