"""Oracle-verifier speculation simulator.

The "verifier" is the true next token of a recorded corpus stream (a committed-token oracle):
a draft token is accepted iff it equals the corpus token at that position; the first mismatch
position yields the corpus token as the correction ("bonus"); a fully accepted block also yields
one bonus token. This is an IDEALIZED simulation over human-written files, NOT North output and
NOT a speed result. Draft compute and CPU proposer cost are taken as zero.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from collections import Counter

ALPHAS = (0.05, 0.15, 0.30)


@dataclass
class SimResult:
    name: str
    tokens: int = 0
    steps: int = 0
    attempts: int = 0          # steps in which a nonempty draft was made
    drafted: int = 0
    accepted: int = 0
    first_hit: int = 0         # steps with nonempty draft whose first token was right
    accept_hist: Counter = field(default_factory=Counter)  # accepted draft len per step (all steps)
    # two-ahead stats
    ta_full: int = 0           # steps where whole block accepted
    ta_bonus_hit: int = 0      # full accept AND predicted next(bonus) == true bonus
    ta_rej: int = 0            # steps with a draft that was rejected at some position
    ta_alt_hit: int = 0        # rejected AND the alt-prediction at the reject position == true token

    def summary(self):
        d = {
            "name": self.name,
            "tokens": self.tokens,
            "steps": self.steps,
            "tokens_per_step": self.tokens / self.steps,
            "draft_rate": self.attempts / self.steps,
            "proposal_hit_rate_first": self.first_hit / max(1, self.attempts),
            "token_precision": self.accepted / max(1, self.drafted),
            "drafted_per_step": self.drafted / self.steps,
            "accepted_per_step": self.accepted / self.steps,
            "accept_len_dist": {k: round(v / self.steps, 4) for k, v in sorted(self.accept_hist.items())},
        }
        if self.attempts and (self.ta_full or self.ta_rej):
            d["twoahead"] = {
                "P_full_accept": self.ta_full / self.attempts,
                "P_bonus_pred_given_full": self.ta_bonus_hit / max(1, self.ta_full),
                "P_alt_hit_given_reject": self.ta_alt_hit / max(1, self.ta_rej),
                "outcome_coverage_top1_plus_alt": (self.ta_bonus_hit + self.ta_alt_hit) / self.attempts,
                "outcome_coverage_top1_only": self.ta_bonus_hit / self.attempts,
            }
        for a in ALPHAS:
            # verify pass cost = 1 + alpha*(draft width); AR baseline = 1 per token
            cost = self.steps + a * self.drafted
            d[f"speedup_alpha{a}"] = self.tokens / cost
        return d


def simulate(proposer, docs, kmax=8, theta=0.0, keep_state=True, two_ahead=False, warm_prefix=None):
    """docs: list of token-id lists (one 'request' each). keep_state=True: proposer tables persist
    across docs (warm session); False: reset tables per doc (cold request)."""
    r = SimResult(proposer.name)
    for di, doc in enumerate(docs):
        proposer.start_doc(keep_state=keep_state)
        pos = 0
        n = len(doc)
        while pos < n:
            if two_ahead and hasattr(proposer, "propose") and getattr(proposer, "cfg", None):
                draft, probs, alts = proposer.propose(kmax, theta, with_alt=True)
            else:
                out = proposer.propose(kmax, theta)
                draft, probs = out[0], out[1]
            draft = draft[: n - pos]
            a = 0
            while a < len(draft) and draft[a] == doc[pos + a]:
                a += 1
            r.steps += 1
            if draft:
                r.attempts += 1
                r.drafted += len(draft)
                r.accepted += a
                r.first_hit += (a > 0)
            r.accept_hist[a] += 1
            if two_ahead and draft and a == len(draft) and pos + a < n:
                r.ta_full += 1
                # predicted bonus = proposer's next prediction after the block
                nxt = proposer.propose(kmax=len(draft) + 1, theta=0.0)[0]
                if len(nxt) > len(draft) and nxt[len(draft)] == doc[pos + a]:
                    r.ta_bonus_hit += 1
            if two_ahead and draft and a < len(draft):
                r.ta_rej += 1
                if alts[a] == doc[pos + a]:
                    r.ta_alt_hit += 1
            take = min(a + 1, n - pos)
            for j in range(take):
                proposer.commit(doc[pos + j])
            r.tokens += take
            pos += take
    return r
