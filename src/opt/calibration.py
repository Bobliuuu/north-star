"""Acceptance calibration table: loader + validator + expectations.

JSONL interface (one row per measured draft configuration and workload):
  {"workload": "code",                  # workload kind the trace came from
   "attn_budget": 0,                     # 0 = full attention, else keys kept on global layers in the draft
   "r": 4,                               # draft expert budget
   "skip_layers": [8, 12, 16],           # draft-only skipped layers ([] = none)
   "accept_len_pmf": [p0, p1, ..., pG],  # P(A=j) j=0..G; last bin = P(A>=G). A = length of the longest draft prefix
                                         #   the verifier accepts (greedy), measured with UNBOUNDED lookahead G
   "n_samples": 12000,                   # number of verify rounds behind the pmf
   "carry_hit_given_full": 0.5,          # optional: P(fused speculative draft token is reusable | all accepted)
   "provenance": "measured",             # "measured" | "SYNTHETIC"  (SYNTHETIC rows are tests/regime-scans only)
   "source": "run id / file"}
Truncation to block length gamma<=G is exact for greedy verification: accepted = min(A, gamma).
Committed tokens per cycle = 1 + min(A, gamma) (correction or bonus token).
"""
from __future__ import annotations

import json
from dataclasses import dataclass


class CalibrationError(ValueError):
    pass


@dataclass(frozen=True)
class AcceptEntry:
    workload: str
    key: tuple                  # (attn_budget, r, sorted skip tuple)
    pmf: tuple
    n_samples: int
    carry_hit_given_full: float
    provenance: str
    source: str

    @property
    def gmax(self):
        return len(self.pmf) - 1

    def surv(self, j):
        """P(A >= j)."""
        return sum(self.pmf[j:]) if j <= self.gmax else 0.0

    def expected_committed(self, gamma):
        if gamma > self.gmax:
            raise CalibrationError(f"gamma={gamma} exceeds calibrated lookahead {self.gmax}")
        return 1.0 + sum(self.surv(j) for j in range(1, gamma + 1))

    def p_full(self, gamma):
        return self.surv(gamma)

    def carry_prob(self, gamma):
        return self.p_full(gamma) * self.carry_hit_given_full


class AcceptTable:
    def __init__(self, entries):
        self.entries = {}
        for e in entries:
            k = (e.workload, e.key)
            if k in self.entries:
                raise CalibrationError(f"duplicate calibration row {k}")
            self.entries[k] = e

    @classmethod
    def from_jsonl(cls, path):
        out = []
        with open(path) as f:
            for i, line in enumerate(f, 1):
                if line.strip():
                    out.append(parse_row(json.loads(line), f"{path}:{i}"))
        return cls(out)

    def get(self, workload, key):
        return self.entries.get((workload, key))

    def keys(self, workload):
        return [k for (w, k) in self.entries if w == workload]

    def workloads(self):
        return sorted({w for (w, _) in self.entries})

    def any_synthetic(self):
        return any(e.provenance.upper() == "SYNTHETIC" for e in self.entries.values())

    def provenance(self):
        return sorted({e.provenance for e in self.entries.values()})

    def fingerprint(self):
        return f"{len(self.entries)}rows/{','.join(self.provenance())}"


def parse_row(d, where="row"):
    try:
        pmf = tuple(float(x) for x in d["accept_len_pmf"])
        skip = tuple(sorted(int(x) for x in d.get("skip_layers", [])))
        if len(pmf) < 2 or any(p < -1e-12 for p in pmf) or abs(sum(pmf) - 1.0) > 1e-6:
            raise CalibrationError(f"{where}: pmf must be >=2 bins, nonnegative, sum to 1 (sum={sum(pmf)})")
        if 0 in skip:
            raise CalibrationError(f"{where}: layer 0 (dense prefix) cannot be skipped")
        ch = float(d.get("carry_hit_given_full", 0.0))
        if not 0.0 <= ch <= 1.0:
            raise CalibrationError(f"{where}: carry_hit_given_full outside [0,1]")
        prov = d.get("provenance")
        if prov is None:
            raise CalibrationError(f"{where}: missing provenance ('measured' or 'SYNTHETIC')")
        return AcceptEntry(d["workload"], (int(d["attn_budget"]), int(d["r"]), skip), pmf,
                           int(d.get("n_samples", 0)), ch, prov, d.get("source", ""))
    except KeyError as e:
        raise CalibrationError(f"{where}: missing field {e}") from None
