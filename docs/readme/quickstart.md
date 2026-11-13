```bash
make setup     # venv with the pinned versions from SETUP.md
make test      # all CPU unit tests
make check     # correctness: speculative == plain greedy, fused == two-pass reference
make plots     # regenerate plots from raw results/
make bench     # PRINTS the Modal commands for GPU runs (costs money; does not run them)
```

Serving with the measured configuration (exact v2, speculation only where it never lost; needs an H100 and the weights, H26):

```python
from serve import Engine                       # harness/serve.py
eng = Engine(model, Lmax=4096)                 # captures graphs + cost tables once
outs, stats = eng.generate([ids_a, ids_b], max_new=192)   # == plain greedy of the same executor, token for token

from cbatch import ContinuousEngine, Request   # harness/cbatch.py: requests join/leave between decode steps (H27, H28)
ce = ContinuousEngine(model, Lmax=4096, slots=32)
reqs = [Request(i, ids, max_new=192, arrival=t) for i, (ids, t) in enumerate(arrivals)]
stats = ce.run(reqs)                           # each request's tokens == the same request decoded alone
```

GPU runs go through [Modal](https://modal.com) and are launched with `modal run --detach`; see [`SETUP.md`](SETUP.md) for pins and [`RUNS.md`](RUNS.md) for the run ledger.
