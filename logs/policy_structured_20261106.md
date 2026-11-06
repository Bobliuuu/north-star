# policy-structured work log (2026-11-06, CPU only, no paid GPU)
- Added src/selfimprove/structured_policy.py; Loop/ProposalRequest/cli/modal take track=policy, structured is default for both tracks.
- scripts/build_accept_calibration.py -> results/accept_calibration.jsonl (15 measured rows from results/spec.jsonl, A100). h100_spec.jsonl refused (rc=2, mean-only rows).
- Tests: `cd src && python -m pytest -q selfimprove/tests` 218 passed, 2 skipped; NS_SLOW policy/slow/real subset 66 passed.
- Finding: with the measured table, policy_eval bench for gammas [1,2]/two_pass equals the full baseline space (7448.86 ms/1000 tok, 134.2 tok/s predicted): little room for a >1.03x predicted gain.
