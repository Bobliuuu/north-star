# Provenance of KVM pieces

Source of truth (read-only, not modified): `~/kvm-wt/integ`, branch `feat/integration`, commit
`fc0ced7ebce3d107a7a0141a9124e32f519024b3` (merge of demo/executor/compiler/sched/ops; superset of `feat/framework-base`).
Also read: `~/kittens-vm/docs/kvm-2026-11-03-integration.md`, `~/kvm-wt/integ/FRAMEWORK.md` (ABI v2, version variables,
`exact` / `at_least` reads).

Copied: `include/kvm_task_abi.h` (verbatim + provenance line).

NOT used: the KVM executor (`kvm_exec.cu`), compiler or op registry. `moe_fused.cu` is a standalone kernel microbenchmark
whose work items mirror the ABI v2 task fields (see DESIGN.md, "KVM fused draft/verify"). It does not run through the KVM
executor, so VM dispatch overhead is NOT measured here.
