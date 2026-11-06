"""Quality command, track 'triton': 24 fixed-seed paired items = 1 - rel_l2(final hidden of a full synthetic exact step with the candidate launch
config, same step with the frozen default launch config) over T in {1,5,13} x 8 context lengths."""
import common
import triton_suite as TS

common.emit({"items": TS.run_quality()})
