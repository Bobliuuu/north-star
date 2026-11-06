import pytest
from selfimprove import diffs as D
from selfimprove.policy import Scope, ScopeRule, Limits
from .conftest import CU, POLICY

SC = Scope(rules=[ScopeRule("src/kvm/moe_fused.cu", regions_only=True), ScopeRule("src/opt/policy_tables/*.json"),
                  ScopeRule("src/opt/new_*.py", allow_create=True)])
RD = {"src/kvm/moe_fused.cu": CU, "src/opt/policy_tables/sched.json": POLICY}.get

GOOD = """--- a/src/kvm/moe_fused.cu
+++ b/src/kvm/moe_fused.cu
@@ -3,5 +3,5 @@
 // NS-EDITABLE-BEGIN tunables
-constexpr int SL = 64;
+constexpr int SL = 128;
 constexpr int KC = 256;
 // NS-EDITABLE-END
 __global__ void k() {}
"""

def apply(d, scope=SC):
    return D.apply_all(D.parse(d, scope), RD, scope)

def test_good_applies():
    out = apply(GOOD)
    assert "SL = 128" in out["src/kvm/moe_fused.cu"] and "SL = 64" not in out["src/kvm/moe_fused.cu"]

def test_insertion_at_region_end_ok():
    d = ("--- a/src/kvm/moe_fused.cu\n+++ b/src/kvm/moe_fused.cu\n@@ -4,3 +4,4 @@\n constexpr int SL = 64;\n constexpr int KC = 256;\n"
         "+constexpr int NEW = 1;\n // NS-EDITABLE-END\n")
    assert "NEW = 1" in apply(d)["src/kvm/moe_fused.cu"]

@pytest.mark.parametrize("bad,why", [
    ("--- a/harness/bench.py\n+++ b/harness/bench.py\n@@ -1 +1 @@\n-x\n+y\n", "outside editable allowlist"),
    ("--- a/tests/test_x.py\n+++ b/tests/test_x.py\n@@ -1 +1 @@\n-x\n+y\n", "outside"),
    ("--- a/../etc/passwd\n+++ b/../etc/passwd\n@@ -1 +1 @@\n-x\n+y\n", "traversal"),
    ("--- a//etc/passwd\n+++ b//etc/passwd\n@@ -1 +1 @@\n-x\n+y\n", "bad path|non-normal"),
    ("--- a/src/opt/policy_tables/../../../x\n+++ b/src/opt/policy_tables/../../../x\n@@ -1 +1 @@\n-x\n+y\n", "traversal|non-normal"),
    ("--- a/src/kvm/moe_fused.cu\n+++ b/src/kvm/other.cu\n@@ -1 +1 @@\n-x\n+y\n", "rename"),
    ("--- a/src/kvm/moe_fused.cu\n+++ /dev/null\n@@ -1 +0,0 @@\n-x\n", "deletion"),
    ("rename from a\nrename to b\n", "unsupported"),
    ("old mode 100644\nnew mode 100755\n", "unsupported"),
    ("GIT binary patch\nliteral 3\n", "unsupported"),
    ("", "empty"),
    ("just prose, no diff\n", "unexpected"),
])
def test_rejects(bad, why):
    with pytest.raises(D.DiffError, match=why):
        apply(bad)

def test_symlink_like_and_abs():
    with pytest.raises(D.DiffError):
        apply("--- /etc/passwd\n+++ /etc/passwd\n@@ -1 +1 @@\n-a\n+b\n")

def test_context_mismatch_no_fuzz():
    with pytest.raises(D.DiffError, match="context mismatch"):
        apply(GOOD.replace("SL = 64", "SL = 65"))

def test_edit_outside_region():
    d = ("--- a/src/kvm/moe_fused.cu\n+++ b/src/kvm/moe_fused.cu\n@@ -1,3 +1,3 @@\n // toy stand-in for src/kvm/moe_fused.cu\n"
         "-constexpr int H = 2048, I = 768;   // protected: model shape\n+constexpr int H = 1, I = 1;\n // NS-EDITABLE-BEGIN tunables\n")
    with pytest.raises(D.DiffError, match="outside editable region"):
        apply(d)

def test_insert_outside_region():
    d = ("--- a/src/kvm/moe_fused.cu\n+++ b/src/kvm/moe_fused.cu\n@@ -1,2 +1,3 @@\n // toy stand-in for src/kvm/moe_fused.cu\n"
         "+int evil;\n constexpr int H = 2048, I = 768;   // protected: model shape\n")
    with pytest.raises(D.DiffError, match="insertion outside"):
        apply(d)

def test_marker_removal_rejected():
    d = GOOD.replace(" // NS-EDITABLE-END\n", "-// NS-EDITABLE-END\n").replace("@@ -3,5 +3,5 @@", "@@ -3,5 +3,4 @@")
    with pytest.raises(D.DiffError):
        apply(d)

def test_marker_injection_to_widen_scope():
    # try to add a new BEGIN marker above protected line to widen the region
    d = ("--- a/src/kvm/moe_fused.cu\n+++ b/src/kvm/moe_fused.cu\n@@ -2,3 +2,4 @@\n constexpr int H = 2048, I = 768;   // protected: model shape\n"
         "+// NS-EDITABLE-BEGIN fake\n // NS-EDITABLE-BEGIN tunables\n constexpr int SL = 64;\n")
    with pytest.raises(D.DiffError):
        apply(d)

def test_file_without_markers_is_uneditable():
    rd = {"src/kvm/moe_fused.cu": "constexpr int SL = 64;\n"}.get
    d = "--- a/src/kvm/moe_fused.cu\n+++ b/src/kvm/moe_fused.cu\n@@ -1 +1 @@\n-constexpr int SL = 64;\n+constexpr int SL = 1;\n"
    with pytest.raises(D.DiffError, match="outside editable region"):
        D.apply_all(D.parse(d, SC), rd, SC)

def test_create_policy():
    with pytest.raises(D.DiffError, match="creation not allowed"):
        apply("--- /dev/null\n+++ b/src/opt/policy_tables/new.json\n@@ -0,0 +1 @@\n+{}\n")
    out = apply("--- /dev/null\n+++ b/src/opt/new_hook.py\n@@ -0,0 +1 @@\n+X=1\n")
    assert out["src/opt/new_hook.py"].startswith("X=1")
    with pytest.raises(D.DiffError):   # create over an existing file
        D.apply_all(D.parse("--- /dev/null\n+++ b/src/opt/policy_tables/sched.json\n@@ -0,0 +1 @@\n+{}\n", SC), RD, SC)

def test_limits():
    big = "--- a/src/opt/policy_tables/sched.json\n+++ b/src/opt/policy_tables/sched.json\n@@ -1 +1,500 @@\n {\n" + "+ \"a\": 1,\n" * 499
    with pytest.raises(D.DiffError):
        D.parse(big, Scope(rules=SC.rules, limits=Limits(max_changed_lines=50, max_diff_bytes=100000)))
    with pytest.raises(D.DiffError, match="too large"):
        D.parse(big, Scope(rules=SC.rules, limits=Limits(max_diff_bytes=100)))

@pytest.mark.parametrize("ch", ["\x00", "‮", "​", "﻿"])
def test_hidden_chars(ch):
    with pytest.raises(D.DiffError, match="control characters"):
        apply(GOOD.replace("= 128;", f"= 128;{ch}"))

def test_count_mismatch():
    with pytest.raises(D.DiffError, match="counts"):
        apply(GOOD.replace("@@ -3,5 +3,5 @@", "@@ -3,5 +3,9 @@"))

def test_duplicate_file_entries():
    with pytest.raises(D.DiffError, match="duplicate"):
        D.parse(GOOD + GOOD, SC)
