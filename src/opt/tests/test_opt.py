import glob, json, os, pathlib, sys, tempfile, unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from opt import config as C, costmodel as CM, dag as D, sched as S, solve as SV, synth
from opt.calibration import AcceptTable, CalibrationError, parse_row
from opt.policy import PolicyCache, quantize_ctx
from opt.perturb import perturbation_study

ROOT = pathlib.Path(__file__).resolve().parents[3]
FILES = sorted(glob.glob(str(ROOT / "results" / "*.jsonl")))


def tiny_rows():
    rows = []
    for tv in range(1, 12):
        for td in (1, 2):
            for r in range(1, 9):
                rows.append({"gpu": "T", "routing": "corr75", "r": r, "Tv": tv, "Td": td, "peak_read_GBs": 1600.0,
                             "v": {"verify_only": {"us_med": 100 + 30 * tv, "GBs": 1000},
                                   "two_pass": {"us_med": 100 + 30 * tv + 80 + r, "GBs": 900},
                                   "fused": {"us_med": 100 + 30 * tv + 5 + r, "GBs": 1100}}})
    return rows


def tiny_cm():
    return CM.NorthCostModel(CM.MicrobenchTable(tiny_rows()))


def synth_cal(**kw):
    return AcceptTable(synth.synthetic_entries(i_understand_this_is_synthetic=True, gmax=6, **kw))


class Calibration(unittest.TestCase):
    def test_roundtrip_and_expectation(self):
        e = synth.synthetic_entries(i_understand_this_is_synthetic=True)[0]
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "c.jsonl"); synth.write_jsonl([e], p)
            t = AcceptTable.from_jsonl(p)
        self.assertTrue(t.any_synthetic())
        x = t.get("code", e.key)
        self.assertAlmostEqual(x.expected_committed(0), 1.0)
        self.assertGreater(x.expected_committed(4), x.expected_committed(2))
        with self.assertRaises(CalibrationError):
            x.expected_committed(x.gmax + 1)

    def test_validation(self):
        base = {"workload": "w", "attn_budget": 0, "r": 4, "accept_len_pmf": [0.5, 0.5], "provenance": "measured"}
        parse_row(base)
        for bad in ({"accept_len_pmf": [0.5, 0.6]}, {"provenance": None}, {"skip_layers": [0]},
                    {"carry_hit_given_full": 2}):
            d = dict(base); d.update(bad)
            if d["provenance"] is None:
                del d["provenance"]
            with self.assertRaises(CalibrationError):
                parse_row(d)

    def test_synth_requires_flag(self):
        with self.assertRaises(RuntimeError):
            synth.synthetic_entries()

    def test_truncation_exact(self):
        pmf = (0.2, 0.3, 0.1, 0.4)
        e = parse_row({"workload": "w", "attn_budget": 0, "r": 1, "accept_len_pmf": list(pmf), "provenance": "measured"})
        for g in range(4):
            self.assertAlmostEqual(e.expected_committed(g), 1 + sum(min(a, g) * p for a, p in enumerate(pmf)))


class Cost(unittest.TestCase):
    def test_real_table_loads_and_provenance(self):
        t = CM.MicrobenchTable.from_jsonl(*FILES)
        self.assertEqual(t.max_tokens, 16)
        self.assertLess(t.fused_us("corr75", 4, 5, 1), t.two_pass_us("corr75", 4, 5, 1))
        cm = CM.NorthCostModel(t)
        v = cm.moe_group(10, 5, 4, "fused", "verify")[0]
        self.assertEqual(set(v.src()), {"measured"})
        self.assertIn("derived", cm.moe_group(10, 1, 4, "two_pass", "draft")[0].src())
        self.assertIn("analytic", cm.lm_head(1)[0].src())

    def test_out_of_grid(self):
        with self.assertRaises(ValueError):
            tiny_cm().table.verify_us("corr75", 99)

    def test_sparse_attention_cheaper_at_long_ctx_only_on_global(self):
        cm = tiny_cm()
        full = cm.attn_group([4, 5], 1, 131072, 0, True)[0].us
        sp = cm.attn_group([4, 5], 1, 131072, 1024, True)[0].us
        self.assertLess(sp, full)
        self.assertAlmostEqual(cm.attn_group([5], 1, 131072, 1024, True)[0].us, cm.attn_group([5], 1, 131072, 0, True)[0].us)


class Schedule(unittest.TestCase):
    def setUp(self):
        self.cm = tiny_cm(); self.b = C.Bucket(ctx=8192)

    def dag(self, **kw):
        cfg = C.Config(**{"gamma": 3, "attn_budget": 0, "r": 4, "skip": (), "mode": "two_pass", **kw})
        return D.build_cycle(self.cm, self.b, cfg, "B", 4)

    def test_precedence_and_bounds(self):
        d = self.dag(attn_budget=512)
        lb = S.lower_bound_us(d); ser = S.serial(d); opt = S.cpsat(d, time_limit_s=5)
        self.assertLessEqual(lb, opt.makespan_us + 1e-6)
        self.assertLessEqual(opt.makespan_us, ser.makespan_us + 1e-6)
        ids = d.by_id()
        for t in d.tasks:
            v = next(x for x in t.variants if x.name == opt.variant[t.id])
            for dep in t.deps:
                dv = next(x for x in ids[dep].variants if x.name == opt.variant[dep])
                self.assertGreaterEqual(opt.start_us[t.id] + 1e-6, opt.start_us[dep] + dv.us - 0.11)

    def test_resource_cumulative_respected(self):
        d = self.dag(); opt = S.cpsat(d, time_limit_s=5)
        ev = []
        for t in d.tasks:
            v = next(x for x in t.variants if x.name == opt.variant[t.id])
            ev.append((opt.start_us[t.id], v.us, v.bw, v.sm))
        pts = sorted({e[0] for e in ev})
        for p in pts:
            self.assertLessEqual(sum(e[2] for e in ev if e[0] <= p + 1e-9 < e[0] + e[1] - 0.2), 100)
            self.assertLessEqual(sum(e[3] for e in ev if e[0] <= p + 1e-9 < e[0] + e[1] - 0.2), 100)

    def test_parallel_branches_overlap(self):
        d = self.dag(); self.assertLess(S.cpsat(d, time_limit_s=5).makespan_us, S.serial(d).makespan_us * 0.95)

    def test_memory_cap_infeasible(self):
        d = self.dag()
        self.assertEqual(S.cpsat(d, mem_cap_kib=1, time_limit_s=5).status, "INFEASIBLE")
        self.assertFalse(S.serial(d, mem_cap_kib=1).feasible)
        self.assertTrue(S.cpsat(d, mem_cap_kib=10**7, time_limit_s=5).feasible)

    def test_skip_shortens_draft_not_verify(self):
        a = self.dag(); b = self.dag(skip=tuple(range(8, 40)))
        va = [t.variants[0].us for t in a.tasks if t.phase == "verify"]
        vb = [t.variants[0].us for t in b.tasks if t.phase == "verify"]
        self.assertEqual(va, vb)
        self.assertLess(sum(t.variants[0].us for t in b.tasks if t.phase == "draft"),
                        sum(t.variants[0].us for t in a.tasks if t.phase == "draft"))

    def test_scenarios(self):
        cfg = C.Config(3, 0, 4, (), "fused")
        sc = D.scenarios(self.cm, self.b, cfg, 0.3, 4)
        self.assertEqual([round(p, 2) for p, _ in sc], [0.3, 0.7])
        self.assertEqual(sc[0][1].meta["n_draft_steps"], 2)
        self.assertEqual(len(D.scenarios(self.cm, self.b, C.Config(3, 0, 4, (), "two_pass"), 0.3, 4)), 1)


class Solver(unittest.TestCase):
    def setUp(self):
        self.cm = tiny_cm(); self.cal = synth_cal()
        self.ev = SV.Evaluator(self.cm, C.Bucket(ctx=8192), self.cal, n_groups=3, time_limit_s=5)
        self.space = SV.Space(gammas=(1, 2, 4))

    def test_dinkelbach_matches_bruteforce(self):
        cands = SV.enumerate_configs(self.ev, self.space)[:40]
        best, lam, st = SV.dinkelbach(self.ev, cands)
        brute = max(cands, key=lambda c: self.ev.N(c) / self.ev.optimise(c)[0])
        self.assertAlmostEqual(self.ev.N(best) / self.ev.optimise(best)[0], self.ev.N(brute) / self.ev.optimise(brute)[0], places=9)
        self.assertLessEqual(st["exact_solves_configs"], len(cands))

    def test_comparator_ordering(self):
        res = SV.run_comparators(self.ev, self.space, C.Config(4, 0, 4, (), "two_pass"))
        t = {k: v.tok_s for k, v in res.items()}
        eps = 1e-6
        self.assertGreaterEqual(t["draft_only"] + eps, t["fixed_baseline"])
        self.assertGreaterEqual(t["schedule_only"] + eps, t["fixed_baseline"])
        self.assertGreaterEqual(t["independent"] + eps, t["draft_only"])
        self.assertGreaterEqual(t["joint"] + eps, t["independent"])
        self.assertGreaterEqual(t["joint"] + eps, t["schedule_only"])

    def test_latency_cap_filters(self):
        ev = SV.Evaluator(self.cm, C.Bucket(ctx=8192), self.cal, n_groups=3, lat_cap_us=1.0, time_limit_s=5)
        cands = SV.enumerate_configs(ev, self.space)[:5]
        best, _, _ = SV.dinkelbach(ev, cands)
        self.assertIsNone(best)

    def test_memory_filters_configs(self):
        ev = SV.Evaluator(self.cm, C.Bucket(ctx=131072), self.cal, n_groups=3, mem_cap_kib=1000)
        sparse = [c for c in SV.enumerate_configs(ev, self.space) if c.attn_budget]
        self.assertEqual(sparse, [])    # sparse index scratch (SYNTHETIC size) exceeds the cap


class PolicyAndPerturb(unittest.TestCase):
    def test_policy_cache_roundtrip(self):
        cm = tiny_cm(); cal = synth_cal()
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "p.json"); pc = PolicyCache(p)
            mk = lambda b: SV.Evaluator(cm, b, cal, n_groups=3, time_limit_s=5)
            pol = pc.solve(C.Bucket(ctx=5000), mk, SV.Space(gammas=(1, 2)))
            self.assertTrue(pol["acceptance_synthetic"])
            self.assertEqual(quantize_ctx(5000), 8192)
            pc2 = PolicyCache(p)
            self.assertEqual(pc2.lookup(C.Bucket(ctx=7000))["config"], pol["config"])

    def test_perturbation_runs(self):
        r = perturbation_study(tiny_cm(), C.Bucket(ctx=8192), synth_cal(), SV.Space(gammas=(1, 2)), draws=3,
                               shortlist=6, n_groups=3, time_limit_s=5)
        self.assertGreaterEqual(r["regret_mean"], -1e-6)
        self.assertEqual(len(r["rows"]), 3)


if __name__ == "__main__":
    unittest.main()
