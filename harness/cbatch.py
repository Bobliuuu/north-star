"""Continuous (iteration-level) batching for exact North on one H100 (branch feat/cbatch).

Requests join and leave between decode steps. KV lives in a fixed pool of `slots` per-sequence caches. One CUDA graph is captured per
batch bucket B in `buckets` (batched exact executor, exact v2: streams + tuned tiles + mode-6 MoE, batched attention for every B). The
batched attention addresses K/V through a device table of data pointers [layer, K/V, B]; each step rewrites that table IN PLACE so the
bucket's graph runs on whichever slots are active (rows beyond the active count point at a scratch slot and are masked as invalid, so
they cost no expert work). No re-capture, no KV copies.

Scheduling (simple, explicit): every iteration first admits arrived requests into free slots (FIFO) and prefills them (blocking; one
batched forward for all of them by default, see `prefill`; no chunked prefill mixed into decode steps), then runs ONE plain decode step for all active slots in the smallest bucket that fits.
Exactness: rows are batch-invariant (H20/H24/H26), so a request's tokens do not depend on which requests share its steps; bench_cbatch.py
checks every request against the same request decoded alone. No speculation here (plain decode; H26 shows PLD loses from B=2)."""
import time
import torch

import batch_exact
import fast_kernels_batch as fkb
import fast_kernels_exact as fx
import fast_kernels_x2 as f2
from northstar.refmodel import RefNorth
from fast_north import FastNorth


class Request:
    __slots__ = ("rid", "ids", "max_new", "arrival", "slot", "pos", "cur", "out", "t_admit", "t_first", "t_done", "prefill_s")

    def __init__(self, rid, ids, max_new, arrival=0.0):
        self.rid, self.ids, self.max_new, self.arrival = rid, list(ids), max_new, arrival
        self.slot = self.pos = self.cur = None
        self.out, self.t_admit, self.t_first, self.t_done, self.prefill_s = [], None, None, None, 0.0


class ContinuousEngine:
    def __init__(self, model, Lmax, slots=32, buckets=(1, 2, 4, 8, 16, 32), prefill="batched", prefill_tokens=8192):
        """prefill: 'single' = one chunked-SDPA forward per request (LCNorth; a request's KV never depends on other requests);
        'graph' = one request at a time like 'single', but through a CUDA graph per padded length bucket (128..pow2 >= Lmax) that
        prefills into a staging KV, then copies the request's positions into its slot: no per-request launch overhead, and the padding
        depends only on the request's own length, so its KV still never depends on other requests;
        'batched' = every request admitted in an iteration is prefilled in ONE forward (concatenated tokens, block-diagonal causal mask,
        up to prefill_tokens per forward): far fewer launches and one expert-weight pass, but the batched expert GEMM may round a
        request's activations differently depending on its co-prefilled requests (measured in bench_cbatch.py)."""
        assert max(buckets) >= slots and prefill in ("single", "batched", "graph")
        self.Lmax, self.slots, self.prefill_mode, self.prefill_tokens = Lmax, slots, prefill, prefill_tokens
        self.buckets = tuple(b for b in buckets if b <= slots) or (slots,)
        eos = model.config.eos_token_id
        self.eos = eos if isinstance(eos, int) else eos[0]
        fkb.TILE_GRID = 6
        batch_exact.STREAMS = True
        batch_exact.BATCH_ATTN = True          # pointer-table attention for every bucket, incl. B=1
        fx.LIN_CFG.update(f2.TUNED_LIN_CFG)
        self.ref = RefNorth(model)
        self.X = FastNorth(self.ref, Lmax, exact=True)
        self.X.set_x2(**f2.recommended(Lmax))
        self.BX = batch_exact.BatchExact(self.X)
        from longctx_north import LCNorth
        self.ln = LCNorth(self.ref, Lmax)
        dev = self.ref.device
        self.kvs = [self.X.new_kv() for _ in range(slots + 1)]            # slot `slots` = scratch for padding rows
        nl = len(self.kvs[0])
        self.colp = [torch.tensor([[kv[i][w].data_ptr() for w in (0, 1)] for i in range(nl)], dtype=torch.int64, device=dev) for kv in self.kvs]
        self.steps, self.tables = {}, {}
        for B in self.buckets:
            kv_list = self.kvs[:B]
            self.steps[B] = self.BX.make_step(B, 1, kv_list)
            self.tables[B] = self.BX.ptrs(kv_list)                         # the [nl, 2, B] tensor this bucket's graph reads
        torch.cuda.synchronize()

    def _bucket(self, n):
        return next(b for b in self.buckets if b >= n)

    @torch.no_grad()
    def _prefill(self, rs):
        """Prefill newly admitted requests rs (slots already assigned)."""
        t = time.perf_counter()
        if self.prefill_mode == "single":
            for r in rs:
                self.ln.prefill(r.ids[:-1], self.kvs[r.slot], chunk=1024)
        elif self.prefill_mode == "graph":
            for r in rs:
                self._prefill_graph(r)
        else:
            grp, n = [], 0
            for r in rs:
                L = len(r.ids) - 1
                if grp and n + L > self.prefill_tokens:
                    self._prefill_batched(grp); grp, n = [], 0
                grp.append(r); n += L
            if grp:
                self._prefill_batched(grp)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t
        for r in rs:
            r.prefill_s = dt / len(rs)
            r.pos, r.cur = len(r.ids) - 1, r.ids[-1]

    # ------------------------------------------------------------------ graphed single-request prefill
    def _pf_forward(self, ids, kv, Tb):
        """Static-shape prefill of Tb tokens at positions 0..Tb-1 into kv (same math as LCNorth.prefill, one chunk)."""
        import torch.nn.functional as F
        from transformers.models.cohere2_moe.modeling_cohere2_moe import apply_rotary_pos_emb
        mdl, cfg = self.ref.model, self.ref.model.config
        D, H, Hkv = cfg.head_dim, cfg.num_attention_heads, cfg.num_key_value_heads
        G, W = H // Hkv, cfg.sliding_window
        pos = self._pf_pos[Tb]
        x = mdl.embed_tokens(ids)
        cos, sin = mdl.rotary_emb(x, pos[None])
        mfull, msl = self._pf_mask[Tb]
        for i, layer in enumerate(mdl.layers):
            h = layer.input_layernorm(x)
            at = layer.self_attn
            sliding = cfg.layer_types[i] == "sliding_attention"
            q = at.q_proj(h).view(1, Tb, H, D).transpose(1, 2)
            k = at.k_proj(h).view(1, Tb, Hkv, D).transpose(1, 2)
            v = at.v_proj(h).view(1, Tb, Hkv, D).transpose(1, 2)
            if sliding or at.force_rope:
                q, k = apply_rotary_pos_emb(q, k, cos, sin)
            kc, vc = kv[i]
            kc[:, :, :Tb].copy_(k); vc[:, :, :Tb].copy_(v)
            qg = q.reshape(1, Hkv, G * Tb, D)
            o = F.scaled_dot_product_attention(qg, k, v, attn_mask=msl if sliding else mfull)
            o = o.reshape(1, Hkv, G, Tb, D).reshape(1, H, Tb, D).transpose(1, 2).reshape(1, Tb, H * D)
            a = at.o_proj(o)
            mlp = layer.mlp
            if not hasattr(mlp, "gate"):
                m = mlp(h)
            else:
                hf = h.reshape(-1, h.shape[-1])
                _, sc_, ix_ = mlp.gate(hf)
                m = mlp.experts(hf, ix_, sc_).reshape(1, Tb, -1)
            x = x + a + m

    def _pf_graph(self, Tb):
        if not hasattr(self, "_pf"):
            self._pf, self._pf_pos, self._pf_mask = {}, {}, {}
            self._pf_kv = self.X.new_kv()          # staging KV shared by every bucket (graphs replay one at a time)
        if Tb not in self._pf:
            dev, cfg = self.ref.device, self.ref.model.config
            G, W = cfg.num_attention_heads // cfg.num_key_value_heads, cfg.sliding_window
            ix = torch.arange(Tb, device=dev)
            mf = ix[None, :] <= ix[:, None]
            ms = mf & (ix[:, None] - ix[None, :] < W)
            self._pf_pos[Tb] = ix
            self._pf_mask[Tb] = (mf.repeat(G, 1), ms.repeat(G, 1))
            ids = torch.zeros(1, Tb, dtype=torch.long, device=dev)
            s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s), torch.no_grad():
                for _ in range(2):
                    self._pf_forward(ids, self._pf_kv, Tb)
            torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g), torch.no_grad():
                self._pf_forward(ids, self._pf_kv, Tb)
            self._pf[Tb] = (g, ids)
        return self._pf[Tb]

    def _prefill_graph(self, r):
        L = len(r.ids) - 1
        Tb = 128
        while Tb < L:
            Tb *= 2
        Tb = min(Tb, self.Lmax)                    # top bucket = KV capacity (staging KV has Lmax positions)
        g, ids = self._pf_graph(Tb)
        pad = r.ids[:-1] + [0] * (Tb - L)          # pad AFTER the prompt: causal rows never see it; depends on this request only
        ids.copy_(torch.tensor([pad], device=ids.device))
        g.replay()
        kv = self.kvs[r.slot]
        for i in range(len(kv)):
            for w in (0, 1):
                kv[i][w][:, :, :L].copy_(self._pf_kv[i][w][:, :, :L])

    def _prefill_batched(self, rs):
        import torch.nn.functional as F
        from transformers.models.cohere2_moe.modeling_cohere2_moe import apply_rotary_pos_emb
        mdl, cfg, dev = self.ref.model, self.ref.model.config, self.ref.device
        D, H, Hkv = cfg.head_dim, cfg.num_attention_heads, cfg.num_key_value_heads
        G, W = H // Hkv, cfg.sliding_window
        seqs = [r.ids[:-1] for r in rs]
        lens = [len(q) for q in seqs]
        N = sum(lens)
        ids = torch.tensor([[t for q in seqs for t in q]], device=dev)
        pos = torch.cat([torch.arange(L, device=dev) for L in lens])
        seg = torch.repeat_interleave(torch.arange(len(lens), device=dev), torch.tensor(lens, device=dev))
        ix = torch.arange(N, device=dev)
        mfull = (seg[:, None] == seg[None, :]) & (ix[None, :] <= ix[:, None])     # block-diagonal causal (sequences are contiguous)
        msl = mfull & (ix[:, None] - ix[None, :] < W)
        offs = [0]
        for L in lens:
            offs.append(offs[-1] + L)
        x = mdl.embed_tokens(ids)
        cos, sin = mdl.rotary_emb(x, pos[None])
        for i, layer in enumerate(mdl.layers):
            h = layer.input_layernorm(x)
            at = layer.self_attn
            sliding = cfg.layer_types[i] == "sliding_attention"
            q = at.q_proj(h).view(1, N, H, D).transpose(1, 2)
            k = at.k_proj(h).view(1, N, Hkv, D).transpose(1, 2)
            v = at.v_proj(h).view(1, N, Hkv, D).transpose(1, 2)
            if sliding or at.force_rope:
                q, k = apply_rotary_pos_emb(q, k, cos, sin)
            for r, a0, L in zip(rs, offs, lens):
                kc, vc = self.kvs[r.slot][i]
                kc[:, :, :L].copy_(k[:, :, a0:a0 + L]); vc[:, :, :L].copy_(v[:, :, a0:a0 + L])
            kr, vr = k.repeat_interleave(G, dim=1), v.repeat_interleave(G, dim=1)
            o = F.scaled_dot_product_attention(q, kr, vr, attn_mask=msl if sliding else mfull)
            a = at.o_proj(o.transpose(1, 2).reshape(1, N, H * D))
            mlp = layer.mlp
            if not hasattr(mlp, "gate"):
                m = mlp(h)
            else:
                hf = h.reshape(-1, h.shape[-1])
                _, sc_, ix_ = mlp.gate(hf)
                m = mlp.experts(hf, ix_, sc_).reshape(1, N, -1)
            x = x + a + m

    @torch.no_grad()
    def _step(self, act):
        n = len(act)
        B = self._bucket(n)
        st, tb = self.steps[B], self.tables[B]
        sl = [r.slot for r in act] + [self.slots] * (B - n)
        tb.copy_(torch.stack([self.colp[s] for s in sl], dim=-1))
        st.set([[r.cur] for r in act] + [[act[0].cur]] * (B - n), [r.pos for r in act] + [0] * (B - n),
               [[True]] * n + [[False]] * (B - n))
        return [p[0] for p in st.run().tolist()[:n]], B

    def run(self, requests, clock=time.perf_counter):
        """Serve `requests` (Request objects with arrival offsets in seconds, relative to the start of this call). Returns stats.
        Times on each Request: t_admit, t_first (first generated token), t_done, all relative to the start."""
        for r in requests:
            assert len(r.ids) + r.max_new + 16 < self.Lmax, (r.rid, len(r.ids))
        pend = sorted(requests, key=lambda r: r.arrival)
        free = list(range(self.slots))
        act = []
        steps, rows, bsum = 0, 0, 0
        t0 = clock()
        while pend or act:
            now = clock() - t0
            new = []
            while pend and pend[0].arrival <= now and free:
                r = pend.pop(0); r.slot = free.pop(0); r.t_admit = clock() - t0; new.append(r)
            if new:
                self._prefill(new); act.extend(new)
            if not act:
                if pend:
                    time.sleep(max(0.0, min(0.002, pend[0].arrival - (clock() - t0))))
                continue
            pred, B = self._step(act)
            tnow = clock() - t0
            steps += 1; rows += len(act); bsum += B
            keep = []
            for r, tk in zip(act, pred):
                r.out.append(tk); r.cur = tk; r.pos += 1
                if r.t_first is None:
                    r.t_first = tnow
                if tk == self.eos or len(r.out) >= r.max_new:
                    r.t_done = tnow; free.append(r.slot); free.sort()
                else:
                    keep.append(r)
            act = keep
        wall = clock() - t0
        return dict(wall=wall, steps=steps, mean_active=rows / max(1, steps), mean_bucket=bsum / max(1, steps),
                    tokens=sum(len(r.out) for r in requests))
