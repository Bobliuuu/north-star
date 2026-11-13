"""Standalone runner for bench_x2's real-traffic phase (one process; used inside the vLLM image so vLLM and exact v2 share a host).
Usage: python run_x2_traffic.py <model_dir> <out.jsonl> [limit]"""
import json, pathlib, sys, time
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))


def main():
    model_dir, out, limit = sys.argv[1], pathlib.Path(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 0
    import torch, transformers, triton
    from loader import load_model
    import bench_x2, fast_kernels_x2 as f2
    t0 = time.time()
    try:
        m, tok = load_model(model_dir, "grouped_mm")
    except TypeError:
        m, tok = load_model(model_dir, None)
    rec = f2.recommended()
    j = bench_x2.run(m, tok, ["traffic"], rec["lin_cfg"], rec["moe_cfg"], limit, cand=rec)
    j.lines.insert(0, json.dumps(dict(phase="meta", torch=torch.__version__, transformers=transformers.__version__, triton=triton.__version__,
                                      gpu=torch.cuda.get_device_name(), load_s=time.time() - t0, limit=limit)))
    out.write_text(j.text())
    print("X2_TRAFFIC_DONE", time.time() - t0, flush=True)


if __name__ == "__main__":
    main()
