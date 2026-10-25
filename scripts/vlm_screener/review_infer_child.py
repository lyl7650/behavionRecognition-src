"""One model per child, covering selected prompts in a 50-video batch.

Exiting the child releases all CUDA memory before the next model is loaded.
"""
import argparse
import sys


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--job", required=True)
    ap.add_argument("--combos", nargs="+", required=True)
    ap.add_argument("--vids", nargs="+", required=True)
    ap.add_argument("--crop", action="store_true")
    a = ap.parse_args(argv)
    from scripts.vlm_screener import review_infer
    from scripts.vlm_screener.prompts import COMBOS_BY_SPECIES
    from scripts.vlm_screener.review_store import require_context
    species = require_context(a.target)["species"]
    cfgs = [c for c in COMBOS_BY_SPECIES[species] if c["combo"] in a.combos]
    if len(cfgs) != len(a.combos):
        print(f"[ERROR] unknown or duplicate combos: {a.combos}", flush=True)
        return 1
    try:
        return review_infer.run_model_for_batch(
            a.target, cfgs, a.crop, a.vids, a.job)
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        try:
            review_infer._write_progress(a.target, {
                "job": a.job, "done": 0, "total": 0,
                "current": None, "error": str(e)})
        except Exception:  # noqa: BLE001
            pass
        return 1


if __name__ == "__main__":
    sys.exit(main())
