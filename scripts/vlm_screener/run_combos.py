"""Stage 2: run the 4 combos (8B/32B x v3/old prompt) over the prepared
windows. Rabbit-absent windows (YOLO gate) are labelled "none" without
touching the VLM; the rest get the compact 4-class one-liner with
first-token-softmax confidence over [Eating, Drinking, Active, Resting].

One model load per combo, released before the next (GPU stays clean).
Resume-safe: per-video docs under results/<combo>/<vid>.json are skipped;
a combo whose docs are all present never loads its model.

Usage (venv python, cwd = project root):
  python -m scripts.vlm_screener.run_combos
  python -m scripts.vlm_screener.run_combos --combos 8b-v3 --limit-videos 2
"""
import argparse
import json
import os
import re
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, ".vlm-screener")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
RESULTS_DIR = os.path.join(DATA_DIR, "results")
PROMPTS_DIR = os.path.join(DATA_DIR, "prompts")

from scripts.vlm_screener.prompts import (
    COMBOS, MAX_NEW_TOKENS, PROMPTS, PROMPT_VERSION,
    SCHEMA_VERSION, MODEL_KEYS, TABLES)


def log(msg):
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        with open(os.path.join(LOG_DIR, "vlm_screener.log"), "a",
                  encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def log_vram(tag):
    try:
        import subprocess
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free,memory.total",
             "--format=csv,noheader"], capture_output=True, text=True,
            timeout=15).stdout.strip().replace("\n", " | ")
        log(f"vram[{tag}] free/total: {out}")
    except Exception as e:  # noqa: BLE001
        log(f"vram[{tag}] query failed: {e}")


def parse_answer(text, table):
    """'<Word> {json...}' -> (label, self_conf). Label resolution: leading
    word first (drives the first-token conf), then JSON field, then any
    allowed word in the text. `table` is prompts.TABLES[species]."""
    labels, words = table["labels"], table["words"]
    label = None
    m = re.match(r"\s*([A-Za-z]+)", text)
    if m and m.group(1).lower() in labels:
        label = m.group(1).lower()
    self_conf = None
    s, e = text.find("{"), text.rfind("}")
    if 0 <= s < e:
        try:
            obj = json.loads(text[s:e + 1])
            if isinstance(obj, dict):
                lab = str(obj.get("label", "")).strip().lower()
                if lab in labels and label is None:
                    label = lab
                c = obj.get("confidence")
                if isinstance(c, (int, float)) and not isinstance(c, bool):
                    self_conf = min(1.0, max(0.0, float(c)))
        except ValueError:
            pass
    if label is None:
        for w in words:
            if re.search(r"\b" + w + r"\b", text, re.IGNORECASE):
                label = w.lower()
                break
    return label, self_conf


def load_model(model_dir, dtype, words):
    import torch
    from transformers import AutoProcessor
    try:
        from transformers import Qwen3VLForConditionalGeneration as ModelCls
    except ImportError:
        from transformers import AutoModelForImageTextToText as ModelCls
    t0 = time.time()
    if dtype == "bf16":
        model = ModelCls.from_pretrained(
            model_dir, torch_dtype=torch.bfloat16,
            attn_implementation="sdpa").to("cuda")
    elif dtype == "bnb4":
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            llm_int8_enable_fp32_cpu_offload=True)
        # no max_memory cap: bnb 4bit refuses CPU-dispatched modules; NF4
        # 32B needs ~18 GiB and must sit fully on the GPU (10-08 lesson)
        model = ModelCls.from_pretrained(
            model_dir, torch_dtype=torch.bfloat16, quantization_config=bnb,
            device_map="auto")
    else:
        raise ValueError(dtype)
    model.eval()
    processor = AutoProcessor.from_pretrained(model_dir)
    tok = getattr(processor, "tokenizer", processor)
    cand_ids = []
    for w in words:
        ids = tok.encode(w, add_special_tokens=False)
        cand_ids.append(ids[0] if ids else None)
    use_logits_conf = (all(c is not None for c in cand_ids)
                       and len(set(cand_ids)) == len(words))
    log(f"model {os.path.basename(model_dir)} ({dtype}) ready in "
        f"{time.time() - t0:.0f}s; words={words} logits_conf={use_logits_conf} "
        f"cand_ids={cand_ids}")
    return model, processor, cand_ids, use_logits_conf


def classify(model, processor, cand_ids, use_logits_conf, pil_frames,
             prompt, tag, table):
    import torch
    from PIL import Image

    def generate(p):
        messages = [{"role": "user", "content":
                    [{"type": "image", "image": im} for im in pil_frames]
                    + [{"type": "text", "text": p}]}]
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt").to(model.device)
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        with torch.inference_mode():
            gen = model.generate(
                **inputs, do_sample=False, max_new_tokens=MAX_NEW_TOKENS,
                return_dict_in_generate=True, output_scores=True)
        text = processor.tokenizer.decode(
            gen.sequences[0][gen.sequences.shape[1] - len(gen.scores):],
            skip_special_tokens=True).strip()
        probs = None
        if use_logits_conf:
            logits = gen.scores[0][0].float()
            cand = torch.tensor(cand_ids, device=logits.device)
            plist = torch.softmax(logits[cand], dim=-1).cpu().tolist()
            probs = {lab: round(v, 4) for lab, v in zip(table["labels"], plist)}
        return text, probs

    last_err = None
    for attempt, p in enumerate([prompt, prompt + table["retry_suffix"]]):
        try:
            text, probs = generate(p)
            label, self_conf = parse_answer(text, table)
            if label is None:
                last_err = f"unparsed: {text[:60]!r}"
                continue
            if probs is not None and label in probs:
                conf = probs[label]
            elif self_conf is not None:
                conf = self_conf
            else:
                conf = 0.5
            return label, round(float(conf), 4), probs, bool(attempt), False
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                last_err = f"OOM attempt {attempt}"
                continue
            raise
    log(f"  [WARN] {tag}: {last_err}")
    return "resting", 0.0, None, False, True


def merge_segments(raw):
    segments = []
    for start, end, label, conf in raw:
        if (segments and segments[-1]["label"] == label
                and abs(segments[-1]["end"] - start) < 1e-6):
            prev = segments[-1]
            w0, w1 = prev["_w"], end - start
            prev["confidence"] = (prev["confidence"] * w0 + conf * w1) / (w0 + w1)
            prev["_w"] = w0 + w1
            prev["end"] = end
        else:
            segments.append({"start": start, "end": end, "label": label,
                             "confidence": round(float(conf), 4), "_w": end - start})
    for s in segments:
        s.pop("_w", None)
        s["confidence"] = round(s["confidence"], 4)
    return segments


def run_combo(cfg, manifest, windows, gates):
    import numpy as np
    from PIL import Image
    import torch

    combo = cfg["combo"]
    out_dir = os.path.join(RESULTS_DIR, combo)
    os.makedirs(out_dir, exist_ok=True)
    model_dir = os.path.join(PROJECT_ROOT, "models-vlm",
                             MODEL_KEYS[cfg["model_key"]])
    pkey = cfg["prompt_key"]
    prompt = PROMPTS[pkey]
    pver = PROMPT_VERSION[pkey]
    with open(os.path.join(PROMPTS_DIR, f"{combo}.txt"), "w",
              encoding="utf-8") as f:
        f.write(prompt)

    win_by_vid = {v["vid"]: v for v in windows["videos"]}
    vid_order = [v["vid"] for v in manifest["videos"]]
    path_by_vid = {v["vid"]: v["path"] for v in manifest["videos"]}

    todo = []
    for vid in vid_order:
        if not os.path.isfile(os.path.join(out_dir, f"{vid}.json")):
            todo.append(vid)
    log(f"=== combo {combo}: {len(todo)}/{len(vid_order)} videos to label "
        f"(model={os.path.basename(model_dir)}, dtype={cfg['dtype']}, "
        f"prompt={pver}) ===")
    if not todo:
        log(f"=== combo {combo}: all docs present, skip model load ===")
        return 0
    log_vram(combo)
    table = TABLES["rabbit"]
    model, processor, cand_ids, use_logits_conf = load_model(
        model_dir, cfg["dtype"], table["words"])

    n_win, n_vlm, n_err = 0, 0, 0
    t_all = time.time()
    for i, vid in enumerate(todo, 1):
        t0 = time.time()
        wv = win_by_vid[vid]
        vdir = os.path.join(DATA_DIR, "frames", vid)
        gvid = gates["windows"].get(vid, {})
        records, raw = [], []
        dist = {l: 0 for l in table["labels"]}
        dist["none"] = 0
        wins = wv["windows"]
        for idx, win in enumerate(wins):
            g = gvid.get(win["w"], {})
            det = int(g.get("det_frames", 0))
            has_pet = bool(g.get("has_pet", False))
            end = (wv["duration"] if idx == len(wins) - 1
                   else wins[idx + 1]["start"])
            ran_vlm = False
            if not has_pet:
                label, conf, probs, perr = "none", 1.0, None, False
            else:
                frames = []
                for k in range(8):
                    fp = os.path.join(vdir, f"{win['w']}_f{k}.jpg")
                    if os.path.isfile(fp):
                        frames.append(Image.open(fp).convert("RGB"))
                if not frames:
                    label, conf, probs, perr = "none", 1.0, None, False
                else:
                    label, conf, probs, retried, perr = classify(
                        model, processor, cand_ids, use_logits_conf,
                        frames, prompt, f"{vid}/{win['w']}", table)
                    n_vlm += 1
                    ran_vlm = True
            raw.append((win["start"], end, label, conf))
            records.append({"w": win["w"], "start": win["start"], "end": end,
                            "gate": "vlm" if ran_vlm else "yolo",
                            "det_frames": det, "label": label,
                            "confidence": conf, "probs": probs,
                            "parse_error": bool(perr)})
            dist[label] = dist.get(label, 0) + 1
            n_win += 1
            n_err += int(bool(perr))
        segments = merge_segments(raw)
        total_w = max(1, sum(dist.values()))
        dom_label = max(dist.items(), key=lambda kv: kv[1])[0]
        doc = {
            "schema_version": SCHEMA_VERSION,
            "vid": vid, "video_path": path_by_vid[vid],
            "duration": wv["duration"], "combo": combo,
            "model_id": model_dir, "dtype": cfg["dtype"],
            "prompt_version": pver,
            "windows": records, "segments": segments,
            "dist": {k: v for k, v in dist.items()},
            "dominant": dom_label if dist[dom_label] else "none",
            "dominant_share": round(dist[dom_label] / total_w, 4),
        }
        tmp = os.path.join(out_dir, f"{vid}.json.tmp{os.getpid()}")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)
        os.replace(tmp, os.path.join(out_dir, f"{vid}.json"))
        doms = " ".join(f"{r['label'][:3]}:{r['confidence']:.2f}"
                        for r in records[:6])
        log(f"  ({i}/{len(todo)}) {vid} wins={len(records)} "
            f"dom={doc['dominant']}({doc['dominant_share']:.0%}) "
            f"[{doms}] {time.time() - t0:.1f}s")
        if i % 10 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    manifest_c = {
        "combo": combo, "model_id": model_dir, "dtype": cfg["dtype"],
        "prompt_version": pver, "prompt_text": prompt,
        "n_videos": len(todo), "n_windows": n_win, "n_vlm_windows": n_vlm,
        "parse_errors": n_err,
        "elapsed_sec": round(time.time() - t_all, 1),
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(out_dir, "manifest.json"), "w",
              encoding="utf-8") as f:
        json.dump(manifest_c, f, ensure_ascii=False, indent=2)
    log(f"=== combo {combo} DONE: {len(todo)} videos, {n_win} windows "
        f"({n_vlm} via VLM), parse_errors={n_err}, "
        f"{time.time() - t_all:.0f}s ===")
    try:
        del model, processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()
        log(f"combo {combo}: model released")
    except Exception:  # noqa: BLE001
        pass
    return 0


def main(argv=None):
    global DATA_DIR, RESULTS_DIR, PROMPTS_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--combos", default="8b-v3,8b-old,32b-v3,32b-old",
                    help="comma list from: 8b-v3,8b-old,32b-v3,32b-old")
    ap.add_argument("--limit-videos", type=int, default=0)
    ap.add_argument("--data-dir", default=DATA_DIR,
                    help="override data dir (smoke tests)")
    a = ap.parse_args(argv)
    if a.data_dir != DATA_DIR:
        DATA_DIR = a.data_dir
        RESULTS_DIR = os.path.join(DATA_DIR, "results")
        PROMPTS_DIR = os.path.join(DATA_DIR, "prompts")
    wanted = [c.strip() for c in a.combos.split(",") if c.strip()]
    cfgs = [c for c in COMBOS if c["combo"] in wanted]
    if not cfgs:
        log(f"[ERROR] no valid combo in {wanted}")
        return 1
    with open(os.path.join(DATA_DIR, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    with open(os.path.join(DATA_DIR, "windows.json"), encoding="utf-8") as f:
        windows = json.load(f)
    with open(os.path.join(DATA_DIR, "gates.json"), encoding="utf-8") as f:
        gates = json.load(f)
    os.makedirs(PROMPTS_DIR, exist_ok=True)
    if a.limit_videos:
        keep = {v["vid"] for v in manifest["videos"][:a.limit_videos]}
        manifest = dict(manifest)
        manifest["videos"] = [v for v in manifest["videos"] if v["vid"] in keep]
        windows = dict(windows)
        windows["videos"] = [v for v in windows["videos"] if v["vid"] in keep]
        log(f"limit-videos={a.limit_videos}: only {len(keep)} vids")
    for cfg in cfgs:
        rc = run_combo(cfg, manifest, windows, gates)
        if rc:
            return rc
    log("run_combos: ALL DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
