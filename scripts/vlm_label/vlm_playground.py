"""Ad-hoc playground engine behind the VLM review server (:8767, tab 2).

Single-slot model holder: lazily loads Qwen3-VL (8B BF16 / 32B NF4) inside
the server process on first use; loading the other tag unloads the current
one (both cannot fit 24 GB together). Reuses the windowing / parsing /
merging helpers from scripts.vlm_label.vlm_label so ad-hoc results are
directly comparable with the batch docs.

Guards:
- process-wide lock: one inference / model switch at a time;
- VRAM pre-check: if free VRAM is below the model's need (e.g. the
  overnight labeling subprocess owns the GPU), refuse with a readable
  error instead of OOM-ing the batch run.

Results are returned to the browser and NOT persisted (same policy as
xclip_lab's playground tab).
"""
import gc
import os
import sys
import threading
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

MODELS = {
    "8b": {
        "desc": "Qwen3-VL-8B BF16",
        "path": os.path.join(PROJECT_ROOT, "models-vlm", "qwen3-vl-8b"),
        "dtype": "bf16", "gpu_mem_gib": 0, "need_gib": 16.0,
    },
    "32b": {
        "desc": "Qwen3-VL-32B NF4",
        "path": os.path.join(PROJECT_ROOT, "models-vlm", "qwen3-vl-32b"),
        # NO max_memory cap: bnb 4bit refuses CPU-dispatched modules, and a
        # 17GiB cap pushed leftovers to CPU -> ValueError (fresh proc) or
        # silent meta weights (server proc). Full-GPU placement (~18 GiB)
        # is the only working mode; need_gib refuses early if VRAM is tight.
        "dtype": "bnb4", "gpu_mem_gib": 0, "need_gib": 20.0,
    },
}

_lock = threading.Lock()   # serializes model switches + inference
_slot = None               # {"key","model","processor","cand_ids","logits_conf"}


def default_prompt():
    """Filled 8-frame 9-class spec prompt (matches batch labeling)."""
    from scripts.vlm_label.vlm_label import default_prompt as _dp
    return _dp()


def _vram_free_gib():
    try:
        import torch
        if not torch.cuda.is_available():
            return 0.0
        free, _total = torch.cuda.mem_get_info()
        return free / 1024 ** 3
    except Exception:  # noqa: BLE001
        return -1.0


def status():
    models = [{"key": k, "desc": m["desc"],
               "ready": os.path.isfile(os.path.join(m["path"], ".download_done"))}
              for k, m in MODELS.items()]
    return {
        "loaded": _slot["key"] if _slot else None,
        "busy": _lock.locked(),
        "vram_free_gib": round(_vram_free_gib(), 1),
        "models": models,
        "default_prompt": default_prompt(),
    }


def _drop_slot():
    global _slot
    if not _slot:
        return
    import torch
    _slot = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def ensure_model(key):
    """Return the live slot for `key`, loading/unloading models as needed."""
    global _slot
    m = MODELS.get(key)
    if not m:
        raise RuntimeError(f"unknown model key: {key}")
    if _slot and _slot["key"] == key:
        return _slot
    if not os.path.isdir(m["path"]):
        raise RuntimeError(f"model dir missing: {m['path']}")
    if _slot:
        _drop_slot()  # free the other model FIRST so switching is possible
    free = _vram_free_gib()
    if free >= 0 and free < m["need_gib"]:
        raise RuntimeError(
            f"显存不足：空闲 {free:.1f} GiB，加载 {m['desc']} 约需 "
            f"{m['need_gib']:.0f} GiB；GPU 被其他进程占用（很可能批量标注"
            f"正在跑），请等标注结束或释放显存后再试")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from scripts.vlm_label import vlm_label as VL
    model, processor, cand_ids, use_conf = VL.load_model(
        m["path"], m["dtype"], m.get("gpu_mem_gib", 0), "playground")
    # Silent-failure guard: a bad device_map can leave params on the meta
    # device (loads "fine", then every window fails with
    # "Cannot copy out of meta tensor"). Detect it right after loading.
    meta = [n for n, p in model.named_parameters() if p.device.type == "meta"]
    if meta:
        _drop_slot()
        raise RuntimeError(
            f"{m['desc']} 加载后仍有 {len(meta)} 个参数在 meta 设备上"
            f"（如 {meta[0]}）；通常是显存不足导致 device_map 异常，"
            f"请关闭占显存的程序后重试")
    _slot = {"key": key, "model": model, "processor": processor,
             "cand_ids": cand_ids, "logits_conf": use_conf}
    return _slot


def _generate_free(slot, pil_frames, prompt, max_new_tokens):
    """Free-form chat completion; returns (text, elapsed_sec)."""
    import torch

    model, processor = slot["model"], slot["processor"]
    messages = [{"role": "user", "content":
                [{"type": "image", "image": im} for im in pil_frames]
                + [{"type": "text", "text": prompt}]}]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt").to(model.device)
    t0 = time.time()
    with torch.inference_mode():
        out = model.generate(**inputs, do_sample=False,
                             max_new_tokens=max_new_tokens)
    text = processor.tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return text, time.time() - t0


def run_adhoc(video_path, prompt="", model_key="8b", window_sec=6.0,
              stride_sec=3.0, frames=8, max_side=448, free_form=False,
              max_new_tokens=0, max_windows=0):
    """Classify one arbitrary local video with a user-editable prompt.

    Label mode (free_form=False): same first-token-softmax confidence as
    batch labeling; the answer must still start with one of
    Resting|Active|Eating|Drinking|None for parsing.
    Free-form mode: raw model text per window, no label parsing.
    """
    if not video_path:
        raise RuntimeError("缺少 video_path")
    if not os.path.isfile(video_path):
        raise RuntimeError(f"视频不存在: {video_path}")
    prompt = (prompt or "").strip() or default_prompt()
    # clamp params (protect window_ranges from degenerate values)
    window_sec = max(1.0, float(window_sec or 6.0))
    stride_sec = max(0.5, float(stride_sec or window_sec))
    frames = min(max(int(frames or 8), 1), 16)
    max_side = min(max(int(max_side or 448), 112), 768)

    with _lock:
        slot = ensure_model(model_key)
        import numpy as np
        import torch
        from PIL import Image
        from scripts.vlm_label import vlm_label as VL

        rel = os.path.basename(video_path)
        _fps, duration, _f = VL.read_frames_at(video_path, [0.0])
        ranges = VL.window_ranges(duration, window_sec, stride_sec)
        truncated = False
        if max_windows and len(ranges) > max_windows:
            ranges = ranges[:max_windows]
            truncated = True
        ts_by_win = [VL.sample_timestamps(s, L, frames) for s, L in ranges]
        all_ts = [t for ts in ts_by_win for t in ts]
        _fps2, dur2, flat = VL.read_frames_at(video_path, all_ts)
        duration = dur2 or duration

        wins, raw, texts, errs = [], [], [], []
        t_all = time.time()
        for i, (start, length) in enumerate(ranges):
            fbgr = VL.resize_frames(flat[i * frames:(i + 1) * frames], max_side)
            pils = [Image.fromarray(np.ascontiguousarray(f[:, :, ::-1]))
                    for f in fbgr]
            end = min(start + stride_sec, duration)
            if not truncated and i == len(ranges) - 1:
                end = duration  # untruncated: the final window tiles to the end
            try:
                if free_form:
                    text, _el = _generate_free(slot, pils, prompt,
                                               max_new_tokens or 256)
                    texts.append({"start": round(start, 3), "end": round(end, 3),
                                  "text": text})
                else:
                    label, conf, probs, perr = VL.classify_window(
                        slot["model"], slot["processor"], slot["cand_ids"],
                        slot["logits_conf"], pils, prompt,
                        max_new_tokens or 40, "playground", rel, i)
                    raw.append((start, end, label, conf))
                    wins.append({"start": round(start, 3), "end": round(end, 3),
                                 "coverage": round(min(start + length, duration), 3),
                                 "label": label, "confidence": conf,
                                 "probs": probs, "parse_error": bool(perr)})
            except Exception as e:  # noqa: BLE001 - keep going window by window
                errs.append(f"win{i} {start:.1f}s: {type(e).__name__}: {e}")
                if free_form:
                    texts.append({"start": round(start, 3), "end": round(end, 3),
                                  "text": f"[ERROR] {e}"})
                else:
                    raw.append((start, end, "unknown", 0.0))
                    wins.append({"start": round(start, 3), "end": round(end, 3),
                                 "coverage": round(min(start + length, duration), 3),
                                 "label": "unknown", "confidence": 0.0,
                                 "probs": None, "parse_error": True})
            if (i + 1) % 8 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

        out = {"ok": True, "video_path": video_path,
               "duration": round(duration, 3), "model": model_key,
               "model_desc": MODELS[model_key]["desc"],
               "free_form": bool(free_form), "prompt": prompt,
               "window_sec": window_sec, "stride_sec": stride_sec,
               "frames": frames, "n_windows": len(ranges),
               "truncated": truncated,
               "elapsed_sec": round(time.time() - t_all, 2)}
        if errs:
            out["errors"] = errs
        if free_form:
            out["texts"] = texts
        else:
            out["windows"] = wins
            out["segments"] = VL.merge_windows(raw)
        return out
