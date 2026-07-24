"""apply 纯逻辑：Review-all 闸门、输出命名（内嵌来源视频名 + 时间段）、ffmpeg 精确切片命令。

命名 <原视频文件名去扩展名>-<start>-<end>.mp4，例如
1057464119_..._ff505f5d-2.5-19.5.mp4。来源分组由 abnormal_crop_video.source_key
按"剥掉尾部 -<start>-<end> 时间段"还原（同一原视频的各切片共享同一 stem，不跨 train/val）。
禁止 -c copy（关键帧吸附会抵消 inward trim）。
"""
import hashlib
import os


def source_id_for(rel_path):
    norm = rel_path.replace("\\", "/")
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()[:8]


def plan_cuts(docs):
    """返回 (cuts, skipped_rel_paths)。只收 approved=True 的 doc（Review-all 闸门）。"""
    cuts, skipped = [], []
    for doc in docs:
        if not doc.get("approved"):
            skipped.append(doc["rel_path"])
            continue
        for seg in doc["segments"]:
            cuts.append({
                "rel_path": doc["rel_path"],
                "source_id": doc["source_id"],
                "label": seg["label"],
                "start": float(seg["start"]),
                "end": float(seg["end"]),
            })
    return cuts, skipped


def clip_output_name(rel_path, start, end):
    """<原视频文件名去扩展名>-<start>-<end>.mp4（label 由所在类别子目录体现）。"""
    stem = os.path.splitext(os.path.basename(rel_path.replace("\\", "/")))[0]
    return f"{stem}-{round(start, 3):g}-{round(end, 3):g}.mp4"


def build_ffmpeg_cmd(input_path, output_path, start, end):
    """精确重编码切片：-ss 前置 + 重编码（帧级精确），-t 用时长，无音轨。"""
    duration = round(end - start, 6)
    return [
        "ffmpeg", "-y",
        "-ss", f"{start:g}",
        "-i", input_path,
        "-t", str(duration),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-an",
        output_path,
    ]


def duration_ok(expected, actual, tolerance=0.25):
    return abs(actual - expected) <= tolerance
