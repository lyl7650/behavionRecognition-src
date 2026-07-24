"""checkpoint 校验 + 真实 head 维度检查。任何不确定即失败。

类别顺序的权威来源是 pipeline_config.SPECIES[species]["class_names"]，由调用方
按品种传入。sidecar <ckpt>.meta.json（{"sha256","class_names","species"}）为可选
交叉校验：若存在，其哈希/类别必须与实际一致，否则拒绝运行。
拿错品种的 checkpoint 由 head 维度检查兜底（pet=3 类，rabbit=4 类）。
"""
import hashlib
import json
import os


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_meta(checkpoint_path, class_names):
    """checkpoint 存在性检查 + 计算 sha256；sidecar 若存在则交叉校验。

    返回 {"sha256", "class_names"}。类别顺序来自调用方（SPECIES 配置）。
    """
    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"teacher checkpoint not found: {checkpoint_path}")
    if not class_names:
        raise ValueError("class_names is empty")
    actual = _sha256(checkpoint_path)
    sidecar = checkpoint_path + ".meta.json"
    if os.path.isfile(sidecar):
        meta = json.loads(open(sidecar, encoding="utf-8").read())
        if meta.get("sha256") != actual:
            raise ValueError(
                f"sidecar hash mismatch: sidecar={meta.get('sha256')} actual={actual}")
        if meta.get("class_names") and list(meta["class_names"]) != list(class_names):
            raise ValueError(
                f"sidecar class_names {meta['class_names']} "
                f"!= species class_names {list(class_names)}")
    return {"sha256": actual, "class_names": list(class_names)}


def _first_dim(weight):
    # 支持真实 torch.Tensor（.shape[0]）与测试用 list（len）
    shape = getattr(weight, "shape", None)
    return int(shape[0]) if shape is not None else len(weight)


def check_head_dim(state_dict, expected_num_classes):
    key = "head.1.weight"
    if key not in state_dict:
        raise ValueError(f"checkpoint missing classifier head key: {key}")
    n = _first_dim(state_dict[key])
    if n != expected_num_classes:
        raise ValueError(
            f"head class count {n} != expected class count {expected_num_classes}")


def verify_and_load_teacher(checkpoint_path, class_names, device="cuda"):
    """meta 校验 → 读 state_dict → head 维度检查 → 构造 SwinTeacher 加载。

    返回 (model, meta)。类别顺序来自传入的 class_names（SPECIES 配置）。import torch。
    """
    import torch
    from models.teacher import SwinTeacher

    meta = resolve_meta(checkpoint_path, class_names)

    raw = torch.load(checkpoint_path, map_location="cpu")
    state = {k.replace("base.model.", "").replace("module.", ""): v
             for k, v in raw.items()}
    check_head_dim(state, expected_num_classes=len(class_names))

    model = SwinTeacher(checkpoint_path=None, num_classes=len(class_names))
    missing, unexpected = model.model.load_state_dict(state, strict=False)
    hard_missing = [k for k in missing if not k.startswith("head.")]
    if hard_missing or unexpected:
        raise ValueError(
            f"checkpoint load mismatch missing={hard_missing} unexpected={list(unexpected)}")
    model = model.to(device).eval()
    return model, meta
