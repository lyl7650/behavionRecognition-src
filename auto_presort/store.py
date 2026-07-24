"""`.autopresort/` JSON 存取层。纯 stdlib。

约定：doc 文件名 = rel_path 扁平化（分隔符→"__"）+ ".json"，存于 <batch_dir>/.autopresort/。
update_doc(action="save") 强制 approved=False（任何编辑使批准失效，硬性规则）；
update_doc(action="approve") 仅置 approved=True，不改 segments。
"""
import json
import os

STORE_DIRNAME = ".autopresort"


def _store_dir(batch_dir):
    return os.path.join(batch_dir, STORE_DIRNAME)


def _doc_filename(rel_path):
    flat = rel_path.replace("\\", "/").replace("/", "__")
    return flat + ".json"


def _doc_path(batch_dir, rel_path):
    return os.path.join(_store_dir(batch_dir), _doc_filename(rel_path))


def save_doc(batch_dir, doc):
    os.makedirs(_store_dir(batch_dir), exist_ok=True)
    path = _doc_path(batch_dir, doc["rel_path"])
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)
    return path


def load_doc(batch_dir, rel_path):
    with open(_doc_path(batch_dir, rel_path), encoding="utf-8") as f:
        return json.load(f)


def list_docs(batch_dir):
    d = _store_dir(batch_dir)
    if not os.path.isdir(d):
        return []
    docs = []
    for name in sorted(os.listdir(d)):
        if name.endswith(".json") and name != "manifest.json":
            with open(os.path.join(d, name), encoding="utf-8") as f:
                docs.append(json.load(f))
    return docs


def update_doc(batch_dir, rel_path, segments, action):
    doc = load_doc(batch_dir, rel_path)
    if action == "save":
        if segments is not None:
            doc["segments"] = segments
        doc["approved"] = False
    elif action == "approve":
        doc["approved"] = True
    else:
        raise ValueError(f"unknown action: {action}")
    save_doc(batch_dir, doc)
    return doc


def save_manifest(batch_dir, manifest):
    os.makedirs(_store_dir(batch_dir), exist_ok=True)
    path = os.path.join(_store_dir(batch_dir), "manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return path


def load_manifest(batch_dir):
    path = os.path.join(_store_dir(batch_dir), "manifest.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)
