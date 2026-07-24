"""共享逻辑：other → active/resting/eat/drink 重新拆分工作流。

唯一集中写"拆分后要哪些类别"和"怎么找散落的 other 目录"的地方，
供 seed / review / apply 三个入口脚本共用。
"""
import os

SPLIT_CLASSES = ["resting", "active", "eat", "drink"]


def find_other_dirs(root):
    """递归找出 root 下所有名字（不分大小写）恰好是 "other" 的目录。"""
    found = []
    for dirpath, dirnames, _ in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for d in dirnames:
            if d.lower() == "other":
                found.append(os.path.join(dirpath, d))
    return sorted(found)


def build_prefixes(base, other_dirs):
    """把每个 other 目录映射成"相对 base 的路径前缀"，用作跨目录命名空间。

    返回 [(prefix, physical_dir), ...]，按前缀长度降序（供 resolve_prefix 最长匹配）。
    发包（挑视频）与收包（放回标注）都用同一个 base（pet raw_root）算前缀，
    因此同一个视频发出去和收回来的命名空间一致，能原样归位。
    """
    base_real = os.path.realpath(base)
    out = []
    for d in other_dirs:
        d_real = os.path.realpath(d)
        prefix = os.path.relpath(d_real, base_real).replace("\\", "/")
        out.append((prefix, d_real))
    out.sort(key=lambda pair: -len(pair[0]))
    return out


def resolve_prefix(prefixes, rel):
    """把命名空间化的 rel（prefix + "/" + 文件名）反解析成 (physical_dir, 文件名)。

    无法匹配任何前缀时返回 (None, None)。
    """
    norm = rel.replace("\\", "/")
    for prefix, d in prefixes:
        if norm == prefix or norm.startswith(prefix + "/"):
            return d, norm[len(prefix):].lstrip("/")
    return None, None
