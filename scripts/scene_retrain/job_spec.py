"""JobSpec：编排器与标注/Review 子进程之间的配置载体。

一次 retrain 的全部输入参数序列化成 job.json 放在工作区，编排器重启/子进程
都可以从它恢复上下文，避免进程间重复交互问答。
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field


@dataclass
class JobSpec:
    species: str
    date: str
    videos_dir: str                        # 原始长视频批次目录（.autopresort/ 就近存放）
    teacher_checkpoint: str                # 预打标 teacher .pth
    classes: list[str] = field(default_factory=list)
    frames: bool = False                   # False=视频条快照（默认），True=帧格式快照
    threshold: float = 0.90                # 预打标置信阈值
    trim_margin: float = 1.0               # 预打标切分余量（秒）

    # ---------- 序列化 ----------
    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    def write(self, path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())

    @classmethod
    def from_dict(cls, data: dict) -> "JobSpec":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})

    @classmethod
    def read(cls, path) -> "JobSpec":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
