"""Minimal isolated production-style FSM decoder.

Copied behavior only: candidate debounce, valid transitions, and configurable
required candidate counts. It does not import dashboard code.
"""
from __future__ import annotations
from collections import defaultdict

class RabbitFSM:
    BASE = {"EATING": 8, "DRINKING": 6, "ACTIVE": 3, "RESTING": 8}
    MIN = {"EATING": 3, "DRINKING": 2, "ACTIVE": 2, "RESTING": 3}
    VALID = {"RESTING": ["ACTIVE", "EATING", "DRINKING"], "ACTIVE": ["RESTING", "EATING", "DRINKING"], "EATING": ["ACTIVE", "RESTING"], "DRINKING": ["ACTIVE", "RESTING"]}
    def __init__(self, fps: float = 2.0, buffer_window_s: float = 5.0, scale_thresholds: bool = True):
        self.state = "RESTING"
        self.fps = fps
        self.buffer_window_s = buffer_window_s
        self.scale_thresholds = scale_thresholds
        self.pending = defaultdict(list)
    def required(self, candidate: str) -> int:
        base = self.BASE.get(candidate, 5)
        minimum = self.MIN.get(candidate, 1)
        if not self.scale_thresholds:
            return minimum
        return max(minimum, round(base * min(self.fps, 24.0) / 24.0))
    def update(self, candidate: str, timestamp_s: float) -> tuple[str, dict]:
        previous = self.state
        if candidate == self.state:
            self.pending.pop(self.state, None)
            return self.state, {"previous_state": previous, "reason": "same_state", "transitioned": False}
        effective = candidate if candidate in self.VALID.get(self.state, []) else "ACTIVE"
        timestamps = [t for t in self.pending[effective] if t >= timestamp_s - self.buffer_window_s]
        timestamps.append(timestamp_s)
        self.pending[effective] = timestamps
        transitioned = False
        if len(timestamps) >= self.required(effective):
            self.state = effective
            self.pending.clear()
            transitioned = True
        reason = "debounce_accepted" if transitioned else "debounce_waiting"
        if effective != candidate:
            reason = "invalid_transition_fallback_to_active"
        return self.state, {"previous_state": previous, "reason": reason, "transitioned": transitioned}
