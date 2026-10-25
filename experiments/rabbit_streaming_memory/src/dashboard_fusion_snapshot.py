"""Isolated Rabbit FusionAgent behavior snapshot from ppc_dashboard.

Fusion/FSM only: inference probabilities are supplied by the caller.
"""
from __future__ import annotations
from collections import deque
import numpy as np

CLASS_NAMES = ["resting", "active", "eating", "drinking"]

class DashboardFusionSnapshot:
    BASE = {"EATING": 8, "DRINKING": 6, "ACTIVE": 3, "RESTING": 8}
    MIN = {"EATING": 3, "DRINKING": 2, "ACTIVE": 2, "RESTING": 3}
    VALID = {
        "RESTING": ["ACTIVE", "EATING", "DRINKING"],
        "ACTIVE": ["RESTING", "EATING", "DRINKING"],
        "EATING": ["ACTIVE", "RESTING"],
        "DRINKING": ["ACTIVE", "RESTING"],
    }
    FEEDING_CONFIDENCE_GATE = 0.70
    ACTIVE_CONFIDENCE_GATE = 0.75
    THRESHOLD_BASE_FPS = 24.0

    def __init__(self, track_id=0):
        self.track_id = track_id
        self.state = "RESTING"
        self.state_buffer = {}
        self.center_history = deque(maxlen=6)
        self.last_update_time = 0.0
        self.last_state_time = 0.0
        self.estimated_update_fps = 0.0
        self.speed = self.dx = self.dy = 0.0
        self.mamba_state = None
        self.seq_idx = self.update_idx = 0

    @staticmethod
    def normalize_state(state):
        value = str(state or "").strip().upper()
        return "ACTIVE" if value == "MOVING" else value

    def _update_physics(self, box, timestamp):
        center = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
        self.center_history.append((center, timestamp))
        dx = dy = speed = 0.0
        if len(self.center_history) >= 2:
            first_center, first_time = self.center_history[0]
            last_center, last_time = self.center_history[-1]
            dt = last_time - first_time
            if dt > 0:
                self.estimated_update_fps = (len(self.center_history) - 1) / dt
                dx = (last_center[0] - first_center[0]) / dt
                dy = (last_center[1] - first_center[1]) / dt
                speed = float(np.linalg.norm(last_center - first_center) / dt)
        self.speed, self.dx, self.dy = speed, dx, dy
        rule = "FAST_MOVING" if speed > 150.0 else "STATIONARY" if speed < 45.0 else "SLOW_MOVING"
        return rule

    def _required_frames_for(self, candidate):
        base = self.BASE.get(candidate, 5)
        minimum = self.MIN.get(candidate, 1)
        if self.estimated_update_fps <= 0:
            return minimum
        scaled = round(base * min(self.estimated_update_fps, self.THRESHOLD_BASE_FPS) / self.THRESHOLD_BASE_FPS)
        return max(minimum, int(scaled))

    def _candidate(self, rule_state, probs):
        if rule_state == "FAST_MOVING":
            return "ACTIVE"
        if probs is not None:
            index = int(np.argmax(probs))
            label = self.normalize_state(CLASS_NAMES[index])
            probability = float(probs[index])
            if label in ("EATING", "DRINKING") and probability > self.FEEDING_CONFIDENCE_GATE:
                return label
            if label == "ACTIVE" and probability > self.ACTIVE_CONFIDENCE_GATE:
                return label
        return "RESTING" if rule_state == "STATIONARY" else "ACTIVE"

    def _transition(self, candidate, timestamp):
        previous = self.state
        if candidate == self.state:
            self.state_buffer.pop(self.state, None)
            return {"previous_state": previous, "reason": "same_state", "transitioned": False}
        effective = candidate
        invalid = candidate not in self.VALID.get(self.state, [])
        if invalid:
            effective = "ACTIVE"
        values = self.state_buffer.get(effective, [])
        values = [t for t in values if t >= timestamp - 5.0]
        values.append(timestamp)
        self.state_buffer[effective] = values
        required = self._required_frames_for(effective)
        transitioned = len(values) >= required
        if transitioned:
            self.state = effective
            self.state_buffer = {}
            self.last_state_time = timestamp
        reason = "invalid_transition_fallback_to_active" if invalid else "debounce_accepted" if transitioned else "debounce_waiting"
        return {"previous_state": previous, "reason": reason, "transitioned": transitioned, "required_frames": required}

    def _begin_update(self, box, timestamp):
        reset = self.last_update_time > 0 and timestamp - self.last_update_time > 2.0
        if reset:
            self.mamba_state = None
            self.seq_idx = 0
            self.center_history.clear()
        rule_state = self._update_physics(box, timestamp)
        self.update_idx += 1
        model_ran = rule_state in ("STATIONARY", "SLOW_MOVING") or (rule_state == "FAST_MOVING" and (self.update_idx - 1) % 5 == 0)
        return reset, rule_state, model_ran

    def _finish_update(self, timestamp, reset, rule_state, model_ran, probs):
        candidate = self._candidate(rule_state, probs)
        fsm = self._transition(candidate, timestamp)
        self.last_update_time = timestamp
        if probs is not None:
            self.seq_idx += 1
        return {
            "rule_state": rule_state, "speed_px_s": self.speed,
            "dx": self.dx, "dy": self.dy,
            "estimated_update_fps": self.estimated_update_fps,
            "model_ran": model_ran, "model_prob_used": probs is not None,
            "reset_after_gap": reset, "candidate": candidate,
            "final_state": self.state, "fsm": fsm,
        }

    def update(self, box, probs, timestamp):
        reset, rule_state, model_ran = self._begin_update(box, timestamp)
        used_probs = probs if model_ran else None
        return self._finish_update(timestamp, reset, rule_state, model_ran, used_probs)

    def update_with_predictor(self, box, predictor, timestamp):
        """Run a real model callback only when the copied FusionAgent would."""
        reset, rule_state, model_ran = self._begin_update(box, timestamp)
        probs = None
        if model_ran:
            probs, next_state = predictor(self.mamba_state)
            if probs is not None:
                self.mamba_state = next_state
        return self._finish_update(timestamp, reset, rule_state, model_ran, probs)
