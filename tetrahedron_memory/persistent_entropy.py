"""
Persistent Entropy — Topological noise/integration signal for TetraMem-XL.

Per Grok's design specification:
  - Persistent entropy measures the information content of the persistence barcode
  - Higher entropy = more topological noise = needs integration
  - Lower entropy after dream/integration cycle indicates successful consolidation
  - Target: persistent entropy drops >= 18% after dream cycles

The persistent entropy formula:
    H = -sum_i (p_i * log(p_i))
where p_i = (d_i - b_i) / sum_j (d_j - b_j) for all persistent intervals (b_j, d_j)

This is used as:
  1. Integration catalyst trigger: when entropy exceeds threshold, trigger integration
  2. Dream cycle quality metric: entropy drop validates dream effectiveness
  3. Self-organization convergence signal: entropy stabilizes at steady state
"""

import logging
import threading
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("tetramem.entropy")


def compute_persistent_entropy(
    simplex_tree,
    dimensions: Optional[List[int]] = None,
) -> float:
    if simplex_tree is None:
        return 0.0

    if dimensions is None:
        dimensions = [0, 1, 2]

    all_intervals = []
    for dim in dimensions:
        try:
            intervals = simplex_tree.persistence_intervals_in_dimension(dim)
            if intervals is not None and len(intervals) > 0:
                for iv in intervals:
                    birth, death = float(iv[0]), float(iv[1])
                    if np.isfinite(death) and death > birth:
                        all_intervals.append((birth, death))
        except Exception:
            continue

    if not all_intervals:
        return 0.0

    persistences = np.array([d - b for b, d in all_intervals])
    total = np.sum(persistences)

    if total <= 0:
        return 0.0

    probs = persistences / total
    probs = probs[probs > 0]

    entropy = float(-np.sum(probs * np.log(probs)))

    return entropy


def compute_entropy_by_dimension(
    simplex_tree,
    dimensions: Optional[List[int]] = None,
) -> Dict[int, float]:
    if simplex_tree is None:
        return {}

    if dimensions is None:
        dimensions = [0, 1, 2]

    result = {}
    for dim in dimensions:
        try:
            intervals = simplex_tree.persistence_intervals_in_dimension(dim)
            if intervals is not None and len(intervals) > 0:
                persistences = []
                for iv in intervals:
                    birth, death = float(iv[0]), float(iv[1])
                    if np.isfinite(death) and death > birth:
                        persistences.append(death - birth)
                if persistences:
                    p = np.array(persistences)
                    total = np.sum(p)
                    if total > 0:
                        probs = p / total
                        probs = probs[probs > 0]
                        result[dim] = float(-np.sum(probs * np.log(probs)))
                    else:
                        result[dim] = 0.0
                else:
                    result[dim] = 0.0
            else:
                result[dim] = 0.0
        except Exception:
            result[dim] = 0.0

    return result


def compute_entropy_delta(
    entropy_before: float,
    entropy_after: float,
) -> float:
    if entropy_before <= 0:
        return 0.0
    return (entropy_before - entropy_after) / entropy_before


def should_trigger_integration(
    current_entropy: float,
    baseline_entropy: float,
    threshold_ratio: float = 1.3,
) -> bool:
    if baseline_entropy <= 0:
        return current_entropy > 0.5
    return current_entropy > baseline_entropy * threshold_ratio


class EntropyTracker:
    def __init__(self, window_size: int = 20):
        self._history: List[float] = []
        self._window_size = window_size
        self._baseline: Optional[float] = None
        self._lock = threading.RLock()

    def record(self, entropy: float) -> None:
        with self._lock:
            self._history.append(entropy)
            if len(self._history) > self._window_size:
                self._history = self._history[-self._window_size :]

    @property
    def baseline(self) -> float:
        with self._lock:
            if self._baseline is not None:
                return self._baseline
            if len(self._history) >= 3:
                return float(np.mean(self._history[:3]))
            return self._history[-1] if self._history else 0.0

    @baseline.setter
    def baseline(self, value: float) -> None:
        with self._lock:
            self._baseline = value

    @property
    def current(self) -> float:
        with self._lock:
            return self._history[-1] if self._history else 0.0

    @property
    def trend(self) -> str:
        with self._lock:
            if len(self._history) < 3:
                return "insufficient_data"
            recent = self._history[-3:]
        if recent[-1] < recent[0] * 0.9:
            return "decreasing"
        elif recent[-1] > recent[0] * 1.1:
            return "increasing"
        return "stable"

    @property
    def last_delta(self) -> float:
        with self._lock:
            if len(self._history) < 2:
                return 0.0
            return compute_entropy_delta(self._history[-2], self._history[-1])

    def should_integrate(self, threshold_ratio: float = 1.3) -> bool:
        return should_trigger_integration(self.current, self.baseline, threshold_ratio)

    def get_summary(self) -> Dict[str, float]:
        with self._lock:
            return {
                "current": self._history[-1] if self._history else 0.0,
                "baseline": self._baseline
                if self._baseline is not None
                else (float(np.mean(self._history[:3])) if len(self._history) >= 3 else 0.0),
                "trend": self.trend,
                "last_delta": self.last_delta,
                "history_len": len(self._history),
            }
