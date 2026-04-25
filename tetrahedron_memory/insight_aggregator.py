import time
import threading
from typing import Any, Dict, List, Optional


class InsightAggregator:
    def __init__(self):
        self._insights: List[Dict] = []
        self._lock = threading.Lock()

    def add_insight(self, insight_type: str, title: str, description: str,
                    priority: str = "medium", metadata: Optional[Dict] = None):
        insight = {
            "type": insight_type,
            "title": title,
            "description": description,
            "priority": priority,
            "metadata": metadata or {},
            "ts": time.time(),
        }
        with self._lock:
            self._insights.append(insight)
            if len(self._insights) > 500:
                self._insights = self._insights[-500:]

    def get_insights(self, n: int = 20, priority: Optional[str] = None) -> List[Dict]:
        with self._lock:
            insights = list(self._insights)
        if priority:
            insights = [i for i in insights if i.get("priority") == priority]
        priority_order = {"high": 0, "medium": 1, "low": 2}
        insights.sort(key=lambda x: priority_order.get(x.get("priority", "medium"), 3))
        return insights[:n]

    def clear(self):
        with self._lock:
            self._insights.clear()

    def collect(self) -> List[Dict]:
        priority_map = {"high": 8, "medium": 5, "low": 2}
        with self._lock:
            insights = list(self._insights)
        result = []
        for ins in insights:
            numeric_priority = priority_map.get(ins.get("priority", "medium"), 5)
            if isinstance(ins.get("metadata", {}).get("priority"), (int, float)):
                numeric_priority = ins["metadata"]["priority"]
            result.append({
                "type": ins.get("type", "unknown"),
                "title": ins.get("title", ""),
                "description": ins.get("description", ""),
                "priority": numeric_priority,
                "action": ins.get("metadata", {}).get("action", ""),
                "ts": ins.get("ts", 0),
            })
        return result

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            total = len(self._insights)
            by_type = {}
            by_priority = {}
            for ins in self._insights:
                t = ins.get("type", "unknown")
                p = ins.get("priority", "medium")
                by_type[t] = by_type.get(t, 0) + 1
                by_priority[p] = by_priority.get(p, 0) + 1
        return {"total": total, "by_type": by_type, "by_priority": by_priority}
