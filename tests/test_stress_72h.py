"""
72-Hour Stress Test Framework for TetraMem-XL.

Run: python -m tests.test_stress_72h --duration-hours 72
"""

import argparse
import json
import logging
import os
import sys
import time
import threading
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tetrahedron_memory.core import GeoMemoryBody
from tetrahedron_memory.tetra_mesh import TetraMesh

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("stress_72h")


class StressTestRunner:
    def __init__(self, duration_hours: float = 72.0, report_interval: int = 300):
        self._duration = duration_hours * 3600
        self._report_interval = report_interval
        self._stop = threading.Event()
        self._body = GeoMemoryBody(dimension=3, precision="fast")
        self._stats: Dict[str, Any] = {
            "total_stores": 0,
            "total_queries": 0,
            "total_dreams": 0,
            "total_self_orgs": 0,
            "total_multiparam_queries": 0,
            "errors": [],
            "memory_snapshots": [],
            "start_time": 0,
            "end_time": 0,
        }
        self._lock = threading.Lock()

    def run(self) -> Dict[str, Any]:
        self._stats["start_time"] = time.time()
        logger.info("Starting %d-hour stress test", self._duration / 3600)

        threads = [
            threading.Thread(target=self._insert_loop, name="insert", daemon=True),
            threading.Thread(target=self._query_loop, name="query", daemon=True),
            threading.Thread(target=self._dream_loop, name="dream", daemon=True),
            threading.Thread(target=self._monitor_loop, name="monitor", daemon=True),
        ]

        for t in threads:
            t.start()

        self._stop.wait(timeout=self._duration)
        self._stop.set()
        logger.info("Stop signal sent, waiting for threads...")

        for t in threads:
            t.join(timeout=30)

        self._stats["end_time"] = time.time()
        self._stats["elapsed_hours"] = (self._stats["end_time"] - self._stats["start_time"]) / 3600
        self._stats["final_memory_count"] = len(self._body._nodes)

        logger.info("Stress test complete: %d stores, %d queries, %d dreams, %d errors",
                     self._stats["total_stores"], self._stats["total_queries"],
                     self._stats["total_dreams"], len(self._stats["errors"]))

        return self._stats

    def _insert_loop(self) -> None:
        i = 0
        while not self._stop.is_set():
            try:
                self._body.store(
                    content=f"stress_mem_{i}",
                    labels=[f"cat_{i % 10}", f"tag_{i % 5}"],
                    weight=0.5 + (i % 10) * 0.1,
                )
                with self._lock:
                    self._stats["total_stores"] += 1
                i += 1
                self._stop.wait(timeout=0.001)
            except Exception as e:
                self._record_error("insert", str(e))
                self._stop.wait(timeout=1.0)

    def _query_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._body.query(f"stress query {np.random.randint(100)}", k=5)
                with self._lock:
                    self._stats["total_queries"] += 1

                if self._stats["total_queries"] % 10 == 0:
                    self._body.query_multiparam("stress multiparam", k=5)
                    with self._lock:
                        self._stats["total_multiparam_queries"] += 1

                self._stop.wait(timeout=0.01)
            except Exception as e:
                self._record_error("query", str(e))
                self._stop.wait(timeout=1.0)

    def _dream_loop(self) -> None:
        while not self._stop.is_set():
            try:
                if self._body._use_mesh and len(self._body._mesh.tetrahedra) >= 5:
                    from tetrahedron_memory.tetra_dream import TetraDreamCycle
                    dc = TetraDreamCycle(self._body._mesh, cycle_interval=999999)
                    dc.trigger_now()
                    with self._lock:
                        self._stats["total_dreams"] += 1

                self._body.self_organize()
                with self._lock:
                    self._stats["total_self_orgs"] += 1

                self._stop.wait(timeout=5.0)
            except Exception as e:
                self._record_error("dream", str(e))
                self._stop.wait(timeout=10.0)

    def _monitor_loop(self) -> None:
        while not self._stop.is_set():
            try:
                snap = {
                    "timestamp": time.time(),
                    "memories": len(self._body._nodes),
                    "tetrahedra": len(self._body._mesh.tetrahedra),
                    "protector": self._body._protector.get_status(),
                }
                with self._lock:
                    self._stats["memory_snapshots"].append(snap)
                    if len(self._stats["memory_snapshots"]) > 1000:
                        self._stats["memory_snapshots"] = self._stats["memory_snapshots"][-500:]

                logger.info(
                    "Monitor: %d memories, %d tetra, %d stores, %d queries, %d dreams",
                    snap["memories"], snap["tetrahedra"],
                    self._stats["total_stores"], self._stats["total_queries"],
                    self._stats["total_dreams"],
                )
            except Exception as e:
                self._record_error("monitor", str(e))

            self._stop.wait(timeout=self._report_interval)

    def _record_error(self, operation: str, error: str) -> None:
        with self._lock:
            self._stats["errors"].append({
                "timestamp": time.time(),
                "operation": operation,
                "error": error,
            })
            if len(self._stats["errors"]) > 500:
                self._stats["errors"] = self._stats["errors"][-250:]


def main():
    parser = argparse.ArgumentParser(description="TetraMem-XL 72-Hour Stress Test")
    parser.add_argument("--duration-hours", type=float, default=72.0)
    parser.add_argument("--report-interval", type=int, default=300)
    parser.add_argument("--output", type=str, default="stress_72h_report.json")
    args = parser.parse_args()

    runner = StressTestRunner(
        duration_hours=args.duration_hours,
        report_interval=args.report_interval,
    )
    result = runner.run()

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\nReport written to {args.output}")
    print(f"Duration: {result['elapsed_hours']:.1f} hours")
    print(f"Stores: {result['total_stores']}")
    print(f"Queries: {result['total_queries']}")
    print(f"Dreams: {result['total_dreams']}")
    print(f"Errors: {len(result['errors'])}")
    print(f"Final memories: {result['final_memory_count']}")


if __name__ == "__main__":
    main()
