"""
TetraDistributedController — Unified distributed API for TetraMem-XL.

Bridges TetraMeshRouter (TetraBucket/TetraMesh) with optional Ray actors.
Provides a single entry point for:
  - Distributed store/query/associate
  - Cross-bucket topology navigation
  - Distributed dream cycles
  - Distributed self-organization
  - Auto-balancing and ghost cell management
"""

import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .partitioning import BoundingBox, GhostCell
from .tetra_mesh import TetraMesh
from .tetra_router import TetraBucket, TetraMeshRouter

logger = logging.getLogger("tetramem.distributed")


class TetraDistributedController:
    """Unified distributed controller for TetraMem-XL.

    Architecture:
      - TetraMeshRouter handles spatial partitioning into buckets
      - Each bucket contains a local TetraMesh
      - Ghost cells bridge boundaries for cross-bucket topology
      - Optional Ray actors for true multi-process distribution

    Usage:
        ctrl = TetraDistributedController(num_buckets=4)
        ctrl.initialize()
        bid, tid = ctrl.store("memory", seed_point=np.array([1,0,0]), labels=["ai"])
        results = ctrl.query(np.array([1,0,0]), k=5)
        ctrl.run_dream_cycle()
    """

    def __init__(
        self,
        num_buckets: int = 4,
        max_tetra_per_bucket: int = 2000,
        ghost_ttl: float = 3600.0,
        use_ray: bool = False,
        initial_bounds: Optional[BoundingBox] = None,
    ):
        self._num_buckets = num_buckets
        self._use_ray = use_ray
        self._initial_bounds = initial_bounds
        self._router = TetraMeshRouter(
            max_tetra_per_bucket=max_tetra_per_bucket,
            ghost_ttl=ghost_ttl,
        )
        self._ray_initialized = False
        self._lock = threading.RLock()
        self._dream_cycle_count = 0
        self._self_org_count = 0
        self._store_count = 0

    def initialize(self) -> Dict[str, Any]:
        bounds = self._initial_bounds or BoundingBox(
            np.array([-10.0, -10.0, -10.0]),
            np.array([10.0, 10.0, 10.0]),
        )
        self._router.initialize(bounds)

        if self._use_ray:
            try:
                import ray
                if not ray.is_initialized():
                    ray.init(ignore_reinit_error=True)
                self._ray_initialized = True
                logger.info("Ray initialized for TetraMem-XL distributed mode")
            except ImportError:
                logger.warning("Ray not available, running in local mode")
                self._use_ray = False
            except Exception as e:
                logger.warning("Ray init failed: %s, running in local mode", e)
                self._use_ray = False

        return {
            "initialized": True,
            "ray_mode": self._ray_initialized,
            "buckets": len(self._router.get_all_bucket_ids()),
        }

    def store(
        self,
        content: str,
        seed_point: np.ndarray,
        labels: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        weight: float = 1.0,
    ) -> Tuple[str, str]:
        bid, tid = self._router.route_store(
            seed_point, content,
            labels=labels, metadata=metadata, weight=weight,
        )
        self._store_count += 1
        return bid, tid

    def store_secondary(self, bucket_id: str, tetra_id: str,
                         content: str, **kwargs) -> Optional[int]:
        bucket = self._router.get_bucket(bucket_id)
        if bucket is None:
            return None
        return bucket.mesh.store_secondary(tetra_id, content, **kwargs)

    def query(
        self,
        query_point: np.ndarray,
        k: int = 5,
    ) -> List[Tuple[str, float, str]]:
        return self._router.route_query(query_point, k=k)

    def associate(
        self,
        bucket_id: str,
        tetra_id: str,
        max_depth: int = 2,
    ) -> List[Tuple[str, float, str, str]]:
        return self._router.cross_bucket_associate(
            bucket_id, tetra_id, max_depth=max_depth,
        )

    def navigate_cross_bucket(
        self,
        bucket_id: str,
        tetra_id: str,
        max_hops: int = 20,
        cross_bucket_limit: int = 3,
    ) -> List[Tuple[str, float, str, str]]:
        return self._router.navigate_cross_bucket(
            bucket_id, tetra_id,
            max_hops=max_hops, cross_bucket_limit=cross_bucket_limit,
        )

    def abstract_reorganize(
        self,
        bucket_id: Optional[str] = None,
        min_density: int = 2,
        max_operations: int = 20,
    ) -> Dict[str, Any]:
        results = []
        target_ids = [bucket_id] if bucket_id else self._router.get_all_bucket_ids()
        for bid in target_ids:
            bucket = self._router.get_bucket(bid)
            if bucket is not None:
                r = bucket.mesh.abstract_reorganize(
                    min_density=min_density, max_operations=max_operations,
                )
                r["bucket_id"] = bid
                results.append(r)
        return {
            "buckets_processed": len(results),
            "results": results,
        }

    def run_dream_cycle(self, walk_steps: int = 12) -> Dict[str, Any]:
        stats = self._router.distributed_dream(walk_steps=walk_steps)
        self._dream_cycle_count += 1
        return stats

    def run_self_organization(self, max_iterations: int = 5) -> Dict[str, Any]:
        stats = self._router.distributed_self_org(max_iterations=max_iterations)
        self._self_org_count += 1
        return stats

    def auto_balance(self) -> Dict[str, Any]:
        return self._router.auto_balance()

    def get_bucket(self, bucket_id: str) -> Optional[TetraBucket]:
        return self._router.get_bucket(bucket_id)

    def get_all_bucket_ids(self) -> List[str]:
        return self._router.get_all_bucket_ids()

    def get_statistics(self) -> Dict[str, Any]:
        router_stats = self._router.get_statistics()
        router_stats["store_count"] = self._store_count
        router_stats["dream_cycles"] = self._dream_cycle_count
        router_stats["self_org_cycles"] = self._self_org_count
        router_stats["ray_mode"] = self._ray_initialized
        return router_stats

    def shutdown(self) -> None:
        if self._ray_initialized:
            try:
                import ray
                if ray.is_initialized():
                    ray.shutdown()
                self._ray_initialized = False
            except Exception:
                pass
