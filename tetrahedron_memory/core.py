"""
Core module for Tetrahedron Memory System.
"""

import hashlib
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

from .partitioning import BoundingBox, Octree
from .tetra_mesh import TetraMesh, MemoryTetrahedron

import numpy as np

import logging

logger = logging.getLogger("tetramem.core")

try:
    import gudhi

    GUDHI_AVAILABLE = True
except ImportError:
    GUDHI_AVAILABLE = False
    gudhi = None


@dataclass
class MemoryNode:
    id: str
    content: str
    geometry: np.ndarray
    timestamp: float
    weight: float = 1.0
    labels: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryResult:
    node: MemoryNode
    distance: float
    persistence_score: float = 0.0
    association_type: str = "direct"


class GeoMemoryBody:
    """
    Core geometric memory body using Weighted Alpha Complex.
    """

    def __init__(
        self,
        dimension: int = 3,
        precision: Literal["fast", "safe", "exact"] = "fast",
        bucket_id: Optional[str] = None,
        persistence: Optional[Any] = None,
        consistency: Optional[Any] = None,
        auto_emerge_interval: float = 0.0,
    ):
        if precision not in ("fast", "safe", "exact"):
            raise ValueError(f"Invalid precision: {precision}. Must be 'fast', 'safe', or 'exact'")

        self.dimension = dimension
        self.precision: Literal["fast", "safe", "exact"] = precision
        self._nodes_dict: Dict[str, MemoryNode] = {}
        self._alpha_complex: Optional[Any] = None
        self._simplex_tree: Optional[Any] = None
        self._lock = threading.RLock()
        self._needs_rebuild = True
        self._label_index_legacy: Dict[str, Set[str]] = defaultdict(set)
        self._geometry_cache: Optional[np.ndarray] = None
        self._weights: Optional[np.ndarray] = None
        self._self_org_enabled = True
        self._ph_threshold = 0.5
        self._id_to_idx: Dict[str, int] = {}
        self._idx_to_id: Dict[int, str] = {}
        self._query_count = 0
        self._topology_shortcuts: Dict[str, Set[str]] = defaultdict(set)
        self._topology_shortcuts_max_per_node = 50
        self._topology_shortcuts_max_total = 5000
        self._label_sector_cache: Dict[str, np.ndarray] = {}
        self._label_sector_cache_max = 200
        self._dirty_nodes: Set[str] = set()
        self._rebuild_threshold: int = 20
        self._octree: Optional[Octree] = None
        self._octree_initialized: bool = False
        self._cached_avg_persistence: float = 0.0
        self._cached_max_weight: float = 1.0
        self._mesh: TetraMesh = TetraMesh(time_lambda=0.001)
        self._use_mesh: bool = True
        self._mesh_node_map: Dict[str, np.ndarray] = {}
        self._self_organizer: Optional[Any] = None

        self._bucket_id: str = bucket_id or "default"
        self._namespace: str = ""
        self._persistence: Optional[Any] = persistence
        self._consistency: Optional[Any] = consistency
        self._auto_emerge_interval = auto_emerge_interval
        self._emerge_thread: Optional[threading.Thread] = None
        self._emerge_stop = threading.Event()
        self._persist_dirty_count = 0
        self._persist_flush_interval = 50
        self._max_content_length = 100_000
        self._max_memories = 0
        self._default_ttl: float = 0.0
        self._nodes_cache: Optional[Dict[str, "MemoryNode"]] = None
        self._nodes_cache_valid = False

        from .emergence import AdaptiveThreshold, EmergencePressure

        self._adaptive_threshold = AdaptiveThreshold()
        self._emergence_pressure = EmergencePressure()

        from .zigzag_persistence import ZigzagTracker
        from .resolution_pyramid import ResolutionPyramid

        self._zigzag_tracker = ZigzagTracker()
        self._pyramid = ResolutionPyramid()

        from .circuit_breaker import EmergenceProtector

        self._protector = EmergenceProtector()

        from .eternity_audit import EternityAudit

        self._eternity_audit = EternityAudit()

        if auto_emerge_interval > 0:
            self.start_emergence_daemon(auto_emerge_interval)

    @property
    def _nodes(self) -> Dict[str, "MemoryNode"]:
        if not self._use_mesh:
            return self._nodes_dict
        if self._nodes_cache_valid and self._nodes_cache is not None:
            return self._nodes_cache
        with self._mesh._lock:
            if not self._mesh._tetrahedra:
                self._nodes_cache = {}
                self._nodes_cache_valid = True
                return {}
            result = {}
            for tid, tetra in self._mesh._tetrahedra.items():
                result[tid] = MemoryNode(
                    id=tetra.id,
                    content=tetra.content,
                    geometry=tetra.centroid.copy(),
                    timestamp=tetra.creation_time,
                    weight=tetra.weight,
                    labels=list(tetra.labels),
                    metadata=dict(tetra.metadata),
                )
            self._nodes_cache = result
            self._nodes_cache_valid = True
            return result

    @_nodes.setter
    def _nodes(self, value):
        if not self._use_mesh:
            object.__setattr__(self, "_nodes_dict", value)
        self._nodes_cache_valid = False
        self._nodes_cache = None

    def _invalidate_nodes_cache(self):
        self._nodes_cache_valid = False
        self._nodes_cache = None

    @property
    def _label_index(self):
        if self._use_mesh:
            return self._mesh.label_index
        return self._label_index_legacy

    def _validate_content(self, content: str) -> None:
        if content is None:
            raise ValueError("content must not be None")
        if not isinstance(content, str):
            raise TypeError(f"content must be str, got {type(content).__name__}")
        if len(content) == 0:
            raise ValueError("content must not be empty")
        if len(content) > self._max_content_length:
            raise ValueError(
                f"content length {len(content)} exceeds limit {self._max_content_length}"
            )

    def store(
        self,
        content: str,
        labels: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        weight: float = 1.0,
        ttl: float = 0.0,
    ) -> str:
        self._validate_content(content)
        if weight is None or not isinstance(weight, (int, float)):
            weight = 1.0

        if (
            self._max_memories > 0
            and self._use_mesh
            and len(self._mesh._tetrahedra) >= self._max_memories
        ):
            raise ValueError(f"Memory capacity limit reached ({self._max_memories})")

        effective_metadata = dict(metadata or {})
        if self._namespace:
            effective_metadata["_namespace"] = self._namespace
        effective_ttl = ttl if ttl > 0 else self._default_ttl
        if effective_ttl > 0:
            effective_metadata["_ttl"] = time.time() + effective_ttl

        geometry = self._text_to_geometry(content, labels=labels)

        if self._use_mesh:
            with self._lock:
                tetra_id = self._mesh.store(
                    content=content,
                    seed_point=geometry,
                    labels=labels,
                    metadata=effective_metadata,
                    weight=weight,
                )
                self._mesh_node_map[tetra_id] = geometry
                self._record_version(tetra_id, content)
                self._auto_persist(tetra_id, content)
                self._eternity_audit.record_store(tetra_id, content, effective_metadata)
                self._invalidate_nodes_cache()
                return tetra_id

        node_id = self._generate_id(content)
        node = MemoryNode(
            id=node_id,
            content=content,
            geometry=geometry,
            timestamp=time.time(),
            weight=weight,
            labels=labels or [],
            metadata=effective_metadata,
        )
        with self._lock:
            self._nodes[node_id] = node
            for label in node.labels:
                self._label_index[label].add(node_id)
            self._dirty_nodes.add(node_id)
            if len(self._dirty_nodes) >= self._rebuild_threshold:
                self._needs_rebuild = True
            self._update_octree_insert(node_id, node)
            self._rebuild_index_cache()
            self._record_version(node_id, content)
            self._auto_persist(node_id, content)
            self._eternity_audit.record_store(node_id, content, effective_metadata)
            self._invalidate_nodes_cache()
        return node_id

    def get(self, memory_id: str) -> Optional[Dict[str, Any]]:
        if memory_id is None:
            return None
        if self._use_mesh:
            with self._mesh._lock:
                tetra = self._mesh._tetrahedra.get(memory_id)
                if tetra is None:
                    return None
                return {
                    "id": tetra.id,
                    "content": tetra.content,
                    "weight": tetra.weight,
                    "labels": list(tetra.labels),
                    "metadata": dict(tetra.metadata),
                    "timestamp": tetra.creation_time,
                    "last_access": tetra.last_access_time,
                    "access_count": tetra.access_count,
                }
        with self._lock:
            node = self._nodes_dict.get(memory_id)
            if node is None:
                return None
            return {
                "id": node.id,
                "content": node.content,
                "weight": node.weight,
                "labels": list(node.labels),
                "metadata": dict(node.metadata),
                "timestamp": node.timestamp,
            }

    def update(
        self,
        memory_id: str,
        content: Optional[str] = None,
        labels: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        weight: Optional[float] = None,
    ) -> bool:
        if memory_id is None:
            return False
        if content is not None:
            self._validate_content(content)
        if self._use_mesh:
            with self._mesh._lock:
                tetra = self._mesh._tetrahedra.get(memory_id)
                if tetra is None:
                    return False
                old_content = tetra.content
                if content is not None and content != old_content:
                    tetra.content = content
                    tetra.centroid = self._text_to_geometry(content, labels=labels or tetra.labels)
                    self._mesh._centroid_index_dirty = True
                if labels is not None:
                    for old_label in tetra.labels:
                        self._mesh._label_index[old_label].discard(memory_id)
                    tetra.labels = list(labels)
                    for label in labels:
                        self._mesh._label_index[label].add(memory_id)
                if metadata is not None:
                    tetra.metadata.update(metadata)
                if weight is not None:
                    tetra.weight = max(0.1, min(10.0, float(weight)))
                tetra.last_access_time = time.time()
                self._invalidate_nodes_cache()
                self._eternity_audit.record_transform(
                    memory_id,
                    memory_id,
                    "update",
                    tetra.content,
                    {
                        "updated_fields": [
                            k
                            for k, v in [
                                ("content", content),
                                ("labels", labels),
                                ("metadata", metadata),
                                ("weight", weight),
                            ]
                            if v is not None
                        ]
                    },
                )
                return True
        with self._lock:
            node = self._nodes_dict.get(memory_id)
            if node is None:
                return False
            if content is not None:
                node.content = content
                node.geometry = self._text_to_geometry(content, labels=labels or node.labels)
            if labels is not None:
                node.labels = list(labels)
            if metadata is not None:
                node.metadata.update(metadata)
            if weight is not None:
                node.weight = max(0.1, min(10.0, float(weight)))
            self._invalidate_nodes_cache()
            return True

    def set_namespace(self, namespace: str) -> None:
        self._namespace = namespace

    def query_namespaced(
        self, query_text: str, k: int = 5, namespace: Optional[str] = None
    ) -> list:
        results = self.query(query_text, k=k * 3)
        ns = namespace if namespace is not None else self._namespace
        if not ns:
            return results[:k]
        filtered = []
        for r in results:
            ns_val = r.node.metadata.get("_namespace", "")
            if ns_val == ns:
                filtered.append(r)
                if len(filtered) >= k:
                    break
        return filtered

    def query_batch(self, queries: List[str], k: int = 5) -> List[list]:
        return [self.query(q, k=k) for q in queries]

    def start_emergence_daemon(self, interval: float) -> None:
        if interval <= 0:
            return
        if self._emerge_thread is not None and self._emerge_thread.is_alive():
            return
        self._auto_emerge_interval = interval
        self._emerge_stop.clear()
        self._emerge_thread = threading.Thread(
            target=self._emergence_loop, name="tetramem-emergence", daemon=True
        )
        self._emerge_thread.start()

    def stop_emergence_daemon(self) -> None:
        self._emerge_stop.set()
        if self._emerge_thread:
            self._emerge_thread.join(timeout=10.0)
            self._emerge_thread = None

    def is_emergence_running(self) -> bool:
        return self._emerge_thread is not None and self._emerge_thread.is_alive()

    def flush_persistence(self) -> None:
        if self._persistence is None:
            return
        with self._lock:
            nodes = dict(self._nodes)
        if nodes:
            self._persistence.write_incremental_full(nodes, snapshot_name=self._bucket_id)
            self._persistence.compact_snapshots(snapshot_name=self._bucket_id)

    def load_from_persistence(self) -> int:
        if self._persistence is None:
            return 0
        self._persistence.compact_snapshots(snapshot_name=self._bucket_id)
        loaded = self._persistence.load_latest_snapshot(snapshot_name=self._bucket_id)
        if not loaded:
            return 0
        count = 0
        for nid, ndata in loaded.items():
            geom = np.array(ndata.get("geometry", [0, 0, 0]))
            if self._use_mesh:
                self._mesh.store(
                    content=ndata["content"],
                    seed_point=geom,
                    labels=ndata.get("labels", []),
                    metadata=ndata.get("metadata", {}),
                    weight=ndata.get("weight", 1.0),
                )
                self._mesh_node_map[nid] = geom
            else:
                node = MemoryNode(
                    id=nid,
                    content=ndata["content"],
                    geometry=geom,
                    timestamp=ndata.get("timestamp", 0.0),
                    weight=ndata.get("weight", 1.0),
                    labels=ndata.get("labels", []),
                    metadata=ndata.get("metadata", {}),
                )
                self._nodes_dict[nid] = node
                for label in node.labels:
                    self._label_index_legacy[label].add(nid)
            count += 1
        self._needs_rebuild = True
        return count

    def get_consistency_status(self) -> Dict[str, Any]:
        if self._consistency is None:
            return {"enabled": False}
        return {
            "enabled": True,
            "bucket_id": self._bucket_id,
            "conflicts": len(self._consistency.detect_conflicts()),
            "staleness": len(self._consistency.get_staleness(self._bucket_id)),
            "pending_compensations": len(self._consistency.compensation_log.get_pending()),
        }

    def _record_version(self, node_id: str, content: str, operation: str = "store") -> None:
        if self._consistency is not None:
            try:
                self._consistency.record_version(
                    node_id, self._bucket_id, content, operation=operation
                )
            except Exception as e:
                logger.warning("Version recording failed for %s: %s", node_id, e)

    def _auto_persist(self, node_id: str, content: str) -> None:
        self._persist_dirty_count += 1
        if (
            self._persistence is not None
            and self._persist_dirty_count >= self._persist_flush_interval
        ):
            self._persist_dirty_count = 0
            try:
                self.flush_persistence()
            except Exception as e:
                logger.warning("Auto-persist failed: %s", e)

    def _persist_zigzag_snapshot(self) -> None:
        if not self._use_mesh or not self._zigzag_tracker._snapshots:
            return
        try:
            snap = self._zigzag_tracker._snapshots[-1]
            self._mesh.store(
                content=f"[meta:zigzag] entropy={snap.total_entropy:.4f} h0={len(snap.h0_barcodes)} h1={len(snap.h1_barcodes)} h2={len(snap.h2_barcodes)}",
                seed_point=np.random.randn(3) * 0.005,
                labels=["__system__", "__zigzag_snapshot__"],
                metadata={
                    "type": "zigzag_snapshot",
                    "total_entropy": snap.total_entropy,
                    "h0_count": len(snap.h0_barcodes),
                    "h1_count": len(snap.h1_barcodes),
                    "h2_count": len(snap.h2_barcodes),
                    "h0_entropy": snap.h0_entropy,
                    "h1_entropy": snap.h1_entropy,
                    "h2_entropy": snap.h2_entropy,
                    "tetra_count": snap.tetra_count,
                    "vertex_count": snap.vertex_count,
                },
                weight=0.05,
            )
        except Exception as e:
            logger.warning("Zigzag snapshot persistence failed: %s", e)

    def _persist_emergence_state(
        self,
        entropy_before: float,
        entropy_after: float,
        pressure: float,
        threshold_record: Dict[str, Any],
    ) -> None:
        if not self._use_mesh:
            return
        try:
            at_status = self._adaptive_threshold.get_status()
            self._mesh.store(
                content=f"[meta:emergence_state] entropy {entropy_before:.4f}->{entropy_after:.4f} pressure={pressure:.3f} threshold={at_status['value']:.3f}",
                seed_point=np.random.randn(3) * 0.005,
                labels=["__system__", "__emergence_state__"],
                metadata={
                    "type": "emergence_state_persistence",
                    "entropy_before": entropy_before,
                    "entropy_after": entropy_after,
                    "entropy_delta": (entropy_before - entropy_after) / entropy_before
                    if entropy_before > 1e-6
                    else 0.0,
                    "pressure": pressure,
                    "adaptive_threshold": at_status,
                    "threshold_record": threshold_record,
                },
                weight=0.05,
            )
        except Exception as e:
            logger.warning("Emergence state persistence failed: %s", e)

    def _emergence_loop(self) -> None:
        import logging
        from .emergence import EmergencePressure, AdaptiveThreshold
        from .persistent_entropy import compute_persistent_entropy, compute_entropy_delta

        logger = logging.getLogger("tetramem.emergence")
        while not self._emerge_stop.wait(timeout=self._auto_emerge_interval):
            try:
                with self._lock:
                    node_count = (
                        len(self._mesh._tetrahedra) if self._use_mesh else len(self._nodes_dict)
                    )
                if node_count < 3:
                    continue

                entropy_before = 0.0
                st = None
                if self._use_mesh:
                    with self._mesh._lock:
                        st = self._mesh.compute_ph()
                    if st is not None:
                        entropy_before = compute_persistent_entropy(st)

                pressure_report = self._emergence_pressure.compute(self._mesh, st)
                pressure = pressure_report["pressure"]
                threshold = self._adaptive_threshold.value

                if pressure < threshold:
                    continue

                if not self._protector.allow_emergence():
                    continue

                with self._lock:
                    self.self_organize()

                if self._use_mesh:
                    if st is not None:
                        from .persistent_entropy import should_trigger_integration, EntropyTracker

                        if should_trigger_integration(
                            entropy_before, self._adaptive_threshold._initial, 1.3
                        ):
                            self.global_catalyze_integration(strength=1.0)

                if not self._protector.allow_dream():
                    continue

                dream_result = self._auto_emerge_dream_with_result()

                if dream_result is not None:
                    self._protector.record_dream_success()
                else:
                    self._protector.record_dream_failure()

                self._protector.record_emergence_success()

                self._emergence_pressure.mark_integration()

                entropy_after = 0.0
                if self._use_mesh:
                    with self._mesh._lock:
                        st2 = self._mesh.compute_ph()
                    if st2 is not None:
                        entropy_after = compute_persistent_entropy(st2)
                    self._zigzag_tracker.record_snapshot(self._mesh)
                    self._pyramid.mark_dirty()

                effect = (
                    compute_entropy_delta(entropy_before, entropy_after)
                    if entropy_before > 0
                    else 0.0
                )
                threshold_record = self._adaptive_threshold.update(effect, pressure)

                if dream_result is not None:
                    self._pyramid.record_dream_feedback(
                        entropy_delta=effect,
                        dreams_created=dream_result.get("dreams_created", 0),
                        dreams_reintegrated=dream_result.get("dreams_reintegrated", 0),
                    )

                self._persist_zigzag_snapshot()
                self._persist_emergence_state(
                    entropy_before, entropy_after, pressure, threshold_record
                )

                if self._use_mesh and dream_result is not None:
                    self._mesh.store(
                        content=f"[meta:threshold_evolved] pressure={pressure:.3f} threshold={threshold:.3f} effect={effect:.3f} direction={threshold_record['direction']}",
                        seed_point=np.random.randn(3) * 0.01,
                        labels=["__system__", "__meta_dream__"],
                        metadata={
                            "type": "threshold_evolution",
                            "pressure": pressure,
                            "threshold": threshold,
                            "effect_delta": effect,
                            "direction": threshold_record["direction"],
                            "components": pressure_report["components"],
                        },
                        weight=0.1,
                    )

            except Exception as e:
                logger.error("Emergence loop error: %s", e, exc_info=True)

    def _auto_emerge_dream_with_result(self) -> Optional[Dict[str, Any]]:
        if not self._use_mesh:
            return None
        try:
            from .tetra_dream import TetraDreamCycle

            dc = TetraDreamCycle(
                self._mesh, cycle_interval=999999, zigzag_tracker=self._zigzag_tracker
            )
            return dc.trigger_now()
        except Exception as e:
            logger.warning("Auto-emerge dream failed: %s", e)
            return None

    def _auto_emerge_dream(self) -> None:
        if not self._use_mesh:
            return
        try:
            from .tetra_dream import TetraDreamCycle

            dc = TetraDreamCycle(self._mesh, cycle_interval=999999)
            dc.trigger_now()
        except Exception as e:
            logger.warning("Auto-emerge dream failed: %s", e)

    def query(self, query_text: str, k: int = 5, use_persistence: bool = True) -> List[QueryResult]:
        query_geometry = self._text_to_geometry(query_text)

        if self._use_mesh and self._mesh.tetrahedra:
            return self._query_mesh(query_geometry, query_text, k)

        with self._lock:
            if not self._nodes:
                return []

            if self._needs_rebuild:
                self._build_alpha_complex()

            self._query_count += 1
            candidate_ids = self._spatial_candidates(query_geometry, k)

            results = []
            hit_ids: List[str] = []
            for nid in candidate_ids:
                node = self._nodes.get(nid)
                if node is None:
                    continue
                distance = self._euclidean_distance(query_geometry, node.geometry)

                persistence_score = 0.0
                if use_persistence:
                    persistence_score = self._calculate_persistence_score(
                        query_geometry, node.geometry
                    )

                results.append(
                    QueryResult(
                        node=node,
                        distance=distance,
                        persistence_score=persistence_score,
                        association_type="geometric",
                    )
                )

            if len(results) < k:
                seen = {r.node.id for r in results}
                for node in self._nodes.values():
                    if node.id not in seen:
                        distance = self._euclidean_distance(query_geometry, node.geometry)
                        persistence_score = 0.0
                        if use_persistence:
                            persistence_score = self._calculate_persistence_score(
                                query_geometry, node.geometry
                            )
                        results.append(
                            QueryResult(
                                node=node,
                                distance=distance,
                                persistence_score=persistence_score,
                                association_type="geometric",
                            )
                        )
                        seen.add(node.id)

            results.sort(key=lambda r: r.distance - r.persistence_score * 0.1)
            top_results = results[:k]

            for r in top_results:
                if r.distance < 1.5:
                    hit_ids.append(r.node.id)

            if hit_ids and self._query_count % 5 == 0:
                self._adaptive_weight_boost(hit_ids, 1.15)

            return top_results

    def query_by_label(self, label: str, k: int = 10) -> List[MemoryNode]:
        with self._lock:
            node_ids = self._label_index.get(label, set())
            nodes = [self._nodes[nid] for nid in node_ids if nid in self._nodes]
            return nodes[:k]

    def associate(self, memory_id: str, max_depth: int = 2) -> List[Tuple[MemoryNode, float, str]]:
        if self._use_mesh and memory_id in self._mesh.tetrahedra:
            return self._associate_mesh(memory_id, max_depth)

        if memory_id not in self._nodes:
            return []

        with self._lock:
            if self._needs_rebuild:
                self._build_alpha_complex()

            source_node = self._nodes[memory_id]
            associations = []
            visited = {memory_id}

            direct = self._find_direct_adjacent(source_node)
            for node, score in direct:
                if node.id not in visited:
                    associations.append((node, score, "adjacency"))
                    visited.add(node.id)

            if max_depth >= 2:
                path_assoc = self._find_path_connected(source_node, visited)
                for node, score in path_assoc:
                    associations.append((node, score, "path"))
                    visited.add(node.id)

            metric_assoc = self._find_metric_proximity(source_node, visited, threshold=2.0)
            for node, score in metric_assoc:
                associations.append((node, score, "metric"))
                visited.add(node.id)

            ph_assoc = self._find_ph_patterns(source_node, visited)
            for node, score in ph_assoc:
                associations.append((node, score, "self_organizing"))
                if (
                    score > 0.3
                    and len(self._topology_shortcuts[memory_id])
                    < self._topology_shortcuts_max_per_node
                ):
                    self._topology_shortcuts[memory_id].add(node.id)
                    if (
                        len(self._topology_shortcuts[node.id])
                        < self._topology_shortcuts_max_per_node
                    ):
                        self._topology_shortcuts[node.id].add(memory_id)

            if len(self._topology_shortcuts) > self._topology_shortcuts_max_total:
                keys_by_size = sorted(
                    self._topology_shortcuts.keys(),
                    key=lambda k: len(self._topology_shortcuts[k]),
                )
                excess = len(self._topology_shortcuts) - self._topology_shortcuts_max_total // 2
                for k in keys_by_size[:excess]:
                    del self._topology_shortcuts[k]

            shortcut_assoc = self._find_shortcut_connections(source_node, visited)
            for node, score in shortcut_assoc:
                associations.append((node, score, "shortcut"))

            associations.sort(key=lambda x: x[1], reverse=True)

            return associations

    def update_weight(
        self, memory_id: str, delta: float, use_ema: bool = True, alpha: float = 0.1
    ) -> None:
        with self._lock:
            if self._use_mesh and memory_id in self._mesh.tetrahedra:
                tetra = self._mesh.get_tetrahedron(memory_id)
                if use_ema:
                    tetra.weight = alpha * delta + (1 - alpha) * tetra.weight
                else:
                    tetra.weight += delta
                tetra.weight = max(0.1, min(10.0, tetra.weight))
                self._invalidate_nodes_cache()
                return

            if memory_id not in self._nodes:
                return

            node = self._nodes[memory_id]

            if use_ema:
                node.weight = alpha * delta + (1 - alpha) * node.weight
            else:
                node.weight += delta

            node.weight = max(0.1, min(10.0, node.weight))

    def update_weights_ph_adaptive(self, ema_alpha: float = 0.1) -> None:
        """
        Update weights using PH-based adaptive boost.

        Memories that participate in persistent topological features
        get weight boosts based on their persistence.
        """
        with self._lock:
            if not self._nodes:
                return

            if self._needs_rebuild:
                self._build_alpha_complex()

            if self._simplex_tree is None:
                return

            self._simplex_tree.persistence()
            persistence_intervals = self._simplex_tree.persistence_intervals_in_dimension(0)

            if len(persistence_intervals) == 0:
                return

            avg_persistence = np.mean(
                [interval[1] - interval[0] for interval in persistence_intervals]
            )

            for node_id, node in self._nodes.items():
                idx = self._get_node_index(node_id)
                if idx is None:
                    continue

                participates = idx in self._node_filtrations

                if participates:
                    boost = min(1.0, avg_persistence * 2)
                    delta = boost * 0.1
                    node.weight = float(ema_alpha * delta + (1 - ema_alpha) * node.weight)
                    node.weight = float(max(0.1, min(10.0, node.weight)))

    def detect_conflicts(self) -> List[Tuple[str, str, float]]:
        conflicts = []

        with self._lock:
            if self._use_mesh:
                tetrahedra = self._mesh.snapshot_tetrahedra()
                centroids = []
                tids = []
                for tid, tetra in tetrahedra.items():
                    centroids.append(tetra.centroid)
                    tids.append(tid)
                if len(centroids) < 2:
                    return conflicts
                centroids = np.array(centroids)
                for i in range(len(tids)):
                    diffs = centroids[i + 1 :] - centroids[i]
                    dists = np.sqrt(np.sum(diffs**2, axis=1))
                    mask = dists < 0.1
                    for j_local in np.where(mask)[0]:
                        j = i + 1 + j_local
                        overlap = 1.0 - dists[j_local] / 0.1
                        conflicts.append((tids[i], tids[j], float(overlap)))
            else:
                nodes = list(self._nodes.values())
                for i, n1 in enumerate(nodes):
                    for n2 in nodes[i + 1 :]:
                        dist = self._euclidean_distance(n1.geometry, n2.geometry)
                        if dist < 0.1:
                            overlap = 1.0 - dist / 0.1
                            conflicts.append((n1.id, n2.id, overlap))

        return conflicts

    def self_organize(self) -> Dict[str, Any]:
        with self._lock:
            self._invalidate_nodes_cache()
            if self._use_mesh and len(self._mesh.tetrahedra) >= 3:
                return self._self_organize_mesh()

            if not GUDHI_AVAILABLE:
                return {"actions": 0, "reason": "gudhi_unavailable"}

            if not self._nodes or len(self._nodes) < 3:
                return {"actions": 0, "reason": "insufficient_nodes"}

            if self._needs_rebuild:
                self._build_alpha_complex()

            if self._simplex_tree is None:
                return {"actions": 0, "reason": "no_simplex_tree"}

            self._simplex_tree.persistence()
            h0 = self._simplex_tree.persistence_intervals_in_dimension(0)
            h1_intervals = self._simplex_tree.persistence_intervals_in_dimension(1)
            h2_intervals = self._simplex_tree.persistence_intervals_in_dimension(2)

            if len(h0) == 0 and len(h1_intervals) == 0 and len(h2_intervals) == 0:
                return {"actions": 0, "reason": "no_persistence"}

            stats = {
                "actions": 0,
                "edge_contractions": 0,
                "repulsions": 0,
                "cave_growths": 0,
                "integrations": 0,
                "nodes_affected": set(),
            }

            for interval in h0:
                birth, death = interval
                persistence = death - birth
                if persistence < self._ph_threshold * 0.5:
                    self._edge_contraction(interval, stats)
                elif persistence > self._ph_threshold * 2:
                    self._repulsion(interval, stats)

            for interval in h1_intervals:
                birth, death = interval
                if (death - birth) > self._ph_threshold * 2:
                    self._repulsion(interval, stats)

            for interval in h2_intervals:
                birth, death = interval
                if (death - birth) > self._ph_threshold * 1.5:
                    self._cave_growth(interval, stats)

            self._integrate_low_persistence(stats)

            stats["nodes_affected"] = len(stats["nodes_affected"])
            return stats

    def _self_organize_mesh(self) -> Dict[str, Any]:
        from .tetra_self_org import TetraSelfOrganizer

        if self._self_organizer is None:
            self._self_organizer = TetraSelfOrganizer(self._mesh)
        result = self._self_organizer.run()
        result["actions"] = result.get("total_actions", 0)
        return result

    def _cave_growth(self, interval: Tuple[float, float], stats: Dict[str, Any]) -> None:
        birth, death = interval
        if self._simplex_tree is None:
            return

        void_vertices: Set[int] = set()
        for simplex, filt_val in self._simplex_tree.get_filtration():
            if len(simplex) == 4 and filt_val >= birth and filt_val <= death:
                void_vertices.update(simplex)

        if len(void_vertices) < 4:
            return

        centroid = np.zeros(3)
        count = 0
        for idx in void_vertices:
            node = self._get_node_by_index(idx)
            if node:
                centroid += node.geometry
                count += 1

        if count == 0:
            return

        centroid = centroid / count
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm > 0:
            centroid = centroid / centroid_norm

        repulsion_id = (
            "_cave_"
            + hashlib.sha256((str(birth) + str(death) + str(time.time())).encode()).hexdigest()[:10]
        )
        repulsion_node = MemoryNode(
            id=repulsion_id,
            content="__cave_repulsion__",
            geometry=centroid,
            timestamp=time.time(),
            weight=8.0,
            labels=["__system__", "__cave__"],
            metadata={"type": "cave_repulsion", "birth": float(birth), "death": float(death)},
        )
        self._nodes[repulsion_id] = repulsion_node
        for label in repulsion_node.labels:
            self._label_index[label].add(repulsion_id)

        stats["cave_growths"] += 1
        stats["actions"] += 1
        stats["nodes_affected"].add(repulsion_id)
        self._needs_rebuild = True

    def _integrate_low_persistence(self, stats: Dict[str, Any]) -> None:
        if self._simplex_tree is None or len(self._nodes) < 5:
            return

        node_persistence: Dict[str, float] = {}
        for simplex, filt_val in self._simplex_tree.get_filtration():
            if len(simplex) == 2:
                node1 = self._get_node_by_index(simplex[0])
                node2 = self._get_node_by_index(simplex[1])
                p = filt_val
                if node1:
                    node_persistence[node1.id] = max(node_persistence.get(node1.id, 0), p)
                if node2:
                    node_persistence[node2.id] = max(node_persistence.get(node2.id, 0), p)

        integrate_threshold = self._ph_threshold * 0.3
        to_integrate = []
        for node_id, node in self._nodes.items():
            if node.labels and "__system__" in node.labels:
                continue
            max_p = node_persistence.get(node_id, 0)
            if max_p < integrate_threshold and node.weight < 0.3:
                to_integrate.append(node_id)

        for node_id in to_integrate:
            node = self._nodes[node_id]
            node.weight = max(node.weight, 0.5)
            node.weight = min(10.0, node.weight + 0.3)
            stats["nodes_affected"].add(node_id)

        if to_integrate:
            stats["integrations"] = len(to_integrate)
            stats["actions"] += len(to_integrate)
            self._needs_rebuild = True

    def global_catalyze_integration(self, strength: float = 1.0) -> Dict[str, Any]:
        with self._lock:
            if self._use_mesh:
                return self._global_catalyze_mesh(strength)

            from .persistent_entropy import EntropyTracker

            catalyzed = 0
            for node_id, node in self._nodes.items():
                if node.labels and "__system__" in node.labels:
                    continue
                node.weight = min(10.0, node.weight + strength * 0.05)
                catalyzed += 1

            return {"catalyzed": catalyzed, "total": len(self._nodes)}

    def _global_catalyze_mesh(self, strength: float) -> Dict[str, Any]:
        return self._mesh.catalyze_integration_batch(list(self._mesh.tetrahedra.keys()), strength)

    def _edge_contraction(self, interval: Tuple[float, float], stats: Dict[str, Any]) -> None:
        if self._simplex_tree is None:
            return
        birth, death = interval
        for simplex, filt_val in self._simplex_tree.get_filtration():
            if len(simplex) == 2 and filt_val >= birth and filt_val <= death:
                idx1, idx2 = simplex
                node1 = self._get_node_by_index(idx1)
                node2 = self._get_node_by_index(idx2)
                if node1 and node2:
                    midpoint = (node1.geometry + node2.geometry) / 2.0
                    node1.geometry = midpoint
                    node2.geometry = midpoint
                    node1.weight = (node1.weight + node2.weight) / 2.0
                    node2.weight = node1.weight
                    stats["nodes_affected"].add(node1.id)
                    stats["nodes_affected"].add(node2.id)
        stats["edge_contractions"] += 1
        stats["actions"] += 1
        self._needs_rebuild = True

    def _repulsion(self, interval: Tuple[float, float], stats: Dict[str, Any]) -> None:
        birth, death = interval

        if self._simplex_tree is None:
            return

        for simplex, filtration_value in self._simplex_tree.get_filtration():
            if len(simplex) == 2 and filtration_value >= birth and filtration_value <= death:
                idx1, idx2 = simplex
                node1 = self._get_node_by_index(idx1)
                node2 = self._get_node_by_index(idx2)

                if node1 and node2:
                    direction = node1.geometry - node2.geometry
                    norm = np.linalg.norm(direction)
                    if norm > 0:
                        direction = direction / norm
                        node1.geometry = node1.geometry + direction * 0.1
                        node2.geometry = node2.geometry - direction * 0.1

                        stats["nodes_affected"].add(node1.id)
                        stats["nodes_affected"].add(node2.id)

        stats["repulsions"] += 1
        stats["actions"] += 1
        self._needs_rebuild = True

    def get_persistence_diagram(self) -> Optional[List[Tuple[int, Tuple[float, float]]]]:
        with self._lock:
            if self._needs_rebuild:
                self._build_alpha_complex()

            if self._simplex_tree is None:
                return None

            self._simplex_tree.persistence()
            return self._simplex_tree.persistence_intervals_in_dimension(1)

    def get_statistics(self) -> Dict[str, Any]:
        with self._lock:
            stats = {
                "total_memories": len(self._nodes),
                "total_labels": len(self._label_index),
                "avg_weight": np.mean([n.weight for n in self._nodes.values()])
                if self._nodes
                else 0,
                "dimension": self.dimension,
                "precision": self.precision,
                "query_count": self._query_count,
                "topology_shortcuts": sum(len(v) for v in self._topology_shortcuts.values()) // 2,
            }
            if self._use_mesh:
                stats["zigzag"] = self._zigzag_tracker.get_status()
                stats["pyramid"] = self._pyramid.get_status()
            return stats

    def record_zigzag_snapshot(self) -> Optional[Dict[str, Any]]:
        if not self._use_mesh:
            return None
        with self._lock:
            snap = self._zigzag_tracker.record_snapshot(self._mesh)
            self._pyramid.mark_dirty()
        return {
            "total_entropy": snap.total_entropy,
            "h0_count": len(snap.h0_barcodes),
            "h1_count": len(snap.h1_barcodes),
            "h2_count": len(snap.h2_barcodes),
            "tetra_count": snap.tetra_count,
            "transitions": len(self._zigzag_tracker.detect_phase_transitions()),
        }

    def get_zigzag_status(self) -> Dict[str, Any]:
        return self._zigzag_tracker.get_status()

    def predict_topology(self) -> Dict[str, Any]:
        return self._zigzag_tracker.predict_emerging_features()

    def get_dynamic_barcode(self, dimension: int = -1) -> Dict[str, Any]:
        return self._zigzag_tracker.get_dynamic_barcode(dimension)

    def build_pyramid(self) -> Dict[str, Any]:
        if not self._use_mesh:
            return {"levels": 0, "reason": "not_mesh_mode"}
        with self._lock:
            return self._pyramid.build(self._mesh)

    def query_pyramid(self, query_text: str, k: int = 5, level: int = -1) -> List[QueryResult]:
        if not self._use_mesh:
            return []
        query_geometry = self._text_to_geometry(query_text)
        with self._lock:
            self._pyramid.ensure_built(self._mesh)
            results = self._pyramid.query(query_geometry, k=k, level=level)
            effective_level = (
                self._pyramid._last_query_level
                if hasattr(self._pyramid, "_last_query_level")
                else 0
            )
            self._pyramid.record_query_feedback(effective_level, len(results) > 0)
        out = []
        for tid, score in results:
            tetra = self._mesh.get_tetrahedron(tid)
            if tetra is None:
                continue
            node = MemoryNode(
                id=tetra.id,
                content=tetra.content,
                geometry=tetra.centroid,
                timestamp=tetra.creation_time,
                weight=tetra.weight,
                labels=list(tetra.labels),
                metadata=dict(tetra.metadata),
            )
            out.append(
                QueryResult(
                    node=node,
                    distance=score,
                    persistence_score=0.0,
                    association_type="pyramid",
                )
            )
        return out

    def query_multiparam(
        self,
        query_text: str,
        k: int = 10,
        spatial_weight: float = 0.4,
        temporal_weight: float = 0.2,
        density_weight: float = 0.15,
        weight_weight: float = 0.15,
        topology_weight: float = 0.1,
        recency_seconds: float = 3600.0,
        max_distance: float = 5.0,
        labels_required: Optional[List[str]] = None,
        labels_preferred: Optional[List[str]] = None,
    ) -> List[QueryResult]:
        if not self._use_mesh:
            return []
        query_geometry = self._text_to_geometry(query_text)
        from .multiparameter_filter import MultiParameterQuery

        mpq = MultiParameterQuery(self._mesh)
        mpq.add_filter(
            "spatial",
            {
                "query_point": query_geometry,
                "max_distance": max_distance,
            },
            weight=spatial_weight,
        )
        mpq.add_filter(
            "temporal",
            {
                "recency_seconds": recency_seconds,
                "mode": "creation",
            },
            weight=temporal_weight,
        )
        mpq.add_filter(
            "density",
            {
                "neighbor_radius": 1.0,
            },
            weight=density_weight,
        )
        mpq.add_filter(
            "weight",
            {
                "min_weight": 0.1,
            },
            weight=weight_weight,
        )
        mpq.add_filter(
            "topology",
            {
                "integration_boost": True,
                "connectivity_weight": 0.3,
            },
            weight=topology_weight,
        )
        label_params: Dict[str, Any] = {}
        if labels_required:
            label_params["required"] = labels_required
        if labels_preferred:
            label_params["preferred"] = labels_preferred
        if label_params:
            mpq.add_filter(
                "label", label_params, weight=0.3, hard_filter=bool(labels_required), min_score=0.01
            )

        results = mpq.execute(k=k)
        out = []
        for mr in results:
            node = MemoryNode(
                id=mr.tetra_id,
                content=mr.content,
                geometry=mr.centroid if mr.centroid is not None else np.zeros(3),
                timestamp=0.0,
                weight=mr.weight,
                labels=mr.labels,
                metadata=mr.metadata,
            )
            out.append(
                QueryResult(
                    node=node,
                    distance=0.0,
                    persistence_score=mr.composite_score,
                    association_type="multiparam",
                )
            )
        return out

    def clear(self) -> None:
        with self._lock:
            if self._use_mesh:
                self._mesh = TetraMesh(time_lambda=0.001)
                self._mesh_node_map.clear()
            self._nodes_dict.clear()
            self._label_index_legacy.clear()
            self._alpha_complex = None
            self._simplex_tree = None
            self._geometry_cache = None
            self._weights = None
            self._needs_rebuild = True
            self._invalidate_nodes_cache()

    def verify_eternity(self) -> Dict[str, Any]:
        return self._eternity_audit.verify(self._mesh)

    def get_eternity_status(self) -> Dict[str, Any]:
        return self._eternity_audit.get_status()

    def get_eternity_trail(self, tetra_id: str) -> List[Dict[str, Any]]:
        return self._eternity_audit.get_audit_trail(tetra_id)

    def get_mapping_cone_history(self, count: int = 10) -> List[Dict[str, Any]]:
        return self._zigzag_tracker.get_mapping_cone_history(count)

    def get_dream_guidance(self) -> Dict[str, Any]:
        return self._zigzag_tracker.get_dream_guidance()

    def get_pyramid_adaptive_status(self) -> Dict[str, Any]:
        return self._pyramid.get_adaptive_status()

    def _query_mesh(self, query_geometry: np.ndarray, query_text: str, k: int) -> List[QueryResult]:
        topo_results = self._mesh.query_topological(query_geometry, k=k)
        results = []
        for tid, score in topo_results:
            tetra = self._mesh.get_tetrahedron(tid)
            if tetra is None:
                continue
            node = MemoryNode(
                id=tetra.id,
                content=tetra.content,
                geometry=tetra.centroid,
                timestamp=tetra.creation_time,
                weight=tetra.weight,
                labels=list(tetra.labels),
                metadata=dict(tetra.metadata),
            )
            results.append(
                QueryResult(
                    node=node,
                    distance=score,
                    persistence_score=0.0,
                    association_type="topological",
                )
            )
        return results

    def _associate_mesh(self, tetra_id: str, max_depth: int) -> List[Tuple[MemoryNode, float, str]]:
        topo_assoc = self._mesh.associate_topological(tetra_id, max_depth)
        results = []
        for tid, score, conn_type in topo_assoc:
            tetra = self._mesh.get_tetrahedron(tid)
            if tetra is None:
                continue
            node = MemoryNode(
                id=tetra.id,
                content=tetra.content,
                geometry=tetra.centroid,
                timestamp=tetra.creation_time,
                weight=tetra.weight,
                labels=list(tetra.labels),
                metadata=dict(tetra.metadata),
            )
            results.append((node, score, conn_type))
        return results

    def store_batch(self, items: List[Dict[str, Any]]) -> List[str]:
        if self._use_mesh:
            ids = []
            with self._lock:
                for item in items:
                    content = item["content"]
                    labels = item.get("labels")
                    metadata = item.get("metadata")
                    weight = item.get("weight", 1.0)
                    geometry = self._text_to_geometry(content, labels=labels)
                    tetra_id = self._mesh.store(
                        content=content,
                        seed_point=geometry,
                        labels=labels,
                        metadata=metadata,
                        weight=weight,
                    )
                    self._mesh_node_map[tetra_id] = geometry
                    self._record_version(tetra_id, content)
                    self._auto_persist(tetra_id, content)
                    self._eternity_audit.record_store(tetra_id, content, metadata)
                    ids.append(tetra_id)
            return ids

        ids = []
        with self._lock:
            for item in items:
                content = item["content"]
                labels = item.get("labels")
                metadata = item.get("metadata")
                weight = item.get("weight", 1.0)
                node_id = self._generate_id(content)
                geometry = self._text_to_geometry(content, labels=labels)
                node = MemoryNode(
                    id=node_id,
                    content=content,
                    geometry=geometry,
                    timestamp=time.time(),
                    weight=weight,
                    labels=labels or [],
                    metadata=metadata or {},
                )
                self._nodes_dict[node_id] = node
                for label in node.labels:
                    self._label_index_legacy[label].add(node_id)
                self._dirty_nodes.add(node_id)
                self._update_octree_insert(node_id, node)
                self._record_version(node_id, content)
                self._auto_persist(node_id, content)
                self._eternity_audit.record_store(node_id, content, metadata)
                ids.append(node_id)
            self._rebuild_index_cache()
            self._needs_rebuild = True
        return ids

    def _spatial_candidates(self, query_geometry: np.ndarray, k: int) -> List[str]:
        if self._octree is not None and self._octree_initialized:
            nearest = self._octree.query_nearest(query_geometry, k=k * 3)
            if len(nearest) >= k:
                return [nid for nid, _ in nearest]
        return list(self._nodes.keys())

    def _update_octree_insert(self, node_id: str, node: "MemoryNode") -> None:
        if self._octree is None:
            if len(self._nodes) >= 8:
                self._init_octree()
            return
        try:
            self._octree.insert(node.geometry, node_id)
        except Exception as e:
            self._octree_initialized = False
            logger.debug("Octree insert failed: %s", e)

    def _init_octree(self) -> None:
        if not self._nodes:
            return
        geoms = np.array([n.geometry for n in self._nodes.values()])
        mins = np.min(geoms, axis=0) - 0.5
        maxs = np.max(geoms, axis=0) + 0.5
        bounds = BoundingBox(mins, maxs)
        self._octree = Octree(bounds, max_points=16, max_depth=12)
        for nid, node in self._nodes.items():
            try:
                self._octree.insert(node.geometry, nid)
            except Exception as e:
                logger.debug("Octree init insert failed for %s: %s", nid, e)
        self._octree_initialized = True

    def _generate_id(self, content: str) -> str:
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _text_to_geometry(self, text: str, labels: Optional[List[str]] = None) -> np.ndarray:
        from .geometry import SemanticEmbedder, _NGramFingerprint

        semantic = SemanticEmbedder.embed(text)
        if semantic is not None:
            direction = semantic
        else:
            direction = _NGramFingerprint.fingerprint(text)

        if labels:
            sector = self._get_label_sector(labels)
            attraction = 0.55
            direction = direction * (1.0 - attraction) + sector * attraction

        return direction.astype(np.float64)

    def _get_label_sector(self, labels: List[str]) -> np.ndarray:
        if not labels:
            return np.array([1.0, 0.0, 0.0])
        primary = labels[0]
        if primary not in self._label_sector_cache:
            if len(self._label_sector_cache) > self._label_sector_cache_max:
                oldest = list(self._label_sector_cache.keys())[: self._label_sector_cache_max // 2]
                for k in oldest:
                    del self._label_sector_cache[k]
            h = int(hashlib.sha256(primary.encode()).hexdigest()[:8], 16)
            rng = np.random.RandomState(h % (2**31))
            theta = rng.uniform(0, 2 * np.pi)
            phi = np.arccos(2 * rng.uniform(0, 1) - 1)
            self._label_sector_cache[primary] = np.array(
                [
                    np.sin(phi) * np.cos(theta),
                    np.sin(phi) * np.sin(theta),
                    np.cos(phi),
                ]
            )
        return self._label_sector_cache[primary]

    @staticmethod
    def _power_distance(p1: np.ndarray, p2: np.ndarray, w1: float, w2: float) -> float:
        euclidean_sq = float(np.sum((p1 - p2) ** 2))
        return float(np.sqrt(max(0.0, euclidean_sq - w1 - w2)))

    def _euclidean_distance(self, p1: np.ndarray, p2: np.ndarray) -> float:
        return float(np.linalg.norm(p1 - p2))

    def _build_alpha_complex(self) -> None:
        if not GUDHI_AVAILABLE or gudhi is None:
            return

        if not self._nodes:
            return

        points = np.array([node.geometry for node in self._nodes.values()])
        weights = np.array([node.weight for node in self._nodes.values()])

        if gudhi is None:
            return

        self._alpha_complex = gudhi.AlphaComplex(
            points=points.tolist(), weights=weights.tolist(), precision=self.precision
        )
        self._simplex_tree = self._alpha_complex.create_simplex_tree()
        self._simplex_tree.compute_persistence(homology_coeff_field=2, min_persistence=0.01)
        self._weights = weights
        self._rebuild_index_cache()
        self._build_simplex_index()

        self._refresh_persistence_cache()
        self._needs_rebuild = False
        self._dirty_nodes.clear()

    def _calculate_persistence_score(self, query_geom: np.ndarray, node_geom: np.ndarray) -> float:
        if self._simplex_tree is None:
            return 0.0

        dist = self._euclidean_distance(query_geom, node_geom)
        base_score = 1.0 / (1.0 + dist)

        ph_weight = min(1.0, self._cached_avg_persistence * 2)
        max_w = max(self._cached_max_weight, 0.1)

        return float(base_score * (0.5 + 0.3 * ph_weight + 0.2 * (1.0 / max_w)))

    def _refresh_persistence_cache(self) -> None:
        if self._simplex_tree is None:
            return
        self._simplex_tree.persistence()
        intervals = self._simplex_tree.persistence_intervals_in_dimension(0)
        if len(intervals) > 0:
            self._cached_avg_persistence = float(np.mean([iv[1] - iv[0] for iv in intervals]))
        else:
            self._cached_avg_persistence = 0.0
        if self._weights is not None and len(self._weights) > 0:
            self._cached_max_weight = float(np.max(self._weights))
        else:
            self._cached_max_weight = 1.0

    def _adaptive_weight_boost(self, hit_ids: List[str], boost_factor: float = 1.15) -> None:
        if self._simplex_tree is None:
            return
        persistence_weights = {}
        for pair in self._simplex_tree.persistence_pairs():
            if len(pair[0]) >= 2:
                persistence = pair[1][1] - pair[1][0]
                for idx in pair[0]:
                    if idx < len(self._idx_to_id):
                        nid = self._idx_to_id.get(idx)
                        if nid:
                            persistence_weights[nid] = max(
                                persistence_weights.get(nid, 0), persistence
                            )
        for nid in hit_ids:
            if nid in self._nodes:
                pw = persistence_weights.get(nid, 0)
                adaptive = boost_factor * (1 + 0.3 * pw)
                node = self._nodes[nid]
                node.weight = min(10.0, node.weight * adaptive)
                node.weight = max(0.1, node.weight)

    def _find_direct_adjacent(self, source: MemoryNode) -> List[Tuple[MemoryNode, float]]:
        adjacent = []
        source_idx = self._get_node_index(source.id)

        if source_idx is None or self._simplex_tree is None:
            return adjacent

        for simplex, _ in self._simplex_tree.get_filtration():
            if source_idx in simplex:
                for idx in simplex:
                    if idx != source_idx and idx < len(self._idx_to_id):
                        nid = self._idx_to_id[idx]
                        node = self._nodes.get(nid) if nid else None
                        if node:
                            adjacent.append((node, 1.0))

        return adjacent

    def _build_node_tetrahedron(self, node: MemoryNode) -> np.ndarray:
        geom = node.geometry
        scale = 0.1 * node.weight
        perp = np.array([1.0, 0.0, 0.0]) if abs(geom[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        d1 = np.cross(geom, perp)
        d1 = d1 / (np.linalg.norm(d1) + 1e-12) * scale
        d2 = np.cross(geom, d1)
        d2 = d2 / (np.linalg.norm(d2) + 1e-12) * scale
        return np.array([geom, geom + d1, geom + d2, geom - d1 - d2])

    def _find_path_connected(
        self, source: MemoryNode, visited: Set[str]
    ) -> List[Tuple[MemoryNode, float]]:
        if self._simplex_tree is None:
            return []

        edge_costs: Dict[Tuple[int, int], float] = {}
        for simplex, filt_val in self._simplex_tree.get_filtration():
            if len(simplex) == 2:
                i, j = simplex
                cost = max(filt_val, 1e-6)
                key = (min(i, j), max(i, j))
                edge_costs[key] = min(edge_costs.get(key, float("inf")), cost)

        adj: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
        for (i, j), cost in edge_costs.items():
            adj[i].append((j, cost))
            adj[j].append((i, cost))

        source_idx = self._get_node_index(source.id)
        if source_idx is None:
            return []

        dist_map: Dict[int, float] = {source_idx: 0.0}
        import heapq as _heapq

        heap: List[Tuple[float, int]] = [(0.0, source_idx)]
        while heap:
            d, u = _heapq.heappop(heap)
            if d > dist_map.get(u, float("inf")):
                continue
            for v, w in adj.get(u, []):
                nd = d + w
                if nd < dist_map.get(v, float("inf")):
                    dist_map[v] = nd
                    _heapq.heappush(heap, (nd, v))

        results = []
        max_dist = max(dist_map.values()) if dist_map else 1.0
        if max_dist == 0:
            max_dist = 1.0
        for idx, d in dist_map.items():
            node = self._get_node_by_index(idx)
            if node and node.id not in visited and node.id != source.id:
                score = 1.0 - d / max_dist
                results.append((node, score))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:10]

    def _find_metric_proximity(
        self, source: MemoryNode, visited: Set[str], threshold: float
    ) -> List[Tuple[MemoryNode, float]]:
        proximal = []
        try:
            from tetrahedron_memory.geometry import GeometryPrimitives

            use_jaccard = True
        except ImportError:
            use_jaccard = False

        source_tet = self._build_node_tetrahedron(source) if use_jaccard else None

        for node in self._nodes.values():
            if node.id in visited:
                continue

            dist = self._euclidean_distance(source.geometry, node.geometry)
            if dist >= threshold:
                continue

            euclidean_score = 1.0 - dist / threshold

            weight_diff = abs(source.weight - node.weight)
            use_power = weight_diff > 0.5

            if use_jaccard and source_tet is not None:
                target_tet = self._build_node_tetrahedron(node)
                jac = GeometryPrimitives.jaccard_index(source_tet, target_tet)
                vol_s = GeometryPrimitives.tetrahedron_volume(source_tet)
                vol_t = GeometryPrimitives.tetrahedron_volume(target_tet)
                vol_ratio = min(vol_s, vol_t) / (max(vol_s, vol_t) + 1e-12)
                if use_power:
                    pd = self._power_distance(
                        source.geometry, node.geometry, source.weight, node.weight
                    )
                    power_score = 1.0 - min(pd / threshold, 1.0)
                    score = 0.5 * power_score + 0.3 * jac + 0.2 * vol_ratio
                else:
                    score = 0.5 * euclidean_score + 0.3 * jac + 0.2 * vol_ratio
            else:
                score = euclidean_score

            proximal.append((node, score))

        return proximal

    def _find_ph_patterns(
        self, source: MemoryNode, visited: Set[str]
    ) -> List[Tuple[MemoryNode, float]]:
        if self._simplex_tree is None:
            return []

        self._simplex_tree.persistence()
        source_idx = self._get_node_index(source.id)

        source_intervals = (
            self._node_filtrations.get(source_idx, []) if source_idx is not None else []
        )

        patterns = []

        if not source_intervals:
            for node in self._nodes.values():
                if node.id not in visited:
                    weight_diff = abs(node.weight - source.weight)
                    if weight_diff < 0.3:
                        patterns.append((node, (1.0 - weight_diff / 0.3) * 0.3))
            return patterns

        s_min, s_max = min(source_intervals), max(source_intervals)

        for node in self._nodes.values():
            if node.id in visited:
                continue
            node_idx = self._get_node_index(node.id)
            if node_idx is None:
                continue

            node_intervals = self._node_filtrations.get(node_idx, [])

            if not node_intervals:
                continue

            n_min, n_max = min(node_intervals), max(node_intervals)
            overlap_low = max(s_min, n_min)
            overlap_high = min(s_max, n_max)
            overlap = max(0, overlap_high - overlap_low)
            union = max(s_max, n_max) - min(s_min, n_min)
            iou = overlap / union if union > 0 else 0.0

            if iou > 0.1:
                patterns.append((node, iou * 0.5))

        patterns.sort(key=lambda x: x[1], reverse=True)
        return patterns[:10]

    def _find_shortcut_connections(
        self, source: MemoryNode, visited: Set[str]
    ) -> List[Tuple[MemoryNode, float]]:
        shortcut_ids = self._topology_shortcuts.get(source.id, set())
        results = []
        for nid in shortcut_ids:
            if nid in self._nodes and nid not in visited:
                node = self._nodes[nid]
                dist = self._euclidean_distance(source.geometry, node.geometry)
                score = max(0.1, 1.0 / (1.0 + dist))
                results.append((node, score))
        return results

    def _rebuild_index_cache(self) -> None:
        self._id_to_idx = {}
        self._idx_to_id = []
        for idx, node_id in enumerate(self._nodes.keys()):
            self._id_to_idx[node_id] = idx
            self._idx_to_id.append(node_id)

    def _build_simplex_index(self) -> None:
        self._node_filtrations: Dict[int, List[float]] = {}
        if self._simplex_tree is None:
            return
        for simplex, filt_val in self._simplex_tree.get_filtration():
            for idx in simplex:
                if idx not in self._node_filtrations:
                    self._node_filtrations[idx] = []
                self._node_filtrations[idx].append(filt_val)
        for idx in self._node_filtrations:
            self._node_filtrations[idx].sort()

    def _get_node_index(self, node_id: str) -> Optional[int]:
        return self._id_to_idx.get(node_id)

    def _get_node_by_index(self, idx: int) -> Optional[MemoryNode]:
        if idx < 0 or idx >= len(self._idx_to_id):
            return None
        node_id = self._idx_to_id[idx]
        return self._nodes.get(node_id) if node_id else None
