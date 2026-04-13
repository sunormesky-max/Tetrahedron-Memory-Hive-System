"""
Topology-driven self-organization module for TetraMem-XL.

Monitors PH (Persistent Homology) changes and triggers structural
modifications: edge contractions (H0), node repulsions (H1),
cave growths (H2), and global integration catalyst.

Also contains DreamCycle — a background daemon that performs
random walks and semantic synthesis to discover hidden connections.

Per TetraMem-XL core principles: NO deletion/decay. All memories
are eternal. Noise is transformed through integration, never removed.
"""

import logging
import random
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

try:
    import gudhi

    GUDHI_AVAILABLE = True
except ImportError:
    GUDHI_AVAILABLE = False

logger = logging.getLogger("tetramem.topology")


class TopologySelfOrganizer:
    def __init__(
        self,
        engine: Any,
        h0_threshold: float = 0.25,
        h1_threshold: float = 0.5,
        h2_threshold: float = 0.4,
        integration_strength: float = 1.0,
        check_interval: float = 60.0,
    ):
        self.engine = engine
        self.h0_threshold = h0_threshold
        self.h1_threshold = h1_threshold
        self.h2_threshold = h2_threshold
        self.integration_strength = integration_strength
        self.check_interval = check_interval
        self._last_diag: Optional[List[Tuple[int, Tuple[float, float]]]] = None
        self._last_check_time: float = 0.0
        self._total_actions = 0

    def check_and_trigger(self, force: bool = False) -> Dict[str, Any]:
        now = time.time()
        if not force and (now - self._last_check_time) < self.check_interval:
            return {"actions": 0, "reason": "too_soon"}

        self._last_check_time = now
        stats: Dict[str, Any] = {
            "actions": 0,
            "edge_contractions": 0,
            "repulsions": 0,
            "cave_growths": 0,
            "integrations": 0,
        }

        with self.engine._lock:
            if len(self.engine._nodes) < 4:
                return {"actions": 0, "reason": "insufficient_nodes"}

            if self.engine._needs_rebuild:
                self.engine._build_alpha_complex()

            st = self.engine._simplex_tree
            if st is None:
                return {"actions": 0, "reason": "no_simplex_tree"}

            st.compute_persistence(homology_coeff_field=2, min_persistence=0.01)
            current_diag = st.persistence()

            h0_intervals = st.persistence_intervals_in_dimension(0)
            h1_intervals = st.persistence_intervals_in_dimension(1)
            h2_intervals = st.persistence_intervals_in_dimension(2)

            if self._last_diag is None:
                self._last_diag = current_diag
                return {"actions": 0, "reason": "first_run_baseline"}

            for birth, death in h0_intervals:
                persistence = death - birth
                if persistence < self.h0_threshold:
                    self._apply_edge_contraction(birth, death, stats)
                elif persistence > self.h1_threshold:
                    self._apply_repulsion(birth, death, stats)

            for birth, death in h1_intervals:
                persistence = death - birth
                if persistence > self.h1_threshold:
                    self._apply_repulsion(birth, death, stats)

            for birth, death in h2_intervals:
                persistence = death - birth
                if persistence > self.h2_threshold:
                    self._apply_cave_growth(birth, death, stats)

            if self.engine._nodes:
                integration_result = self.engine.global_catalyze_integration(
                    strength=1.0
                )
                stats["integrations"] = integration_result.get("catalyzed", 0)
                stats["actions"] += integration_result.get("catalyzed", 0)

            self._last_diag = current_diag
            self._total_actions += stats["actions"]

        return stats

    def _apply_edge_contraction(
        self, birth: float, death: float, stats: Dict[str, Any]
    ) -> None:
        st = self.engine._simplex_tree
        if st is None:
            return

        for simplex, filt_val in st.get_filtration():
            if len(simplex) == 2 and birth <= filt_val <= death:
                idx1, idx2 = simplex
                node1 = self.engine._get_node_by_index(idx1)
                node2 = self.engine._get_node_by_index(idx2)
                if node1 and node2 and node1.id != node2.id:
                    if "__system__" in node1.labels or "__system__" in node2.labels:
                        continue
                    midpoint = (node1.geometry + node2.geometry) / 2.0
                    node1.geometry = midpoint
                    node2.geometry = midpoint
                    avg_weight = (node1.weight + node2.weight) / 2.0
                    node1.weight = avg_weight
                    node2.weight = avg_weight
                    self.engine._needs_rebuild = True
                    stats["edge_contractions"] += 1
                    stats["actions"] += 1

    def _apply_repulsion(
        self, birth: float, death: float, stats: Dict[str, Any]
    ) -> None:
        st = self.engine._simplex_tree
        if st is None:
            return

        repelled_pairs: set = set()
        for simplex, filt_val in st.get_filtration():
            if len(simplex) == 2 and birth <= filt_val <= death:
                idx1, idx2 = simplex
                pair_key = (min(idx1, idx2), max(idx1, idx2))
                if pair_key in repelled_pairs:
                    continue
                repelled_pairs.add(pair_key)

                node1 = self.engine._get_node_by_index(idx1)
                node2 = self.engine._get_node_by_index(idx2)
                if node1 and node2:
                    direction = node1.geometry - node2.geometry
                    norm = np.linalg.norm(direction)
                    if norm > 1e-8:
                        direction = direction / norm
                        strength = 0.05 * (death - birth)
                        node1.geometry = node1.geometry + direction * strength
                        node2.geometry = node2.geometry - direction * strength
                        self.engine._needs_rebuild = True
                        stats["repulsions"] += 1
                        stats["actions"] += 1

    def _apply_cave_growth(
        self, birth: float, death: float, stats: Dict[str, Any]
    ) -> None:
        self.engine._cave_growth((birth, death), stats)

    def get_status(self) -> Dict[str, Any]:
        return {
            "total_actions": self._total_actions,
            "last_check_time": self._last_check_time,
            "check_interval": self.check_interval,
            "h0_threshold": self.h0_threshold,
            "h1_threshold": self.h1_threshold,
            "h2_threshold": self.h2_threshold,
        }


class DreamCycle:
    """
    Background dream cycle for TetraMem-XL.

    Periodically performs random walks through the memory graph,
    discovers hidden semantic connections, and synthesizes new
    'dream' memories that bridge related clusters.

    The dream cycle operates in three phases per cycle:
    1. Random Walk — traverse the association graph from a seed node
    2. Cluster Detection — group visited nodes by geometry proximity
    3. Synthesis — create lightweight bridge nodes connecting clusters

    Dream nodes are tagged with '__dream__' label and have lower
    initial weights. They get reintegrated (weight boost) rather
    than pruned — all memories are eternal.
    """

    def __init__(
        self,
        engine: Any,
        organizer: Optional[TopologySelfOrganizer] = None,
        cycle_interval: float = 300.0,
        walk_steps: int = 15,
        walk_temperature: float = 0.7,
        cluster_eps: float = 1.5,
        min_cluster_size: int = 2,
        dream_weight: float = 0.3,
        dream_min_weight: float = 0.08,
        dream_decay_factor: float = 0.85,
        max_dream_nodes: int = 50,
        synthesis_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        self.engine = engine
        self.organizer = organizer
        self.cycle_interval = cycle_interval
        self.walk_steps = walk_steps
        self.walk_temperature = walk_temperature
        self.cluster_eps = cluster_eps
        self.min_cluster_size = min_cluster_size
        self.dream_weight = dream_weight
        self.dream_min_weight = dream_min_weight
        self.dream_decay_factor = dream_decay_factor
        self.max_dream_nodes = max_dream_nodes
        self.synthesis_callback = synthesis_callback

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._cycle_count = 0
        self._dream_nodes_created = 0
        self._dream_nodes_reintegrated = 0
        self._last_cycle_time: float = 0.0
        self._last_cycle_stats: Dict[str, Any] = {}

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            logger.warning("DreamCycle already running")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop, name="tetramem-dream", daemon=True
        )
        self._thread.start()
        logger.info(
            "DreamCycle started (interval=%.0fs, walk_steps=%d, max_dreams=%d)",
            self.cycle_interval, self.walk_steps, self.max_dream_nodes,
        )

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=10.0)
            self._thread = None
        logger.info("DreamCycle stopped")

    def trigger_now(self) -> Dict[str, Any]:
        stats = self._execute_cycle()
        self._last_cycle_stats = stats
        return stats

    def get_status(self) -> Dict[str, Any]:
        return {
            "running": self._thread is not None and self._thread.is_alive(),
            "cycle_count": self._cycle_count,
            "dream_nodes_created": self._dream_nodes_created,
            "dream_nodes_reintegrated": self._dream_nodes_reintegrated,
            "last_cycle_time": self._last_cycle_time,
            "last_cycle_stats": self._last_cycle_stats,
        }

    def _run_loop(self) -> None:
        while not self._stop_event.wait(timeout=self.cycle_interval):
            try:
                stats = self._execute_cycle()
                self._last_cycle_stats = stats
                self._last_cycle_time = time.time()
                self._cycle_count += 1
                logger.debug("DreamCycle #%d complete: %s", self._cycle_count, stats)
            except Exception as e:
                logger.error("DreamCycle error: %s", e, exc_info=True)

    def _execute_cycle(self) -> Dict[str, Any]:
        stats: Dict[str, Any] = {
            "phase": "idle",
            "walk_visited": 0,
            "clusters_found": 0,
            "dreams_created": 0,
            "dreams_reintegrated": 0,
            "syntheses": 0,
        }

        with self.engine._lock:
            node_ids = list(self.engine._nodes.keys())
            if len(node_ids) < 4:
                stats["phase"] = "skipped_insufficient_nodes"
                return stats

            regular_ids = [
                nid for nid in node_ids
                if "__dream__" not in self.engine._nodes[nid].labels
            ]
            if not regular_ids:
                stats["phase"] = "skipped_no_regular_nodes"
                return stats

        all_visited: Dict[str, float] = {}
        all_walk_edges: List[Tuple[str, str, float]] = []
        num_walks = min(3, max(1, len(regular_ids) // 8))
        used_seeds: Set[str] = set()

        for _ in range(num_walks):
            remaining = [nid for nid in regular_ids if nid not in used_seeds]
            if not remaining:
                break
            visited, edges = self._random_walk(remaining)
            all_visited.update(visited)
            all_walk_edges.extend(edges)
            used_seeds.update(list(visited.keys())[:1])

        stats["walk_visited"] = len(all_visited)

        if len(all_visited) < self.min_cluster_size:
            stats["phase"] = "walk_too_short"
            return stats

        clusters = self._cluster_label_aware(all_visited)
        stats["clusters_found"] = len(clusters)

        if len(clusters) < 2:
            stats["phase"] = "no_cross_cluster_connections"
            return stats

        syntheses = self._synthesize_bridges(clusters, all_walk_edges)
        stats["syntheses"] = syntheses
        stats["dreams_created"] = syntheses
        self._dream_nodes_created += syntheses

        reintegrated = self._reintegrate_dreams()
        stats["dreams_reintegrated"] = reintegrated
        self._dream_nodes_reintegrated += reintegrated

        stats["phase"] = "complete"

        if self.synthesis_callback:
            try:
                self.synthesis_callback(stats)
            except Exception:
                pass

        return stats

    def _random_walk(
        self, seed_pool: List[str]
    ) -> Tuple[Dict[str, float], List[Tuple[str, str, float]]]:
        seed = random.choice(seed_pool)
        visited: Dict[str, float] = {seed: 1.0}
        edges: List[Tuple[str, str, float]] = []

        current = seed
        for _ in range(self.walk_steps):
            if current not in self.engine._nodes:
                break

            associations = self.engine.associate(current, max_depth=1)
            if not associations:
                candidates = [
                    nid for nid in self.engine._nodes if nid not in visited
                ]
                if not candidates:
                    break
                current = random.choice(candidates)
                visited[current] = 0.1
                continue

            weights = []
            targets = []
            for node, score, _ in associations:
                if node.id not in visited:
                    w = score ** (1.0 / max(self.walk_temperature, 0.1))
                    weights.append(w)
                    targets.append(node.id)

            if not targets:
                break

            total = sum(weights)
            probs = [w / total for w in weights]
            chosen_idx = random.choices(range(len(targets)), weights=probs, k=1)[0]
            chosen = targets[chosen_idx]
            chosen_score = associations[chosen_idx][1]

            visited[chosen] = chosen_score
            edges.append((current, chosen, chosen_score))
            current = chosen

        return visited, edges

    def _cluster_visited(
        self, visited: Dict[str, float]
    ) -> List[List[str]]:
        nodes_with_geom = []
        for nid in visited:
            if nid in self.engine._nodes:
                nodes_with_geom.append(nid)

        if len(nodes_with_geom) < self.min_cluster_size:
            return [[nid] for nid in nodes_with_geom]

        geometries = np.array([
            self.engine._nodes[nid].geometry for nid in nodes_with_geom
        ])

        n = len(nodes_with_geom)
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(geometries[i] - geometries[j])
                if dist <= self.cluster_eps:
                    union(i, j)

        cluster_map: Dict[int, List[str]] = {}
        for i in range(n):
            root = find(i)
            cluster_map.setdefault(root, []).append(nodes_with_geom[i])

        clusters = [
            members for members in cluster_map.values()
            if len(members) >= self.min_cluster_size
        ]

        return clusters

    def _cluster_label_aware(
        self, visited: Dict[str, float]
    ) -> List[List[str]]:
        valid_ids = [nid for nid in visited if nid in self.engine._nodes]
        if not valid_ids:
            return []

        label_groups: Dict[str, List[str]] = {}
        for nid in valid_ids:
            node = self.engine._nodes[nid]
            primary_label = node.labels[0] if node.labels else "__nolabel__"
            label_groups.setdefault(primary_label, []).append(nid)

        final_clusters: List[List[str]] = []
        for label, members in label_groups.items():
            if len(members) < self.min_cluster_size:
                final_clusters.append(members)
                continue
            sub_clusters = self._cluster_by_geometry(members)
            final_clusters.extend(sub_clusters)

        return final_clusters

    def _cluster_by_geometry(self, node_ids: List[str]) -> List[List[str]]:
        if len(node_ids) <= self.min_cluster_size:
            return [node_ids]

        geometries = np.array([
            self.engine._nodes[nid].geometry for nid in node_ids
        ])

        n = len(node_ids)
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(geometries[i] - geometries[j])
                if dist <= self.cluster_eps:
                    union(i, j)

        cluster_map: Dict[int, List[str]] = {}
        for i in range(n):
            root = find(i)
            cluster_map.setdefault(root, []).append(node_ids[i])

        return list(cluster_map.values())

    def _synthesize_bridges(
        self, clusters: List[List[str]], walk_edges: List[Tuple[str, str, float]]
    ) -> int:
        dream_count = self._count_dream_nodes()
        if dream_count >= self.max_dream_nodes:
            return 0

        created = 0
        edge_set: Set[frozenset] = {
            frozenset([a, b]) for a, b, _ in walk_edges
        }

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                if dream_count + created >= self.max_dream_nodes:
                    return created

                has_connection = False
                for nid_a in clusters[i]:
                    for nid_b in clusters[j]:
                        if frozenset([nid_a, nid_b]) in edge_set:
                            has_connection = True
                            break
                    if has_connection:
                        break

                if not has_connection:
                    continue

                geom_a = np.mean([
                    self.engine._nodes[nid].geometry for nid in clusters[i]
                    if nid in self.engine._nodes
                ], axis=0)
                geom_b = np.mean([
                    self.engine._nodes[nid].geometry for nid in clusters[j]
                    if nid in self.engine._nodes
                ], axis=0)

                bridge_geom = (geom_a + geom_b) / 2.0
                jitter = np.random.normal(0, 0.02, size=bridge_geom.shape)
                bridge_geom = bridge_geom + jitter

                contents_a = [
                    self.engine._nodes[nid].content[:60]
                    for nid in clusters[i][:3]
                    if nid in self.engine._nodes
                ]
                contents_b = [
                    self.engine._nodes[nid].content[:60]
                    for nid in clusters[j][:3]
                    if nid in self.engine._nodes
                ]

                bridge_content = (
                    "[dream] "
                    + "; ".join(contents_a)
                    + " <-> "
                    + "; ".join(contents_b)
                )

                cluster_ids = set(clusters[i] + clusters[j])
                shared_labels = set()
                for nid in cluster_ids:
                    if nid in self.engine._nodes:
                        shared_labels.update(self.engine._nodes[nid].labels)
                shared_labels.discard("__dream__")
                shared_labels.discard("__system__")

                dream_id = self.engine.store(
                    content=bridge_content,
                    labels=list(shared_labels) + ["__dream__"],
                    metadata={
                        "type": "dream_bridge",
                        "source_clusters": [
                            clusters[i][:5], clusters[j][:5]
                        ],
                        "cycle": self._cycle_count,
                    },
                    weight=self.dream_weight,
                )

                if dream_id:
                    created += 1
                    logger.info(
                        "Dream bridge created: %s <-> %s (id=%s)",
                        clusters[i][:2], clusters[j][:2], dream_id[:8],
                    )

        return created

    def _reintegrate_dreams(self) -> int:
        reintegrated = 0
        with self.engine._lock:
            for nid, node in list(self.engine._nodes.items()):
                if "__dream__" in node.labels:
                    node.weight *= 1.15
                    node.weight = min(10.0, node.weight)
                    if node.weight < self.dream_min_weight:
                        node.weight = self.dream_min_weight + 0.1
                        reintegrated += 1
        return reintegrated

    def _count_dream_nodes(self) -> int:
        count = 0
        with self.engine._lock:
            for node in self.engine._nodes.values():
                if "__dream__" in node.labels:
                    count += 1
        return count
