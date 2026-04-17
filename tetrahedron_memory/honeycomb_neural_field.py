"""
HoneycombNeuralField — BCC Lattice Honeycomb + Continuous Neural Pulse Engine.

Architecture:
  Layer 1: BCC lattice honeycomb (space-filling tetrahedral mesh)
  Layer 2: Memory activation (content mapped to lattice nodes)
  Layer 3: Neural pulses (continuous background activity, never stops)

The pulse engine runs a daemon thread that continuously:
  - Emits pulses from high-activation nodes
  - Propagates pulses along face/edge/vertex connections
  - Converging pulses create bridge memories
  - Low-activation nodes decay (but never below threshold)
  - High-frequency access reinforces activation
"""

import hashlib
import logging
import random
import threading
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger("tetramem.honeycomb")


class HoneycombNode:
    __slots__ = (
        "id", "position", "face_neighbors", "edge_neighbors", "vertex_neighbors",
        "content", "labels", "weight", "activation", "base_activation",
        "last_pulse_time", "pulse_accumulator", "creation_time",
        "metadata", "access_count", "decay_rate",
    )

    def __init__(self, id: str, position: np.ndarray):
        self.id = id
        self.position = position
        self.face_neighbors: List[str] = []
        self.edge_neighbors: List[str] = []
        self.vertex_neighbors: List[str] = []
        self.content: Optional[str] = None
        self.labels: List[str] = []
        self.weight: float = 0.0
        self.activation: float = 0.0
        self.base_activation: float = 0.01
        self.last_pulse_time: float = 0.0
        self.pulse_accumulator: float = 0.0
        self.creation_time: float = time.time()
        self.metadata: Dict[str, Any] = {}
        self.access_count: int = 0
        self.decay_rate: float = 0.001

    @property
    def is_occupied(self) -> bool:
        return self.content is not None

    def touch(self):
        self.last_pulse_time = time.time()
        self.access_count += 1

    def decay(self, dt: float):
        if self.is_occupied:
            rate = self.decay_rate / max(self.weight, 0.5)
            self.activation = max(self.base_activation, self.activation - rate * dt)
        else:
            self.pulse_accumulator *= 0.95

    def reinforce(self, amount: float):
        boost = amount * max(self.weight, 0.5) * 0.3
        self.activation = min(10.0, self.activation + amount + boost)


class NeuralPulse:
    __slots__ = ("source_id", "strength", "hops", "path", "direction", "birth_time", "max_hops")

    def __init__(self, source_id: str, strength: float, max_hops: int = 6):
        self.source_id = source_id
        self.strength = strength
        self.hops = 0
        self.path = [source_id]
        self.direction = "face"
        self.birth_time = time.time()
        self.max_hops = max_hops

    def propagate(self, decay: float = 0.7) -> float:
        self.hops += 1
        self.strength *= decay
        return self.strength

    @property
    def alive(self) -> bool:
        return self.strength > 0.01 and self.hops < self.max_hops


class HoneycombNeuralField:
    """BCC Lattice Honeycomb with continuous neural pulse engine."""

    def __init__(self, resolution: int = 5, spacing: float = 1.0):
        self._lock = threading.RLock()
        self._nodes: Dict[str, HoneycombNode] = {}
        self._position_index: Dict[Tuple, str] = {}
        self._label_index: Dict[str, Set[str]] = defaultdict(set)
        self._content_hash_index: Dict[str, str] = {}
        self._edges: List[Tuple[str, str, str]] = []
        self._pulse_engine: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._pulse_interval: float = 0.5
        self._pulse_count: int = 0
        self._bridge_count: int = 0
        self._resolution = resolution
        self._spacing = spacing
        self._pulse_log: List[Dict] = []
        self._max_log = 200

    def initialize(self) -> Dict[str, Any]:
        with self._lock:
            self._build_bcc_lattice()
            self._build_connectivity()
            return self.stats()

    def _build_bcc_lattice(self):
        res = self._resolution
        sp = self._spacing
        for ix in range(-res, res + 1):
            for iy in range(-res, res + 1):
                for iz in range(-res, res + 1):
                    pos = np.array([ix * sp, iy * sp, iz * sp], dtype=np.float32)
                    nid = hashlib.sha256(pos.tobytes()).hexdigest()[:16]
                    self._nodes[nid] = HoneycombNode(nid, pos)
                    self._position_index[(ix, iy, iz)] = nid

        for ix in range(-res, res):
            for iy in range(-res, res):
                for iz in range(-res, res):
                    bpos = np.array([(ix + 0.5) * sp, (iy + 0.5) * sp, (iz + 0.5) * sp], dtype=np.float32)
                    bid = hashlib.sha256(bpos.tobytes()).hexdigest()[:16]
                    self._nodes[bid] = HoneycombNode(bid, bpos)
                    self._position_index[(ix, iy, iz, "b")] = bid

        logger.info("BCC lattice: %d nodes", len(self._nodes))

    def _build_connectivity(self):
        sp = self._spacing
        face_count = 0
        edge_count = 0

        for key, bid in list(self._position_index.items()):
            if not (isinstance(key, tuple) and len(key) == 4 and key[3] == "b"):
                continue
            ix, iy, iz = key[0], key[1], key[2]
            for dx in (0, 1):
                for dy in (0, 1):
                    for dz in (0, 1):
                        ck = (ix + dx, iy + dy, iz + dz)
                        cnid = self._position_index.get(ck)
                        if not cnid:
                            continue
                        bnode = self._nodes.get(bid)
                        cnode = self._nodes.get(cnid)
                        if not bnode or not cnode:
                            continue
                        if cnid not in bnode.face_neighbors:
                            bnode.face_neighbors.append(cnid)
                        if bid not in cnode.face_neighbors:
                            cnode.face_neighbors.append(bid)
                        self._edges.append((bid, cnid, "face"))
                        face_count += 1

        for key, nid in list(self._position_index.items()):
            if not (isinstance(key, tuple) and len(key) == 3):
                continue
            ix, iy, iz = key
            if (ix, iy, iz, "b") in self._position_index:
                continue
            for dx, dy, dz in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
                nk = (ix + dx, iy + dy, iz + dz)
                nnid = self._position_index.get(nk)
                if not nnid or nnid == nid:
                    continue
                node = self._nodes.get(nid)
                nn = self._nodes.get(nnid)
                if not node or not nn:
                    continue
                if nnid not in node.face_neighbors and nnid not in node.edge_neighbors:
                    node.edge_neighbors.append(nnid)
                    if nid not in nn.edge_neighbors:
                        nn.edge_neighbors.append(nid)
                    self._edges.append((nid, nnid, "edge"))
                    edge_count += 1

        logger.info("Connectivity: %d face edges, %d edge connections", face_count, edge_count)

    def store(self, content: str, labels: Optional[List[str]] = None,
              weight: float = 1.0, metadata: Optional[Dict] = None) -> str:
        with self._lock:
            chash = hashlib.sha256(content.encode()).hexdigest()[:12]
            if chash in self._content_hash_index:
                existing_id = self._content_hash_index[chash]
                existing = self._nodes.get(existing_id)
                if existing and existing.is_occupied:
                    existing.reinforce(weight * 0.1)
                    return existing_id

            nid = self._find_nearest_empty_node(content, labels)
            node = self._nodes[nid]
            node.content = content
            node.labels = labels or []
            node.weight = weight
            node.activation = weight
            node.base_activation = max(0.01, weight * 0.1)
            node.metadata = metadata or {}
            node.creation_time = time.time()
            node.touch()

            self._content_hash_index[chash] = nid
            for lbl in node.labels:
                self._label_index[lbl].add(nid)

            self._emit_pulse(nid, strength=weight * 0.5)

            logger.debug("Stored memory at node %s: %s", nid[:8], content[:40])
            return nid

    def _find_nearest_empty_node(self, content: str, labels=None) -> str:
        occupied_nodes = [n for n in self._nodes.values() if n.is_occupied]

        if not occupied_nodes:
            origin = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            best_id = None
            best_dist = float('inf')
            for nid, node in self._nodes.items():
                d = float(np.sum((node.position - origin) ** 2))
                if d < best_dist:
                    best_dist = d
                    best_id = nid
            return best_id

        label_set = set(labels or [])
        related_occ = []
        for on in occupied_nodes:
            overlap = len(label_set & set(on.labels))
            if overlap > 0:
                related_occ.append((on, overlap * 2))
            else:
                related_occ.append((on, 0))

        face_candidates = []
        for on, bonus in related_occ:
            for fnid in on.face_neighbors:
                fn = self._nodes.get(fnid)
                if fn and not fn.is_occupied:
                    face_candidates.append((fnid, fn, bonus + 10))

        if face_candidates:
            face_candidates.sort(key=lambda x: -x[2])
            top_bonus = face_candidates[0][2]
            top_tier = [c for c in face_candidates if c[2] >= top_bonus - 1]
            chosen = random.choice(top_tier)
            return chosen[0]

        edge_candidates = []
        for on, bonus in related_occ:
            for enid in on.edge_neighbors:
                en = self._nodes.get(enid)
                if en and not en.is_occupied:
                    edge_candidates.append((enid, en, bonus + 5))

        if edge_candidates:
            edge_candidates.sort(key=lambda x: -x[2])
            top_bonus = edge_candidates[0][2]
            top_tier = [c for c in edge_candidates if c[2] >= top_bonus - 1]
            chosen = random.choice(top_tier)
            return chosen[0]

        occ_positions = [n.position for n in occupied_nodes]
        centroid = np.mean(occ_positions, axis=0)

        best_id = None
        best_dist = float('inf')
        sample = min(500, len(self._nodes))
        for nid in random.sample(list(self._nodes.keys()), sample):
            node = self._nodes[nid]
            if node.is_occupied:
                continue
            min_occ_dist = min(float(np.sum((node.position - op) ** 2)) for op in occ_positions)
            centroid_dist = float(np.sum((node.position - centroid) ** 2))
            score = min_occ_dist * 0.3 + centroid_dist * 0.7
            if score < best_dist:
                best_dist = score
                best_id = nid

        return best_id or random.choice(list(self._nodes.keys()))

    def query(self, text: str, k: int = 5, labels=None) -> List[Dict]:
        with self._lock:
            qtokens = self._extract_tokens(text) if text else set()

            scored = []
            for nid, node in self._nodes.items():
                if not node.is_occupied:
                    continue
                if labels and not any(l in node.labels for l in labels):
                    continue

                text_score = 0.0
                if qtokens:
                    ctokens = self._extract_tokens(node.content)
                    if ctokens:
                        overlap = len(qtokens & ctokens)
                        text_score = overlap / max(len(qtokens), 1)

                label_score = 0.0
                if labels:
                    overlap = len(set(labels) & set(node.labels))
                    label_score = overlap / max(len(labels), 1)

                activation_score = min(node.activation / 5.0, 1.0)
                weight_score = min(node.weight / 5.0, 1.0)

                final = 0.40 * text_score + 0.25 * label_score + 0.20 * activation_score + 0.15 * weight_score
                scored.append((nid, final))

            scored.sort(key=lambda x: -x[1])

            results = []
            for nid, score in scored[:k]:
                node = self._nodes[nid]
                node.touch()
                node.reinforce(0.05)
                results.append({
                    "id": nid,
                    "content": node.content,
                    "distance": score,
                    "weight": node.weight,
                    "labels": list(node.labels),
                    "activation": node.activation,
                    "metadata": node.metadata,
                })

            if results:
                best_id = results[0]["id"]
                self._emit_pulse(best_id, strength=0.3)

            return results

    def _extract_tokens(self, text: str) -> set:
        import re
        tokens = set()
        for w in re.findall(r"[a-zA-Z0-9]{2,}", text.lower()):
            tokens.add(w)
        for c in re.findall(r"[\u4e00-\u9fff]", text):
            tokens.add(c)
        for bigram in re.findall(r"[\u4e00-\u9fff]{2}", text):
            tokens.add(bigram)
        return tokens

    def _emit_pulse(self, source_id: str, strength: float = 0.5):
        pulse = NeuralPulse(source_id, strength, max_hops=4)
        self._propagate_pulse(pulse)

    def _propagate_pulse(self, pulse: NeuralPulse):
        if not pulse.alive:
            return

        current_id = pulse.path[-1]
        current = self._nodes.get(current_id)
        if current is None:
            return

        current.pulse_accumulator += pulse.strength
        current.last_pulse_time = time.time()

        if current.is_occupied:
            current.reinforce(pulse.strength * 0.01)

        neighbors = []
        for nid in current.face_neighbors:
            if nid not in pulse.path:
                neighbors.append((nid, pulse.strength * 0.7, "face"))
        for nid in current.edge_neighbors:
            if nid not in pulse.path:
                neighbors.append((nid, pulse.strength * 0.35, "edge"))

        if not neighbors:
            return

        next_id, next_strength, direction = random.choice(neighbors)
        new_pulse = NeuralPulse(pulse.source_id, next_strength, pulse.max_hops)
        new_pulse.hops = pulse.hops + 1
        new_pulse.path = pulse.path + [next_id]
        new_pulse.direction = direction
        self._propagate_pulse(new_pulse)

    def start_pulse_engine(self):
        if self._pulse_engine is not None and self._pulse_engine.is_alive():
            return
        self._stop_event.clear()
        self._pulse_engine = threading.Thread(target=self._pulse_loop, name="neural-pulse", daemon=True)
        self._pulse_engine.start()
        logger.info("Neural pulse engine started — pulses never stop")

    def stop_pulse_engine(self):
        self._stop_event.set()
        if self._pulse_engine:
            self._pulse_engine.join(timeout=5)
            self._pulse_engine = None

    def _pulse_loop(self):
        cycle = 0
        while not self._stop_event.wait(timeout=self._pulse_interval):
            try:
                self._pulse_cycle()
                cycle += 1
                if cycle % 60 == 0:
                    self._check_convergence_bridges()
                if cycle % 120 == 0:
                    self._global_decay()
            except Exception as e:
                logger.error("Pulse cycle error: %s", e, exc_info=True)

    def _pulse_cycle(self):
        with self._lock:
            occupied = [(nid, n) for nid, n in self._nodes.items() if n.is_occupied]
            if not occupied:
                return

            if random.random() < 0.7:
                weighted = [(nid, n.activation * max(n.weight, 0.5)) for nid, n in occupied]
                total = sum(w for _, w in weighted)
                if total <= 0:
                    nid = random.choice(occupied)[0]
                else:
                    nid = random.choices([i for i, _ in weighted], weights=[w for _, w in weighted], k=1)[0]
            else:
                unoccupied = [(nid, n) for nid, n in self._nodes.items() if not n.is_occupied and n.pulse_accumulator > 0.01]
                if unoccupied:
                    nid = random.choice(unoccupied)[0]
                else:
                    nid = random.choice(occupied)[0]

            node = self._nodes[nid]
            strength = 0.1 + min(node.weight, 8.0) * 0.05
            strength = min(strength, 0.6) * random.uniform(0.8, 1.2)
            self._emit_pulse(nid, strength)
            self._pulse_count += 1

    def _check_convergence_bridges(self):
        with self._lock:
            hot_empty = [(nid, n) for nid, n in self._nodes.items()
                         if not n.is_occupied and n.pulse_accumulator > 0.5]
            if len(hot_empty) < 2:
                return

            for nid, node in hot_empty[:3]:
                sources = set()
                for fnid in node.face_neighbors:
                    fn = self._nodes.get(fnid)
                    if fn and fn.is_occupied:
                        sources.add(fnid)
                for enid in node.edge_neighbors[:4]:
                    en = self._nodes.get(enid)
                    if en and en.is_occupied:
                        sources.add(enid)

                if len(sources) >= 2 and not node.is_occupied:
                    source_contents = []
                    source_labels = set()
                    for sid in list(sources)[:4]:
                        sn = self._nodes.get(sid)
                        if sn and sn.is_occupied:
                            source_contents.append(sn.content[:60])
                            source_labels.update(sn.labels)
                    source_labels.discard("__dream__")
                    source_labels.discard("__system__")
                    label_str = ", ".join(list(source_labels)[:4]) if source_labels else "general"
                    bridge = f"[pulse:bridge:{label_str}] {' | '.join(source_contents[:3])}"
                    node.content = bridge
                    node.labels = list(source_labels)[:6] + ["__pulse_bridge__"]
                    node.weight = min(node.pulse_accumulator, 1.0)
                    node.activation = node.pulse_accumulator * 0.5
                    node.base_activation = 0.05
                    chash = hashlib.sha256(bridge.encode()).hexdigest()[:12]
                    self._content_hash_index[chash] = nid
                    for lbl in node.labels:
                        self._label_index[lbl].add(nid)
                    node.pulse_accumulator = 0.0
                    self._bridge_count += 1

                    self._pulse_log.append({
                        "time": time.time(),
                        "bridge_id": nid,
                        "strength": node.weight,
                        "sources": len(sources),
                    })
                    if len(self._pulse_log) > self._max_log:
                        self._pulse_log = self._pulse_log[-self._max_log // 2:]

    def _global_decay(self):
        now = time.time()
        for node in self._nodes.values():
            dt = now - node.last_pulse_time if node.last_pulse_time > 0 else 60
            node.decay(dt)

    def get_node(self, nid: str) -> Optional[Dict]:
        node = self._nodes.get(nid)
        if node is None:
            return None
        return {
            "id": node.id,
            "content": node.content,
            "position": node.position.tolist(),
            "centroid": node.position.tolist(),
            "labels": list(node.labels),
            "weight": node.weight,
            "activation": node.activation,
            "face_neighbors": len(node.face_neighbors),
            "edge_neighbors": len(node.edge_neighbors),
            "creation_time": node.creation_time,
            "access_count": node.access_count,
            "metadata": node.metadata,
        }

    def list_occupied(self) -> List[Dict]:
        with self._lock:
            results = []
            for nid, node in self._nodes.items():
                if node.is_occupied:
                    results.append(self.get_node(nid))
            return results

    def topology_graph(self) -> Dict:
        with self._lock:
            nodes = []
            for nid, node in self._nodes.items():
                if node.is_occupied or node.pulse_accumulator > 0.1:
                    nodes.append({
                        "id": nid,
                        "centroid": node.position.tolist(),
                        "weight": node.weight,
                        "labels": list(node.labels),
                        "is_dream": "__pulse_bridge__" in node.labels,
                        "activation": node.activation,
                        "occupied": node.is_occupied,
                    })

            occupied_ids = {n["id"] for n in nodes}
            edges = []
            for n1, n2, etype in self._edges:
                if n1 in occupied_ids and n2 in occupied_ids:
                    edges.append({"source": n1, "target": n2, "type": etype})

            return {"nodes": nodes, "edges": edges}

    def stats(self) -> Dict:
        with self._lock:
            total = len(self._nodes)
            occupied = sum(1 for n in self._nodes.values() if n.is_occupied)
            bridges = sum(1 for n in self._nodes.values() if "__pulse_bridge__" in n.labels)
            face_edges = sum(1 for _, _, t in self._edges if t == "face")
            edge_edges = sum(1 for _, _, t in self._edges if t == "edge")
            avg_activation = np.mean([n.activation for n in self._nodes.values()]) if self._nodes else 0
            avg_face_conn = np.mean([len(n.face_neighbors) for n in self._nodes.values()]) if self._nodes else 0

            return {
                "total_nodes": total,
                "occupied_nodes": occupied,
                "bridge_nodes": bridges,
                "empty_nodes": total - occupied,
                "face_edges": face_edges,
                "edge_edges": edge_edges,
                "avg_activation": float(avg_activation),
                "avg_face_connections": float(avg_face_conn),
                "pulse_count": self._pulse_count,
                "bridge_count": self._bridge_count,
                "pulse_engine_running": self._pulse_engine is not None and self._pulse_engine.is_alive(),
            }

    def browse_timeline(self, direction: str = "newest", limit: int = 20,
                        label_filter=None, min_weight: float = 0.0) -> List[Dict]:
        with self._lock:
            items = []
            for nid, node in self._nodes.items():
                if not node.is_occupied:
                    continue
                if min_weight > 0 and node.weight < min_weight:
                    continue
                if label_filter and not any(l in node.labels for l in label_filter):
                    continue
                items.append({
                    "id": nid,
                    "content": node.content,
                    "weight": node.weight,
                    "labels": list(node.labels),
                    "creation_time": node.creation_time,
                    "activation": node.activation,
                })
            items.sort(key=lambda x: x["creation_time"], reverse=(direction == "newest"))
            return items[:limit]

    def associate(self, tetra_id: str, max_depth: int = 2) -> List[Dict]:
        with self._lock:
            node = self._nodes.get(tetra_id)
            if node is None:
                return []

            visited = {tetra_id}
            frontier = [tetra_id]
            results = []

            for depth in range(max_depth):
                next_frontier = []
                for fid in frontier:
                    fn = self._nodes.get(fid)
                    if fn is None:
                        continue
                    for nid in fn.face_neighbors + fn.edge_neighbors:
                        if nid in visited:
                            continue
                        visited.add(nid)
                        nn = self._nodes.get(nid)
                        if nn and nn.is_occupied:
                            conn = "face" if nid in fn.face_neighbors else "edge"
                            results.append({
                                "id": nid,
                                "content": nn.content,
                                "type": conn,
                                "weight": nn.weight,
                                "labels": list(nn.labels),
                                "activation": nn.activation,
                            })
                        next_frontier.append(nid)
                frontier = next_frontier

            return results

    def pulse_status(self) -> Dict:
        with self._lock:
            recent = self._pulse_log[-10:]
            hot_nodes = sorted(
                [(nid, n.pulse_accumulator) for nid, n in self._nodes.items() if n.pulse_accumulator > 0.1],
                key=lambda x: -x[1]
            )[:10]
            return {
                "pulse_count": self._pulse_count,
                "bridge_count": self._bridge_count,
                "engine_running": self._pulse_engine is not None and self._pulse_engine.is_alive(),
                "recent_bridges": recent,
                "hot_nodes": [(nid[:8], round(acc, 3)) for nid, acc in hot_nodes],
            }
