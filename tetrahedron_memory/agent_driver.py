from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .pcnn_types import PCNNConfig

if TYPE_CHECKING:
    from .honeycomb_neural_field import HoneycombNeuralField


class AgentMemoryDriver:
    """
    Memory-driven agent capability layer.

    Provides the interface between TetraMem's neural memory and agent actions:
    - Context injection: assemble relevant memories for current task context
    - Reasoning chains: multi-hop paths from source memory to target insight
    - Proactive suggestions: memory pattern-based action recommendations
    - Navigate: path-finding through the memory topology graph
    """

    def __init__(self, field: "HoneycombNeuralField"):
        self._field = field

    def get_context(self, topic: str, max_memories: int = 15) -> Dict[str, Any]:
        field = self._field
        cfg = PCNNConfig
        n = min(max_memories, cfg.AGENT_CONTEXT_MAX_MEMORIES)

        memories = field.query(topic, k=n * 2)
        if not memories:
            return {"topic": topic, "context": [], "reasoning": "no relevant memories found"}

        core = [m for m in memories if "__dream__" not in m.get("labels", [])][:n // 2]
        dreams = [m for m in memories if "__dream__" in m.get("labels", [])][:n // 4]
        bridges = [m for m in memories if "__pulse_bridge__" in m.get("labels", [])][:n // 4]

        context_memories = core + dreams + bridges
        context_memories.sort(key=lambda m: -m.get("distance", 0))

        labels_encountered = set()
        for m in context_memories:
            for l in m.get("labels", []):
                if not l.startswith("__"):
                    labels_encountered.add(l)

        associations = []
        if core:
            top_id = core[0]["id"]
            assoc = field.associate(top_id, max_depth=2)
            for a in assoc[:5]:
                if "__dream__" not in a.get("labels", []) and "__pulse_bridge__" not in a.get("labels", []):
                    associations.append({
                        "content": a["content"][:80],
                        "weight": a["weight"],
                        "connection": a["type"],
                    })

        reasoning = self._build_reasoning(topic, context_memories)

        return {
            "topic": topic,
            "context_count": len(context_memories),
            "context": [
                {
                    "id": m["id"][:12],
                    "content": m["content"][:120],
                    "weight": m.get("weight", 0),
                    "relevance": round(m.get("distance", 0), 3),
                    "labels": [l for l in m.get("labels", []) if not l.startswith("__")],
                    "is_dream": "__dream__" in m.get("labels", []),
                }
                for m in context_memories[:n]
            ],
            "related_labels": list(labels_encountered),
            "associations": associations,
            "reasoning": reasoning,
        }

    def reasoning_chain(self, source_id: str, target_query: str, max_hops: int = 5) -> Dict[str, Any]:
        field = self._field
        cfg = PCNNConfig
        hops = min(max_hops, cfg.AGENT_REASONING_MAX_HOPS)

        source = field.get_node(source_id)
        if source is None:
            return {"error": "source node not found"}

        targets = field.query(target_query, k=3)
        if not targets:
            return {"source": source_id[:12], "chain": [], "conclusion": "no target found"}

        target_ids = {t["id"] for t in targets}

        visited = {source_id}
        frontier = [(source_id, [source_id])]
        found_path = None

        for depth in range(hops):
            next_frontier = []
            for nid, path in frontier:
                node = field._nodes.get(nid)
                if node is None:
                    continue
                neighbors = []
                for fnid in node.face_neighbors:
                    fn = field._nodes.get(fnid)
                    if fn and fn.is_occupied and fnid not in visited:
                        score = fn.weight * fn.activation
                        if fnid in target_ids:
                            score += 100.0
                        neighbors.append((fnid, score))
                for enid in node.edge_neighbors[:6]:
                    en = field._nodes.get(enid)
                    if en and en.is_occupied and enid not in visited:
                        score = en.weight * en.activation * 0.5
                        if enid in target_ids:
                            score += 100.0
                        neighbors.append((enid, score))

                neighbors.sort(key=lambda x: -x[1])

                for nnid, score in neighbors[:3]:
                    visited.add(nnid)
                    new_path = path + [nnid]
                    if nnid in target_ids:
                        found_path = new_path
                        break
                    next_frontier.append((nnid, new_path))

                if found_path:
                    break
            frontier = next_frontier
            if found_path:
                break

        if not found_path:
            return {
                "source": source_id[:12],
                "target_query": target_query,
                "chain": [],
                "conclusion": "no path found within hop limit",
            }

        chain_nodes = []
        for nid in found_path:
            node = field._nodes.get(nid)
            if node:
                chain_nodes.append({
                    "id": nid[:12],
                    "content": node.content[:80],
                    "weight": round(node.weight, 2),
                    "labels": [l for l in node.labels if not l.startswith("__")],
                })

        return {
            "source": source_id[:12],
            "target_query": target_query,
            "chain_length": len(chain_nodes),
            "chain": chain_nodes,
            "conclusion": chain_nodes[-1]["content"][:100] if chain_nodes else "",
        }

    def suggest_actions(self, context: str = "") -> Dict[str, Any]:
        field = self._field
        cfg = PCNNConfig

        suggestions = []

        isolated = field.detect_isolated()
        if isolated:
            suggestions.append({
                "action": "connect_isolated_memories",
                "priority": "high",
                "description": f"{len(isolated)} memories have no occupied neighbors. Consider adding bridging memories or triggering cascade pulses.",
                "affected_count": len(isolated),
            })

        stats = field.stats()
        if stats.get("bridge_rate", 0) < 0.001:
            suggestions.append({
                "action": "increase_pulse_activity",
                "priority": "medium",
                "description": "Bridge rate is very low. The neural field may benefit from more exploration pulses or manual memory additions.",
                "current_bridge_rate": round(stats.get("bridge_rate", 0), 6),
            })

        so_stats = stats.get("self_organize", {})
        if so_stats.get("active_clusters", 0) > 0:
            suggestions.append({
                "action": "leverage_clusters",
                "priority": "medium",
                "description": f"{so_stats['active_clusters']} semantic clusters detected. Use cluster-aware queries for better recall.",
                "cluster_count": so_stats["active_clusters"],
            })

        hc = stats.get("honeycomb_cells", {})
        if hc and hc.get("density", {}).get("occupied_cells", 0) > 0:
            avg_density = hc["density"].get("mean", 0)
            if avg_density > 0.5:
                suggestions.append({
                    "action": "expand_memory_space",
                    "priority": "low",
                    "description": f"Average tetrahedral cell density is {avg_density:.2f}. Consider storing memories in new areas.",
                    "avg_density": round(avg_density, 3),
                })

        if context:
            relevant = field.query(context, k=3)
            if relevant:
                top = relevant[0]
                suggestions.append({
                    "action": "deepen_context",
                    "priority": "medium",
                    "description": f"Most relevant memory (weight={top['weight']:.1f}): {top['content'][:80]}",
                    "memory_id": top["id"][:12],
                })

        return {
            "context": context or "general",
            "suggestion_count": len(suggestions),
            "suggestions": suggestions[:cfg.AGENT_SUGGESTION_TOP_N],
        }

    def navigate(self, source_id: str, target_id: str, max_hops: int = 6) -> Dict[str, Any]:
        field = self._field

        src_node = field._nodes.get(source_id)
        tgt_node = field._nodes.get(target_id)
        if not src_node or not tgt_node:
            return {"path": [], "length": 0, "error": "source or target not found"}

        visited = {source_id}
        frontier = [(source_id, [source_id], 0.0)]
        best_path = None
        best_score = float('inf')

        for depth in range(max_hops):
            next_frontier = []
            for nid, path, cost in frontier:
                node = field._nodes.get(nid)
                if node is None:
                    continue
                neighbors = list(node.face_neighbors) + list(node.edge_neighbors[:4])
                if field._self_organize and hasattr(field._self_organize, '_shortcut_by_node'):
                    for sc_key, sc_str in field._self_organize._shortcut_by_node.get(nid, []):
                        partner = sc_key[1] if sc_key[0] == nid else sc_key[0]
                        if partner not in visited:
                            neighbors.append(partner)
                for nnid in neighbors:
                    if nnid in visited:
                        continue
                    visited.add(nnid)
                    nn = field._nodes.get(nnid)
                    step_cost = 1.0
                    if nn:
                        if nn.is_occupied:
                            step_cost = 1.0 / (nn.weight + 0.1)
                        else:
                            step_cost = 2.0
                        crystal_boost = field._crystallized.get_boost(nid, nnid)
                        if crystal_boost > 1.0:
                            step_cost /= crystal_boost
                        hebbian_w = field._hebbian.get_path_bias(nid, nnid)
                        if hebbian_w > 0:
                            step_cost /= (1.0 + hebbian_w * 0.5)
                        if field._reflection_field:
                            energy = field._reflection_field._node_energy.get(nnid, 0.5)
                            step_cost *= (0.8 + energy * 0.4)
                    new_cost = cost + step_cost
                    new_path = path + [nnid]
                    if nnid == target_id:
                        if new_cost < best_score:
                            best_path = new_path
                            best_score = new_cost
                    else:
                        next_frontier.append((nnid, new_path, new_cost))
            frontier = next_frontier
            if not frontier:
                break

        if not best_path:
            return {"path": [], "length": 0, "source": source_id[:12], "target": target_id[:12]}

        path_data = []
        for nid in best_path:
            node = field._nodes.get(nid)
            if node:
                path_data.append({
                    "id": nid[:12],
                    "content": node.content[:60] if node.content else "",
                    "weight": round(node.weight, 2),
                    "occupied": node.is_occupied,
                })

        return {
            "source": source_id[:12],
            "target": target_id[:12],
            "path": path_data,
            "length": len(best_path) - 1,
            "cost": round(best_score, 3),
        }

    def _build_reasoning(self, topic: str, memories: List[Dict]) -> str:
        if not memories:
            return f"No memories related to '{topic}'."
        top = memories[0]
        labels = [l for l in top.get("labels", []) if not l.startswith("__")]
        parts = [f"Regarding '{topic}':"]
        parts.append(f"- Primary recall: {top.get('content', '')[:80]} (relevance: {top.get('distance', 0):.2f})")
        if labels:
            parts.append(f"- Related domains: {', '.join(labels[:5])}")
        dream_count = sum(1 for m in memories if "__dream__" in m.get("labels", []))
        if dream_count > 0:
            parts.append(f"- {dream_count} dream insights available")
        parts.append(f"- Total context: {len(memories)} memories")
        return "\n".join(parts)
