from __future__ import annotations

import time

import numpy as np
from fastapi import APIRouter, HTTPException, Request

from tetrahedron_memory.app_state import AppState

router = APIRouter(prefix="/api/v1", tags=["agent"])


def _get_state(request: Request) -> AppState:
    return request.app.state.tetramem


@router.post("/agent/context")
def agent_context(request: Request, body: dict = None):
    s = _get_state(request)
    req = body or {}
    topic = req.get("topic", "")
    max_mem = req.get("max_memories", 15)
    if not topic:
        raise HTTPException(400, "topic is required")
    with s.state_lock:
        return s.field.agent_get_context(topic, max_mem)


@router.post("/agent/reasoning")
def agent_reasoning(request: Request, body: dict = None):
    s = _get_state(request)
    req = body or {}
    source_id = req.get("source_id", "")
    target_query = req.get("target_query", "")
    if not source_id or not target_query:
        return {"chain": [], "error": "source_id and target_query are required"}
    with s.state_lock:
        return s.field.agent_reasoning_chain(source_id, target_query, req.get("max_hops", 5))


@router.post("/agent/suggest")
def agent_suggest(request: Request, body: dict = None):
    s = _get_state(request)
    req = body or {}
    context = req.get("context", "")
    with s.state_lock:
        return s.field.agent_suggest(context)


@router.post("/agent/heartbeat")
def agent_heartbeat(request: Request, body: dict = None):
    s = _get_state(request)
    req = body or {}
    agent_id = req.get("agent_id", "default")
    status = req.get("status", "active")
    with s.agent_hb_lock:
        s.agent_heartbeats[agent_id] = time.time()
    if s.insight_aggregator is not None:
        s.insight_aggregator.register_agent(agent_id)
    idle_triggers = []
    if status == "idle":
        with s.state_lock:
            st = s.field.stats()
        if st.get("bridge_nodes", 0) > 10:
            idle_triggers.append("bridge_review")
        if st.get("cascade_count", 0) % 1000 < 50:
            idle_triggers.append("cascade_activity")
    return {"acknowledged": True, "agent_id": agent_id, "idle_triggers": idle_triggers}


@router.get("/agent/status")
def agent_status(request: Request):
    s = _get_state(request)
    now = time.time()
    agents = []
    with s.agent_hb_lock:
        items = list(s.agent_heartbeats.items())
    for aid, ts in items:
        agents.append({"agent_id": aid, "last_seen": ts, "idle_seconds": int(now - ts), "status": "active" if now - ts < 60 else "idle"})
    return {"agents": agents, "total": len(agents)}


@router.post("/agent/{agent_id}/notifications")
def agent_notifications(request: Request, agent_id: str, body: dict = None):
    s = _get_state(request)
    req = body or {}
    unread_only = req.get("unread_only", True)
    if s.insight_aggregator is None:
        return {"notifications": [], "total": 0}
    s.insight_aggregator.register_agent(agent_id)
    items = s.insight_aggregator.get_notifications(agent_id, unread_only=unread_only)
    return {"notifications": items, "total": len(items)}


@router.post("/agent/{agent_id}/notifications/consume")
def agent_notifications_consume(request: Request, agent_id: str, body: dict = None):
    s = _get_state(request)
    req = body or {}
    ids = req.get("notification_ids", [])
    if s.insight_aggregator is None:
        return {"consumed": 0}
    s.insight_aggregator.mark_consumed(agent_id, ids)
    return {"consumed": len(ids)}


@router.post("/agent/evolution-cycle")
def agent_evolution_cycle(request: Request):
    s = _get_state(request)
    if s.agent_loop is None:
        raise HTTPException(503, "Agent loop not initialized")
    with s.state_lock:
        report = s.agent_loop.run_evolution_cycle()
    s.emit_event("evolution_cycle_completed", {
        "cycle": report.get("cycle"),
        "quality": report.get("phases", {}).get("LEARN", {}).get("quality_score"),
        "duration": report.get("duration_seconds"),
    })
    return report


@router.get("/agent/evolution-report")
def agent_evolution_report(request: Request):
    s = _get_state(request)
    if s.agent_loop is None:
        raise HTTPException(503, "Agent loop not initialized")
    return s.agent_loop.get_evolution_report()


@router.get("/agent/proactive-suggestions")
def agent_proactive_suggestions(request: Request, context: str = ""):
    s = _get_state(request)
    if s.agent_loop is None:
        raise HTTPException(503, "Agent loop not initialized")
    suggestions = s.agent_loop.get_proactive_suggestions(context)
    return {"suggestions": suggestions, "count": len(suggestions)}


@router.get("/agent/{agent_id}/recommendations")
def agent_recommendations(request: Request, agent_id: str):
    s = _get_state(request)
    if s.insight_aggregator is None:
        return {"recommendations": []}
    s.insight_aggregator.register_agent(agent_id)
    notifications = s.insight_aggregator.get_notifications(agent_id, unread_only=True)
    recommendations = []
    for n in notifications:
        if n.get("action"):
            recommendations.append({
                "id": n.get("id"),
                "type": n.get("type"),
                "priority": n.get("priority"),
                "title": n.get("title"),
                "action": n.get("action"),
                "timestamp": n.get("timestamp"),
            })
    recommendations.sort(key=lambda x: -x.get("priority", 0))
    return {"recommendations": recommendations[:20]}


@router.post("/feedback/record")
def feedback_record(request: Request, body: dict = None):
    s = _get_state(request)
    req = body or {}
    action = req.get("action", "")
    context_id = req.get("context_id", "")
    outcome = req.get("outcome", "neutral")
    confidence = req.get("confidence", 0.5)
    reasoning = req.get("reasoning", "")
    metadata = req.get("metadata")
    if not action or not context_id:
        raise HTTPException(400, "action and context_id are required")
    with s.state_lock:
        result = s.field.feedback_record(action, context_id, outcome, confidence, reasoning, metadata)
    s.emit_event("feedback_recorded", {"action": action, "outcome": outcome})
    return result


@router.post("/feedback/learn")
def feedback_learn(request: Request, body: dict = None):
    s = _get_state(request)
    req = body or {}
    action = req.get("action", "")
    source_id = req.get("source_id", "")
    target_id = req.get("target_id", "")
    success = req.get("success", True)
    confidence = req.get("confidence", 0.5)
    if not action or not source_id or not target_id:
        raise HTTPException(400, "action, source_id, and target_id are required")
    with s.state_lock:
        result = s.field.feedback_learn(action, source_id, target_id, success, confidence)
    if success:
        s.emit_event("feedback_learned", {"action": action, "source": source_id[:12], "target": target_id[:12]})
    return result


@router.get("/feedback/stats")
def feedback_stats(request: Request):
    s = _get_state(request)
    with s.state_lock:
        return s.field.feedback_stats()


@router.get("/feedback/insights")
def feedback_insights(request: Request):
    s = _get_state(request)
    with s.state_lock:
        return {"insights": s.field.feedback_insights()}


@router.post("/session/create")
def session_create(request: Request, body: dict = None):
    s = _get_state(request)
    req = body or {}
    agent_id = req.get("agent_id", "default")
    metadata = req.get("metadata")
    auto_load = req.get("auto_load_context", True)
    with s.state_lock:
        session_id = s.field.session_create(agent_id, metadata)
    s.emit_event("session_created", {"session_id": session_id, "agent_id": agent_id})

    ctx = {"session_id": session_id, "auto_loaded": False, "identity": None, "preferences": []}
    if auto_load:
        try:
            with s.state_lock:
                identity = s.field.query("agent identity opencode", k=3)
                prefs = s.field.query("用户偏好 BOSS project status", k=5)
            ctx["auto_loaded"] = True
            ctx["identity"] = identity[:2] if identity else []
            ctx["preferences"] = prefs[:3] if prefs else []
        except Exception:
            pass

    with s.agent_hb_lock:
        s.agent_heartbeats[agent_id] = time.time()
    return ctx


@router.post("/session/{session_id}/add")
def session_add(request: Request, session_id: str, body: dict = None):
    s = _get_state(request)
    req = body or {}
    role = req.get("role", "user")
    content = req.get("content", "")
    metadata = req.get("metadata")
    with s.state_lock:
        return s.field.session_add(session_id, role, content, metadata)


@router.get("/session/{session_id}/recall")
def session_recall(request: Request, session_id: str, n: int = 20):
    s = _get_state(request)
    with s.state_lock:
        return s.field.session_recall(session_id, n)


@router.post("/session/{session_id}/consolidate")
def session_consolidate(request: Request, session_id: str):
    s = _get_state(request)
    with s.state_lock:
        result = s.field.session_consolidate(session_id)
    s.emit_event("session_consolidated", {"session_id": session_id})
    return result


@router.get("/session/list")
def session_list(request: Request):
    s = _get_state(request)
    with s.state_lock:
        return {"sessions": s.field.session_list()}


@router.get("/session/{session_id}")
def session_get(request: Request, session_id: str):
    s = _get_state(request)
    with s.state_lock:
        result = s.field.session_get(session_id)
    if result is None:
        raise HTTPException(404, "Session not found")
    return result


@router.post("/session/{session_id}/close")
def session_close(request: Request, session_id: str):
    s = _get_state(request)
    with s.state_lock:
        result = s.field.session_consolidate(session_id)
    s.emit_event("session_closed", {"session_id": session_id})
    return result


@router.post("/proactive/trigger")
def proactive_trigger(request: Request, body: dict = None):
    s = _get_state(request)
    req = body or {}
    action = req.get("action", "dream")
    results = {}
    with s.state_lock:
        if action in ("dream", "all"):
            results["dream"] = s.field.run_dream_cycle()
        if action in ("self_organize", "all"):
            results["self_organize"] = s.field.run_self_organize()
        if action in ("cascade", "all"):
            results["cascade"] = s.field.trigger_cascade()
        if action in ("self_check", "all"):
            results["self_check"] = s.field.self_check_status()
    if results:
        s.emit_event("proactive_triggered", {"action": action, "results": {k: type(v).__name__ for k, v in results.items()}})
    s.log_op("proactive_trigger")
    return {"action": action, "results": results}


@router.post("/closed-loop")
def closed_loop(request: Request, body: dict = None):
    s = _get_state(request)
    req = body or {}
    context = req.get("context", "")
    k = req.get("k", 5)
    force_dream = req.get("force_dream", False)
    result = {"phases": [], "stores": 0, "integrations": 0}
    with s.state_lock:
        s_before = s.field.stats()
        result["entropy_before"] = {
            "occupied": s_before["occupied_nodes"],
            "avg_activation": s_before["avg_activation"],
            "cascade_count": s_before["cascade_count"],
            "bridge_nodes": s_before["bridge_nodes"],
        }

        result["phases"].append("RECALL")
        if context:
            recall_raw = s.field.query(context, k=k)
            memories = [
                {"id": r.get("id", ""), "content": r.get("content", "")[:200], "weight": r.get("weight", 0), "labels": r.get("labels", [])}
                for r in recall_raw
            ]
            recall_method = "query"
        else:
            all_occ = [(n.id, n) for n in s.field.all_nodes_values() if n.is_occupied]
            sample_n = min(k, len(all_occ))
            if sample_n > 0:
                indices = np.random.choice(len(all_occ), size=sample_n, replace=False)
                sampled = [all_occ[i] for i in indices]
            else:
                sampled = []
            memories = [
                {"id": nid, "content": n.content[:200], "weight": n.weight, "labels": n.labels}
                for nid, n in sampled
            ]
            recall_method = "random"
        result["recall"] = {"count": len(memories), "method": recall_method}

        result["phases"].append("THINK")
        if memories:
            weights_arr = [m["weight"] for m in memories]
            avg_w = float(np.mean(weights_arr)) if weights_arr else 0
            all_labels = set()
            for m in memories:
                all_labels.update(l for l in m["labels"] if not l.startswith("__"))
            confidence = min(1.0, avg_w / 5.0)
            think = {"avg_weight": avg_w, "confidence": confidence, "patterns": list(all_labels)[:5], "memory_count": len(memories)}
        else:
            think = {"avg_weight": 0, "confidence": 0.0, "patterns": [], "memory_count": 0}
        result["think"] = think

        result["phases"].append("EXECUTE")
        execute = {"action": f"derived_from_{think['memory_count']}_memories", "confidence": think["confidence"]}
        result["execute"] = execute

        result["phases"].append("REFLECT")
        should_integrate = think["confidence"] >= 0.2
        should_dream = force_dream or think["confidence"] < 0.5
        reflect = {"quality": think["confidence"], "should_integrate": should_integrate, "should_dream": should_dream}
        result["reflect"] = reflect

        result["phases"].append("INTEGRATE")
        integration_count = 0
        if should_integrate:
            so_result = s.field.run_self_organize()
            integration_count = so_result.get("consolidations_done", 0) + so_result.get("shortcuts_created", 0)
            s.field.check_convergence_bridges()
            if think["confidence"] >= 0.4 and think["patterns"]:
                insight = f"Closed-loop insight: patterns={think['patterns'][:3]}, avg_weight={think['avg_weight']:.2f}"
                s.field.store(content=insight, labels=["__closed_loop__", "__insight__"] + think["patterns"][:2], weight=0.3 + think["confidence"] * 0.7)
                result["stores"] += 1
        result["integrations"] = integration_count

        result["phases"].append("DREAM")
        if should_dream:
            dream_result = s.field.run_dream_cycle()
            result["dream"] = {"triggered": True, "summary": {k: dream_result.get(k) for k in ["dreams_created", "cross_domain", "insights", "depth_levels"] if k in dream_result}}
        else:
            result["dream"] = {"triggered": False}

        s_after = s.field.stats()
        result["entropy_after"] = {
            "occupied": s_after["occupied_nodes"],
            "avg_activation": s_after["avg_activation"],
            "cascade_count": s_after["cascade_count"],
            "bridge_nodes": s_after["bridge_nodes"],
        }
        result["phases"].append("COMPLETE")

    s.log_op("closed_loop")
    s.emit_event("closed_loop_completed", {"context": context[:100], "phases": result["phases"], "stores": result["stores"]})
    return result
