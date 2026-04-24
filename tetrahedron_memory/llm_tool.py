import json
from typing import Any, Dict, List, Optional

TOOL_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "tetramem_store",
            "description": "Store a new memory in the TetraMem geometric memory system.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The memory content to store.",
                    },
                    "labels": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Labels/tags for categorizing the memory.",
                    },
                    "weight": {
                        "type": "number",
                        "description": "Importance weight (0.1-10.0, default 1.0).",
                        "default": 1.0,
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Optional additional metadata key-value pairs.",
                    },
                },
                "required": ["content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tetramem_query",
            "description": "Query memories by text similarity in the TetraMem system. Returns the k most relevant memories.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query text to search for.",
                    },
                    "k": {
                        "type": "integer",
                        "description": "Number of results to return (default 5).",
                        "default": 5,
                    },
                    "use_persistence": {
                        "type": "boolean",
                        "description": "Whether to use persistent homology scoring (default true).",
                        "default": True,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tetramem_associate",
            "description": "Find memories associated with a specific memory using multi-layer association rules (adjacency, path, metric, PH patterns).",
            "parameters": {
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "The ID of the memory to find associations for.",
                    },
                    "max_depth": {
                        "type": "integer",
                        "description": "Maximum association depth (default 2).",
                        "default": 2,
                    },
                },
                "required": ["memory_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tetramem_self_organize",
            "description": "Trigger self-organization of the memory space. Uses persistent homology to perform geometric surgery: edge contractions, repulsions, cave growths, and integration catalyst.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tetramem_stats",
            "description": "Get statistics about the current state of the memory system.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tetramem_dream",
            "description": "Trigger a dream cycle for memory consolidation. The system performs a PH-weighted random walk, discovers topologically related memories, and synthesizes new insights. No memories are deleted.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tetramem_closed_loop",
            "description": "Run a complete cognitive closed-loop cycle: RECALL -> THINK -> EXECUTE -> REFLECT -> INTEGRATE -> DREAM. This is the core autonomous cognitive process.",
            "parameters": {
                "type": "object",
                "properties": {
                    "context": {
                        "type": "string",
                        "description": "Context text for the cycle. Empty string triggers self-emergent cycle.",
                        "default": "",
                    },
                    "k": {
                        "type": "integer",
                        "description": "Number of memories to recall (default 5).",
                        "default": 5,
                    },
                    "force_dream": {
                        "type": "boolean",
                        "description": "Force dream phase regardless of entropy.",
                        "default": False,
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tetramem_weight_update",
            "description": "Update the weight of an existing memory. Supports both direct delta and EMA smoothing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "The ID of the memory to update.",
                    },
                    "delta": {
                        "type": "number",
                        "description": "Weight change amount.",
                    },
                    "use_ema": {
                        "type": "boolean",
                        "description": "Use exponential moving average (default true).",
                        "default": True,
                    },
                    "alpha": {
                        "type": "number",
                        "description": "EMA smoothing factor (default 0.1).",
                        "default": 0.1,
                    },
                },
                "required": ["memory_id", "delta"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tetramem_batch_store",
            "description": "Store multiple memories at once for batch insertion.",
            "parameters": {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "content": {"type": "string"},
                                "labels": {"type": "array", "items": {"type": "string"}},
                                "weight": {"type": "number", "default": 1.0},
                            },
                            "required": ["content"],
                        },
                        "description": "Array of memory items to store.",
                    },
                },
                "required": ["items"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tetramem_persist",
            "description": "Flush current memory state to persistent storage (Parquet). Ensures durability of all stored memories.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tetramem_query_multiparam",
            "description": "Multi-parameter filtered query combining spatial proximity, temporal recency, density, weight, topology connectivity, and label matching. More powerful than basic query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query text to search for.",
                    },
                    "k": {
                        "type": "integer",
                        "description": "Number of results (default 10).",
                        "default": 10,
                    },
                    "labels_required": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Labels that MUST be present on results.",
                    },
                    "labels_preferred": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Labels that boost result score.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tetramem_build_pyramid",
            "description": "Build or rebuild the multi-scale resolution pyramid. Creates hierarchical clusters for fast coarse-to-fine queries.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tetramem_query_pyramid",
            "description": "Query using the resolution pyramid for fast multi-scale retrieval. Good for large memory stores.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query text.",
                    },
                    "k": {
                        "type": "integer",
                        "description": "Number of results (default 5).",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tetramem_zigzag_snapshot",
            "description": "Record a zigzag persistence snapshot to track topological feature evolution. Detects phase transitions, predicts emerging features.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tetramem_predict_topology",
            "description": "Predict upcoming topological changes based on zigzag persistence trends. Returns divergence/convergence prediction.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tetramem_get",
            "description": "Retrieve a specific memory by its ID. Returns content, labels, weight, and metadata.",
            "parameters": {
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "The ID of the memory to retrieve.",
                    },
                },
                "required": ["memory_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tetramem_update",
            "description": "Update an existing memory's content, labels, metadata, or weight.",
            "parameters": {
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "The ID of the memory to update.",
                    },
                    "content": {
                        "type": "string",
                        "description": "New content text (optional).",
                    },
                    "labels": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "New labels (optional, replaces all).",
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Metadata keys to add/update (optional, merges).",
                    },
                    "weight": {
                        "type": "number",
                        "description": "New weight (0.1-10.0, optional).",
                    },
                },
                "required": ["memory_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tetramem_query_by_label",
            "description": "Query memories by label/tag. Returns all memories matching the given label.",
            "parameters": {
                "type": "object",
                "properties": {
                    "label": {
                        "type": "string",
                        "description": "The label to search for.",
                    },
                    "k": {
                        "type": "integer",
                        "description": "Maximum results (default 20).",
                        "default": 20,
                    },
                },
                "required": ["label"],
            },
        },
    },
]


def get_tool_definitions() -> List[Dict[str, Any]]:
    return TOOL_DEFINITIONS


def execute_tool_call(
    tool_name: str,
    arguments: Dict[str, Any],
    memory: Optional[Any] = None,
) -> Dict[str, Any]:
    if memory is None:
        return {
            "error": "No memory instance provided. Pass an initialized GeoMemoryBody via the memory parameter."
        }

    if tool_name == "tetramem_store":
        content = arguments.get("content", "")
        labels = arguments.get("labels", [])
        weight = arguments.get("weight", 1.0)
        metadata = arguments.get("metadata")
        node_id = memory.store(content=content, labels=labels, weight=weight, metadata=metadata)
        return {"id": node_id, "stored": True}

    elif tool_name == "tetramem_query":
        query = arguments.get("query", "")
        k = arguments.get("k", 5)
        use_persistence = arguments.get("use_persistence", True)
        results = memory.query(query, k=k, use_persistence=use_persistence)
        return {
            "results": [
                {
                    "id": r.node.id,
                    "content": r.node.content,
                    "distance": r.distance,
                    "persistence_score": r.persistence_score,
                }
                for r in results
            ]
        }

    elif tool_name == "tetramem_associate":
        memory_id = arguments.get("memory_id", "")
        max_depth = arguments.get("max_depth", 2)
        results = memory.associate(memory_id=memory_id, max_depth=max_depth)
        return {
            "associations": [
                {
                    "id": node.id,
                    "content": node.content,
                    "score": score,
                    "type": assoc_type,
                }
                for node, score, assoc_type in results
            ]
        }

    elif tool_name == "tetramem_self_organize":
        stats = memory.self_organize()
        return {"stats": stats}

    elif tool_name == "tetramem_stats":
        return memory.get_statistics()

    elif tool_name == "tetramem_dream":
        if not memory._use_mesh:
            return {"status": "no_mesh"}
        from .tetra_dream import TetraDreamCycle

        dc = TetraDreamCycle(memory._mesh, cycle_interval=999999)
        return dc.trigger_now()

    elif tool_name == "tetramem_closed_loop":
        from .closed_loop import ClosedLoopEngine

        engine = ClosedLoopEngine(memory)
        context = arguments.get("context", "")
        k = arguments.get("k", 5)
        force_dream = arguments.get("force_dream", False)
        return engine.run_cycle(context=context, k=k, force_dream=force_dream)

    elif tool_name == "tetramem_weight_update":
        memory_id = arguments.get("memory_id", "")
        delta = arguments.get("delta", 0.0)
        use_ema = arguments.get("use_ema", True)
        alpha = arguments.get("alpha", 0.1)
        memory.update_weight(memory_id=memory_id, delta=delta, use_ema=use_ema, alpha=alpha)
        return {"memory_id": memory_id, "status": "ok"}

    elif tool_name == "tetramem_batch_store":
        items = arguments.get("items", [])
        ids = memory.store_batch(items)
        return {"ids": ids, "stored": len(ids)}

    elif tool_name == "tetramem_persist":
        memory.flush_persistence()
        return {"status": "ok", "bucket_id": memory._bucket_id}

    elif tool_name == "tetramem_query_multiparam":
        results = memory.query_multiparam(
            query_text=arguments.get("query", ""),
            k=arguments.get("k", 10),
            labels_required=arguments.get("labels_required"),
            labels_preferred=arguments.get("labels_preferred"),
        )
        return {
            "results": [
                {
                    "id": r.node.id,
                    "content": r.node.content,
                    "composite_score": r.persistence_score,
                    "association_type": r.association_type,
                    "labels": r.node.labels,
                }
                for r in results
            ]
        }

    elif tool_name == "tetramem_build_pyramid":
        return memory.build_pyramid()

    elif tool_name == "tetramem_query_pyramid":
        results = memory.query_pyramid(
            query_text=arguments.get("query", ""),
            k=arguments.get("k", 5),
        )
        return {
            "results": [
                {
                    "id": r.node.id,
                    "content": r.node.content,
                    "distance": r.distance,
                    "association_type": r.association_type,
                }
                for r in results
            ]
        }

    elif tool_name == "tetramem_zigzag_snapshot":
        result = memory.record_zigzag_snapshot()
        return result if result else {"status": "not_available"}

    elif tool_name == "tetramem_predict_topology":
        return memory.predict_topology()

    else:
        return {"error": f"Unknown tool: {tool_name}"}


def create_tool_response(
    tool_call_id: str,
    tool_name: str,
    arguments: Dict[str, Any],
    memory: Optional[Any] = None,
) -> Dict[str, Any]:
    result = execute_tool_call(tool_name, arguments, memory=memory)
    return {
        "tool_call_id": tool_call_id,
        "role": "tool",
        "content": json.dumps(result),
    }
