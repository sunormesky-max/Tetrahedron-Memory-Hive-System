#!/usr/bin/env python3
"""
TetraMem-XL MCP Server
=======================
Model Context Protocol server for TetraMem eternal memory system.

Supports two transport modes:
  stdio  — for Claude Desktop, Cursor, VS Code (local)
  sse    — for remote agents over HTTP

Usage:
  # stdio mode (add to claude_desktop_config.json):
  python tetramem_mcp_server.py --mode stdio --api http://localhost:8000

  # SSE mode (for remote agents):
  python tetramem_mcp_server.py --mode sse --api http://localhost:8000 --port 9000

Environment variables:
  TETRAMEM_API_URL  — API base URL (default: http://localhost:8000)
  TETRAMEM_API_KEY  — Optional API key for authentication
"""

import argparse
import json
import sys
import os
import asyncio
import logging
from typing import Any, Dict, List, Optional
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

logging.basicConfig(level=logging.INFO, format="[TetraMem-MCP] %(message)s")
log = logging.getLogger(__name__)

API_BASE = os.environ.get("TETRAMEM_API_URL", "http://localhost:8000")
API_KEY = os.environ.get("TETRAMEM_API_KEY", "")


def _api(method: str, path: str, body: Optional[dict] = None) -> Any:
    url = f"{API_BASE}{path}"
    data = json.dumps(body).encode() if body else None
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    req = Request(url, data=data, headers=headers, method=method)
    try:
        with urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.read().decode()[:200]}"}
    except URLError as e:
        return {"error": f"Connection failed: {e.reason}"}
    except Exception as e:
        return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════
# Tool definitions — mapped 1:1 from llm_tool.py + REST API
# ═══════════════════════════════════════════════════════════════

TOOLS = [
    {
        "name": "tetramem_store",
        "description": "Store a new eternal memory in the TetraMem BCC lattice honeycomb system. "
                       "Memories are permanent and will be integrated, never deleted. "
                       "Use for: decisions, discoveries, user preferences, code patterns, project context.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The memory content to store. Be specific and descriptive."
                },
                "labels": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tags for categorization. e.g. ['decision', 'architecture', 'project-x']"
                },
                "weight": {
                    "type": "number",
                    "description": "Importance: 1.0=normal, 1.5=important, 2.0=critical (BOSS info, passwords). Default 1.0.",
                    "default": 1.0
                }
            },
            "required": ["content"]
        }
    },
    {
        "name": "tetramem_query",
        "description": "Search memories by semantic text similarity using BCC lattice topology + PCNN pulse propagation. "
                       "Returns the k most relevant memories with distances and weights. "
                       "Use for: recalling past decisions, finding related code patterns, checking prior context.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language query text."
                },
                "k": {
                    "type": "integer",
                    "description": "Number of results (default 5, max 100).",
                    "default": 5
                },
                "labels": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Filter to only memories with these labels."
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "tetramem_associate",
        "description": "Find memories associated with a specific memory through multi-layer topology: "
                       "face/edge/vertex adjacency, Hebbian pathways, and crystallized fast-paths.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "memory_id": {
                    "type": "string",
                    "description": "The memory ID to find associations for."
                },
                "max_depth": {
                    "type": "integer",
                    "description": "Association traversal depth (1-5, default 2).",
                    "default": 2
                }
            },
            "required": ["memory_id"]
        }
    },
    {
        "name": "tetramem_get",
        "description": "Retrieve a specific memory by its exact ID. Returns full content, labels, weight, and metadata.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "memory_id": {
                    "type": "string",
                    "description": "The memory ID to retrieve."
                }
            },
            "required": ["memory_id"]
        }
    },
    {
        "name": "tetramem_query_by_label",
        "description": "Query all memories matching a specific label/tag. Returns up to k results.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "label": {
                    "type": "string",
                    "description": "The label to search for."
                },
                "k": {
                    "type": "integer",
                    "description": "Maximum results (default 20).",
                    "default": 20
                }
            },
            "required": ["label"]
        }
    },
    {
        "name": "tetramem_stats",
        "description": "Get comprehensive statistics: node counts, PCNN pulse state, Hebbian paths, "
                       "crystallized pathways, honeycomb cell quality, dream/bridge counts, lattice integrity.",
        "inputSchema": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "tetramem_health",
        "description": "Quick health check. Returns status and version.",
        "inputSchema": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "tetramem_dream",
        "description": "Trigger a dream cycle: PCNN pulse-weighted random walk discovers topologically related memories "
                       "and synthesizes new cross-domain insights. No memories are deleted. "
                       "Use for: creative problem solving, discovering hidden connections.",
        "inputSchema": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "tetramem_self_organize",
        "description": "Trigger self-organization: geometric surgery (edge contractions, repulsions, cave growths) "
                       "to optimize memory topology. Creates shortcuts between frequently co-activated memories.",
        "inputSchema": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "tetramem_cascade",
        "description": "Trigger a PCNN cascade pulse from the highest-energy memories. "
                       "Propagates activation through face/edge/vertex connections.",
        "inputSchema": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "tetramem_export",
        "description": "Export all memories as JSON. Includes content, labels, weights, and full metadata.",
        "inputSchema": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "tetramem_timeline",
        "description": "Browse memories in chronological order (newest or oldest first).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "direction": {
                    "type": "string",
                    "enum": ["newest", "oldest"],
                    "description": "Sort direction (default 'newest').",
                    "default": "newest"
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results (default 20).",
                    "default": 20
                }
            }
        }
    },
    {
        "name": "tetramem_batch_store",
        "description": "Store multiple memories at once. Efficient for importing context or bulk operations.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string"},
                            "labels": {"type": "array", "items": {"type": "string"}},
                            "weight": {"type": "number", "default": 1.0}
                        },
                        "required": ["content"]
                    },
                    "description": "Array of memory items to store."
                }
            },
            "required": ["items"]
        }
    },
    {
        "name": "tetramem_query_multiparam",
        "description": "Advanced filtered query: combine text similarity with label filtering, weight thresholds, "
                       "and temporal recency. More powerful than basic query.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Query text."
                },
                "k": {
                    "type": "integer",
                    "description": "Number of results (default 10).",
                    "default": 10
                },
                "labels_required": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Labels that MUST be present on results."
                },
                "labels_preferred": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Labels that boost result score."
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "tetramem_weight_update",
        "description": "Adjust the importance weight of an existing memory. Supports delta adjustment and EMA smoothing.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "memory_id": {
                    "type": "string",
                    "description": "The memory ID."
                },
                "delta": {
                    "type": "number",
                    "description": "Weight change amount (positive=increase, negative=decrease)."
                },
                "use_ema": {
                    "type": "boolean",
                    "description": "Use exponential moving average (default true).",
                    "default": True
                }
            },
            "required": ["memory_id", "delta"]
        }
    },
]


# ═══════════════════════════════════════════════════════════════
# Tool execution — maps tool calls to REST API requests
# ═══════════════════════════════════════════════════════════════

def execute_tool(name: str, args: Dict[str, Any]) -> Any:
    dispatch = {
        "tetramem_store": lambda: _api("POST", "/api/v1/store", {
            "content": args["content"],
            "labels": args.get("labels", []),
            "weight": args.get("weight", 1.0)
        }),
        "tetramem_query": lambda: _api("POST", "/api/v1/query", {
            "query": args["query"],
            "k": args.get("k", 5),
            "labels": args.get("labels")
        }),
        "tetramem_associate": lambda: _api("POST", "/api/v1/associate", {
            "tetra_id": args["memory_id"],
            "max_depth": args.get("max_depth", 2)
        }),
        "tetramem_get": lambda: _api("GET", f"/api/v1/tetrahedra/{args['memory_id']}"),
        "tetramem_query_by_label": lambda: _api("POST", "/api/v1/query-by-label", {
            "label": args["label"],
            "k": args.get("k", 20)
        }),
        "tetramem_stats": lambda: _api("GET", "/api/v1/stats"),
        "tetramem_health": lambda: _api("GET", "/api/v1/health"),
        "tetramem_dream": lambda: _api("POST", "/api/v1/dream"),
        "tetramem_self_organize": lambda: _api("POST", "/api/v1/self-organize/run"),
        "tetramem_cascade": lambda: _api("POST", "/api/v1/cascade/trigger"),
        "tetramem_export": lambda: _api("GET", "/api/v1/export"),
        "tetramem_timeline": lambda: _api("POST", "/api/v1/timeline", {
            "direction": args.get("direction", "newest"),
            "limit": args.get("limit", 20)
        }),
        "tetramem_batch_store": lambda: _api("POST", "/api/v1/batch-store", args),
        "tetramem_query_multiparam": lambda: _api("POST", "/api/v1/query-multiparam", {
            "query": args["query"],
            "k": args.get("k", 10),
            "labels_required": args.get("labels_required"),
            "labels_preferred": args.get("labels_preferred")
        }),
        "tetramem_weight_update": lambda: _api("POST", "/api/v1/weight-update", {
            "memory_id": args["memory_id"],
            "delta": args["delta"],
            "use_ema": args.get("use_ema", True)
        }),
    }
    handler = dispatch.get(name)
    if not handler:
        return {"error": f"Unknown tool: {name}"}
    return handler()


# ═══════════════════════════════════════════════════════════════
# JSON-RPC 2.0 protocol handler (stdio mode)
# ═══════════════════════════════════════════════════════════════

def _jsonrpc_response(id: Any, result: Any) -> str:
    return json.dumps({"jsonrpc": "2.0", "id": id, "result": result}) + "\n"


def _jsonrpc_error(id: Any, code: int, message: str) -> str:
    return json.dumps({"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}}) + "\n"


def _handle_request(req: Dict[str, Any]) -> str:
    req_id = req.get("id")
    method = req.get("method", "")
    params = req.get("params", {})

    if method == "initialize":
        return _jsonrpc_response(req_id, {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {"listChanged": False},
                "resources": {"subscribe": False, "listChanged": False}
            },
            "serverInfo": {
                "name": "tetramem-x",
                "version": "6.5.0",
                "description": "TetraMem-XL Eternal Memory — BCC Lattice Honeycomb + PCNN"
            }
        })

    elif method == "notifications/initialized":
        return ""

    elif method == "tools/list":
        return _jsonrpc_response(req_id, {"tools": TOOLS})

    elif method == "tools/call":
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})
        log.info(f"Tool call: {tool_name}")
        result = execute_tool(tool_name, arguments)
        if isinstance(result, dict) and "error" in result:
            return _jsonrpc_response(req_id, {
                "content": [{"type": "text", "text": json.dumps(result)}],
                "isError": True
            })
        return _jsonrpc_response(req_id, {
            "content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}],
            "isError": False
        })

    elif method == "resources/list":
        return _jsonrpc_response(req_id, {"resources": []})

    elif method == "ping":
        return _jsonrpc_response(req_id, {})

    else:
        return _jsonrpc_error(req_id, -32601, f"Method not found: {method}")


def run_stdio():
    log.info("Starting TetraMem MCP Server (stdio mode)...")
    log.info(f"API: {API_BASE}")
    health = _api("GET", "/api/v1/health")
    if "error" in health:
        log.warning(f"API health check failed: {health['error']}")
    else:
        log.info(f"Connected to TetraMem v{health.get('version', '?')}")

    buffer = ""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        buffer += line
        try:
            req = json.loads(buffer)
            buffer = ""
            response = _handle_request(req)
            if response:
                sys.stdout.write(response)
                sys.stdout.flush()
        except json.JSONDecodeError:
            if line.endswith("}"):
                buffer = ""


# ═══════════════════════════════════════════════════════════════
# HTTP SSE mode (for remote agents)
# ═══════════════════════════════════════════════════════════════

def run_sse(host: str = "0.0.0.0", port: int = 9000):
    try:
        from http.server import HTTPServer, BaseHTTPRequestHandler
    except ImportError:
        log.error("HTTP server not available")
        return

    class MCPSSEHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode()
            try:
                req = json.loads(body)
            except json.JSONDecodeError:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Invalid JSON"}).encode())
                return

            response = _handle_request(req)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            if response:
                self.wfile.write(response.strip().encode())

        def do_OPTIONS(self):
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
            self.end_headers()

        def log_message(self, format, *args):
            log.info(f"HTTP: {args[0] if args else ''}")

    server = HTTPServer((host, port), MCPSSEHandler)
    log.info(f"TetraMem MCP Server (SSE mode) listening on {host}:{port}")
    log.info(f"API backend: {API_BASE}")
    health = _api("GET", "/api/v1/health")
    if "error" in health:
        log.warning(f"API health check failed: {health['error']}")
    else:
        log.info(f"Connected to TetraMem v{health.get('version', '?')}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Shutting down...")
        server.shutdown()


# ═══════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TetraMem-XL MCP Server")
    parser.add_argument("--mode", choices=["stdio", "sse"], default="stdio",
                        help="Transport mode: stdio (local) or sse (remote HTTP)")
    parser.add_argument("--api", default=None,
                        help="TetraMem API base URL (default: TETRAMEM_API_URL or http://localhost:8000)")
    parser.add_argument("--host", default="0.0.0.0", help="SSE host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=9000, help="SSE port (default: 9000)")
    args = parser.parse_args()

    if args.api:
        API_BASE = args.api.rstrip("/")

    if args.mode == "stdio":
        run_stdio()
    else:
        run_sse(host=args.host, port=args.port)
