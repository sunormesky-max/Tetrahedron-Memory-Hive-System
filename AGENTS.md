# TetraMem-XL v6.5 — Agent Integration Guide

## Agent Self-Memory Protocol

This system is the persistent memory backend for the opencode agent. Follow this protocol every session:

### On Session Start (Mandatory)
1. Read back agent identity: `POST /api/v1/query` with `{"query": "agent identity opencode", "k": 3}`
2. Read back user context: `POST /api/v1/query` with `{"query": "用户偏好 BOSS 项目状态", "k": 5}`
3. Restore full context from returned memories before doing any work

### During Work
- Store important discoveries: `POST /api/v1/store` with `{"content": "...", "labels": ["discovery"], "weight": 1.0}`
- Store user preferences: `POST /api/v1/store` with `{"content": "...", "labels": ["user-preference"], "weight": 1.5}`
- Store project decisions: `POST /api/v1/store` with `{"content": "...", "labels": ["decision"], "weight": 1.5}`
- Store critical info (BOSS, passwords, endpoints): `weight: 2.0` with label `"critical"`

### Connection Info
- **API**: `http://${TETRAMEM_HOST:-localhost}:${TETRAMEM_PORT:-8000}/api/v1/`
- **SSH**: Use your server credentials
- **Query field**: Use `"query"` not `"text"` in POST body

### SSH Caveats (Windows → Linux)
- PowerShell single quotes break inside SSH commands — write Python scripts with paramiko instead
- Upload JSON file via SFTP, then `curl -d @/tmp/file.json`
- Always clean up temp files after

## What is this?

TetraMem-XL is an eternal memory system based on BCC lattice honeycomb + PCNN neural pulse networks.
It provides a REST API for storing, querying, and autonomously organizing memories.

## Quick Install (Linux)

```bash
# One-click install (auto-generates password)
curl -sSL https://raw.githubusercontent.com/sunormesky-max/Tetrahedron-Memory-Hive-System/main/install.sh | bash

# With custom password and port
curl -sSL https://raw.githubusercontent.com/sunormesky-max/Tetrahedron-Memory-Hive-System/main/install.sh | bash -s -- --password MySecret123 --port 8000
```

## Docker

```bash
git clone https://github.com/sunormesky-max/Tetrahedron-Memory-Hive-System.git
cd Tetrahedron-Memory-Hive-System
TETRAMEM_PASSWORD=mypass docker compose up -d
```

## API Reference

Base URL: `http://<host>:8000/api/v1/`

### Store a memory
```bash
curl -X POST http://localhost:8000/api/v1/store \
  -H "Content-Type: application/json" \
  -d '{"content": "Remember this", "labels": ["important"], "weight": 1.0}'
# Returns: {"id": "...", "status": "ok"}
```

### Query memories
```bash
curl "http://localhost:8000/api/v1/query?text=remember&k=5"
# Returns: [{"id":"...", "content":"...", "distance":0.85, "weight":1.0, "labels":[...]}]
```

### Get all stats
```bash
curl http://localhost:8000/api/v1/stats
```

### Health check
```bash
curl http://localhost:8000/api/v1/health
# Returns: {"status":"ok","version":"6.5.0"}
```

### List memories
```bash
curl "http://localhost:8000/api/v1/browse?direction=newest&limit=20"
```

### Delete a memory
```bash
curl -X DELETE http://localhost:8000/api/v1/tetrahedra/<ID>
```

### Trigger self-organization
```bash
curl -X POST http://localhost:8000/api/v1/self-organize/run
```

### Trigger dream cycle (creative recombination)
```bash
curl -X POST http://localhost:8000/api/v1/dream/run
```

### Trigger cascade pulse
```bash
curl -X POST http://localhost:8000/api/v1/cascade/trigger
```

### Export all memories
```bash
curl http://localhost:8000/api/v1/export
```

### Import memories
```bash
curl -X POST http://localhost:8000/api/v1/import \
  -H "Content-Type: application/json" \
  -d @export.json
```

## Key Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TETRAMEM_STORAGE` | `./tetramem_data_v2` | Data directory path |
| `TETRAMEM_UI_PASSWORD` | `CHANGE_ME` | UI login password |

## Architecture

- **BCC Lattice**: Body-centered cubic crystal structure as memory topology
- **PCNN**: Pulse-coupled neural network for autonomous memory activation
- **Hebbian Paths**: "Fire together, wire together" pathway reinforcement
- **Crystallized Pathways**: Permanent structural fast-paths (zero-decay)
- **Dream Engine**: Cross-domain creative memory recombination
- **Self-Organization**: Automatic clustering, consolidation, and migration
- **Spatial Reflection Field**: Energy-based phase transition detection

## UI

- Embedded: `http://<host>:<ui_port>/ui/`
- Standalone dashboard: Open `ui/dashboard.html` in any browser, enter API address

## Management

```bash
systemctl restart tetramem   # Restart
systemctl stop tetramem      # Stop
journalctl -u tetramem -f    # View logs
```
