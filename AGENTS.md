# TetraMem-XL v7.1 — Agent Integration Guide

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
  -H "X-API-Key: your-key" \
  -d '{"content": "Remember this", "labels": ["important"], "weight": 1.0}'
# Returns: {"id": "..."}
```

### Query memories
```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-key" \
  -d '{"query": "remember", "k": 5}'
# Returns: {"results": [{"id":"...", "content":"...", "distance":0.85, ...}]}
```

### Get all stats
```bash
curl http://localhost:8000/api/v1/stats -H "X-API-Key: your-key"
```

### Health check
```bash
curl http://localhost:8000/api/v1/health
# Returns: {"status":"ok","version":"7.0.0"}
```

### List memories
```bash
curl "http://localhost:8000/api/v1/browse?direction=newest&limit=20" -H "X-API-Key: your-key"
```

### Trigger self-organization
```bash
curl -X POST http://localhost:8000/api/v1/self-organize -H "X-API-Key: your-key"
```

### Trigger dream cycle
```bash
curl -X POST http://localhost:8000/api/v1/dream -H "X-API-Key: your-key"
```

### Trigger cascade pulse
```bash
curl -X POST http://localhost:8000/api/v1/cascade/trigger -H "X-API-Key: your-key"
```

### Dark plane flow
```bash
curl -X POST http://localhost:8000/api/v1/dark-plane/flow -H "X-API-Key: your-key"
```

### Self-regulation trigger
```bash
curl -X POST http://localhost:8000/api/v1/regulation/trigger -H "X-API-Key: your-key"
```

### Export memories
```bash
curl http://localhost:8000/api/v1/export -H "X-API-Key: your-key"
```

### Import memories
```bash
curl -X POST http://localhost:8000/api/v1/import \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-key" \
  -d '{"memories": [{"content": "...", "labels": [...], "weight": 1.0}]}'
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
- **Dark Plane**: Thermodynamic energy landscape with adaptive thresholds
- **Self-Regulation**: Six-layer physiological control (PID + circadian + autonomic + immune + endocrine + stress)
- **Spatial Reflection Field**: Energy-based phase transition detection

## UI

- Embedded: `http://<host>:8000/ui/`
- Standalone dashboard: Open `ui/dashboard.html` in any browser, enter API address

## Management

```bash
systemctl restart tetramem-api   # Restart
systemctl stop tetramem-api      # Stop
journalctl -u tetramem-api -f    # View logs
```
