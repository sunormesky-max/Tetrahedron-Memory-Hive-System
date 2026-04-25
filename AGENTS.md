# TetraMem-XL v8.0 — Agent Integration Guide

## What is this?

TetraMem-XL is an eternal memory system based on BCC lattice honeycomb + PCNN neural pulse networks.
It provides a REST API for storing, querying, and autonomously organizing memories.

**v8.0 核心新能力: RuntimeObserver** — 记忆系统可以观察自身运行轨迹，自动将运行日志蒸馏为低权重"自省记忆"，由暗位面引擎自然整合。安装后必须确认 Observer 正常运行（见下方 §Observer）。

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
# Returns: {"status":"ok","version":"8.0.0"}
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

### Observer stats (v8.0 — 安装后必须检查)
```bash
curl http://localhost:8000/api/v1/observer/stats -H "X-API-Key: your-key"
# Must see: "enabled": true, "memories_stored" slowly growing
```

### Observer flush
```bash
curl -X POST http://localhost:8000/api/v1/observer/flush -H "X-API-Key: your-key"
```

### Observer enable/disable
```bash
curl -X POST http://localhost:8000/api/v1/observer/enable -H "X-API-Key: your-key"
curl -X POST http://localhost:8000/api/v1/observer/disable -H "X-API-Key: your-key"
```

### Observer manual observe
```bash
curl -X POST http://localhost:8000/api/v1/observer/observe \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-key" \
  -d '{"level": "ERROR", "module": "my-agent", "message": "Connection failed"}'
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
- **Runtime Observer** (v8.0): Self-observation layer — captures runtime logs, classifies by semantics, aggregates into trajectory memories, auto-stores into dark plane

## UI

- Embedded: `http://<host>:8000/ui/`
- Standalone dashboard: Open `ui/dashboard.html` in any browser, enter API address

## Management

```bash
systemctl restart tetramem-api   # Restart
systemctl stop tetramem-api      # Stop
journalctl -u tetramem-api -f    # View logs
```

## Observer (v8.0) — 安装后必须检查

RuntimeObserver 在 `initialize()` 时自动启动。它捕获 `tetramem` logger 的 WARNING+ 日志，经语义分类、聚合窗口、轨迹叙述后，以低权重存入晶格。暗位面引擎在 `dark_plane_flow()` 中自动拾取这些轨迹记忆并整合。

### 数据流

```
日志 → 语义分类(6类) → 聚合窗口(300s) → 轨迹叙述 → 速率控制(30/min) → store(低权重)
                                                                           ↓
                                                                    暗位面自动拾取整合
```

### 分类与权重

| 分类 | 触发条件 | 权重 | 暗位面层级 |
|------|----------|------|-----------|
| error | ERROR/CRITICAL | 2.0 | deep/abyss（创伤记忆） |
| anomaly | timeout/crash/slow... | 1.8 | deep/abyss |
| system | WARNING | 0.8 | shallow/deep |
| performance | latency/throughput | 0.5 | surface/shallow |
| behavior | store/query/dream... | 0.3 | surface（行为轨迹） |
| noise | DEBUG/heartbeat | 0.0 | 丢弃 |

### 安全阀

- **循环隔离**: `source="self-observation"` 的日志自动丢弃
- **噪声过滤**: DEBUG + heartbeat 直接丢弃
- **速率硬上限**: 30 stores/min
- **隐私脱敏**: api_key/password/token/Bearer → `[REDACTED]`

### 安装后必须确认

```bash
curl http://localhost:8000/api/v1/observer/stats -H "X-API-Key: your-key"
```

检查项：
1. `"enabled": true` — 观察器已启动
2. `"memories_stored"` 缓慢增长 — 系统在记忆自己的轨迹
3. `"events_dropped_loop"` 不在疯长 — 循环隔离正常
