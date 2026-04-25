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

## Agent 必须使用说明 — Observer 接入指南

### 你是谁

你是接入 TetraMem-XL 记忆系统的 AI Agent。记忆系统有一个**自省观察层（RuntimeObserver）**，它像一面镜子——让记忆系统知道"自己做了什么"。但这面镜子需要**你的日志**才能照出内容。

### 你必须做的事

#### 方式一：让记忆系统 tail 你的日志文件（推荐）

如果你的运行日志写到文件（如 `/var/log/my-agent.log`）：

```python
from tetrahedron_memory import HoneycombNeuralField
from tetrahedron_memory.runtime_observer import attach_file_observer

field = HoneycombNeuralField(resolution=5)
field.initialize()

# 指向你的日志文件，自动捕获并蒸馏
observer = attach_file_observer(field, "/var/log/my-agent.log")
```

#### 方式二：程序内直接注入

如果你在代码内运行，没有日志文件：

```python
from tetrahedron_memory.runtime_observer import attach_callback_observer, LogEvent
import time

observer = attach_callback_observer(field)

# 在关键点注入事件
observer.ingest(LogEvent(time.time(), "INFO", "my_agent", "开始推理"))
observer.ingest(LogEvent(time.time(), "ERROR", "my_agent", "外部API超时"))
observer.ingest(LogEvent(time.time(), "INFO", "my_agent", "完成规划"))
```

#### 方式三：零接触自动挂载（最简单）

```python
from tetrahedron_memory.observer_config import auto_attach

# 首次调用自动生成 ./observer_config.json 模板
# 修改其中的 log_source.path 指向你的日志文件
# 再次调用即可生效
observer = auto_attach(field, config_path="./observer_config.json")
```

### 配置文件说明

`auto_attach()` 首次运行会生成 `observer_config.json`，你必须修改以下字段：

```json
{
  "enabled": true,
  "window_seconds": 300,
  "max_stores_per_minute": 30,
  "log_sources": {
    "python_logging": {
      "enabled": true,
      "level": "WARNING",
      "logger_name": "tetramem"
    },
    "file_tail": [
      {
        "path": "/var/log/YOUR_AGENT_LOG_FILE.log",
        "poll_interval": 1.0
      }
    ]
  },
  "rules": [
    {"category": "error", "level_filter": ["ERROR", "CRITICAL"], "weight": 2.0, "immediate": true},
    {"category": "anomaly", "pattern": "(timeout|crash|slow)", "weight": 1.5, "immediate": true}
  ]
}
```

**必须修改的字段：**
- `log_sources.file_tail[0].path` → 改成你自己的日志文件路径
- `rules` → 按需增删规则（匹配你自己日志格式的关键模式）

**可选环境变量覆盖：**

| 变量 | 作用 |
|------|------|
| `TETRAMEM_OBSERVER_ENABLED` | `false` 禁用观察器 |
| `TETRAMEM_OBSERVER_WINDOW` | 聚合窗口秒数（默认 300） |
| `TETRAMEM_OBSERVER_MAX_STORES` | 每分钟最大存储数（默认 30） |
| `TETRAMEM_OBSERVER_LOG_PATH` | 日志文件路径 |

### 日志格式要求

LogFileTailer 默认匹配以下格式：

```
2025-01-01 12:00:00 ERROR my_module 错误消息内容
2025-01-01 12:00:01 INFO my_module 正常消息
```

即：`时间戳 LEVEL 模块名 消息内容`

如果你的日志格式不同，可以在配置中自定义 `pattern`（正则表达式），或者用 `ingest()` 直接注入。

### 日志文件支持的日志格式

标准格式（默认自动解析）：
```
2025-01-01 12:00:00 ERROR my_app Connection timeout
2025-01-01 12:00:01 WARNING my_app Slow response: 5000ms
2025-01-01 12:00:02 INFO my_agent Planning completed
```

非标准格式（未匹配则整行作为 INFO 消息存入，模块名标记为 `external`）。

### 完整接入检查清单

```
□ 1. 安装 TetraMem-XL v8.0+ 并启动 API
□ 2. 确认 observer 正常：
     curl http://localhost:8000/api/v1/observer/stats
     → "enabled": true
□ 3. 选择接入方式：
     □ 文件 tail → attach_file_observer(field, "你的日志路径")
     □ 程序注入 → attach_callback_observer(field) + ingest()
     □ 零接触 → auto_attach(field) + 修改 observer_config.json
□ 4. 运行一段时间后再次检查 stats：
     → "memories_stored" > 0（系统已开始记录你的轨迹）
     → "events_dropped_loop" 未疯长（循环隔离正常）
□ 5. 查询自省记忆验证：
     curl -X POST http://localhost:8000/api/v1/query \
       -d '{"query": "trajectory", "k": 5}'
     → 应看到 source="self-observation" 的记忆
```

### 你不需要做的

- 不需要手动调用 store 来存运行日志 — observer 自动处理
- 不需要担心循环 — observer 自己产生的日志会被自动丢弃
- 不需要担心隐私 — api_key/password/token 自动脱敏
- 不需要担心性能 — 速率硬上限 30/min，溢出自动丢弃低权重

### 三种接入方式对比

| | 文件 tail | 程序注入 | 零接触 auto_attach |
|---|---|---|---|
| **适用场景** | 外部 AI，日志写文件 | 内嵌 Agent，无日志文件 | 快速原型，不确定选哪个 |
| **接入代码** | 1 行 | 2 行 | 1 行 + 改配置 |
| **延迟** | ~1s (poll 间隔) | 实时 | 取决于配置 |
| **配置** | 代码内 | 代码内 | JSON 文件 |
| **推荐度** | ★★★ 生产环境 | ★★★ 内嵌场景 | ★★★ 不确定场景 |
