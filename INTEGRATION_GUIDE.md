# TetraMem-XL v8.0 集成与配置指南

本文档记录了 TetraMem-XL 与 AI Agent 框架（如 OpenClaw）集成的完整过程，供其他 Agent 复现配置。

---

## 1. 系统架构总览

```
┌─────────────────────────────────────────────────────┐
│                   AI Agent 框架                      │
│            (OpenClaw / 自定义 Agent)                  │
│                                                      │
│  ┌──────────┐  ┌──────────┐  ┌───────────────────┐  │
│  │ MCP 工具  │  │ SSE 订阅  │  │ 原生记忆后端适配    │  │
│  │ (22个)    │  │ (事件推送) │  │ (tetramem-manager) │  │
│  └────┬─────┘  └────┬─────┘  └────────┬──────────┘  │
└───────┼──────────────┼─────────────────┼─────────────┘
        │              │                 │
        ▼              ▼                 ▼
┌─────────────────────────────────────────────────────┐
│              FastAPI 网关层 (v8.0)                    │
│              AppState 依赖注入容器                     │
│                                                      │
│  routers/memory   — store, query, browse, export     │
│  routers/agent    — 上下文/推理/建议/导航              │
│  routers/system   — health, stats, login, config     │
│  routers/neural   — dream, pulse, cascade, crystal   │
│  routers/spatial  — lattice, honeycomb, integrity    │
  │  routers/darkplane— 暗位面, 调节, 注意力              │
  │  routers/observer — 运行观察器, 自省轨迹              │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│         HoneycombNeuralField (PCNN 核心)              │
│                                                      │
│  DarkPlaneEngine       — 热力学暗位面引擎             │
│    ├ 自适应分位数阈值    (25th/50th/75th percentile)  │
│    ├ 玻尔兹曼重分布      (5% 热跃迁概率/周期)         │
│    ├ WKB 隧道穿透        (方向修正)                   │
│    └ 查询能量注入        (命中 deep/abyss 节点)       │
│                                                      │
│  SelfRegulationEngine  — 六层自我调节                 │
│    ├ 5 个 PID 控制器     (抗饱和积分限幅)             │
│    ├ 迟滞带              (自主神经 0.35/0.65)         │
│    ├ 暗位面反馈          (跃迁→内分泌反应)            │
│    └ 内分泌激素系统      (dopamine/cortisol/...)     │
│                                                      │
│  AgentMemoryDriver     — 上下文组装 + 推理链 + 建议    │
│  FeedbackLoop          — Agent 决策学习闭环            │
│  SessionManager        — 临时/永久记忆生命周期管理      │
│  DreamEngine           — 跨域创意记忆重组             │
│  SelfCheckEngine       — 自检脉冲引擎                │
│  SelfOrganizeEngine    — 拓扑自组织                  │
│  CrystallizedPathway   — 结晶化永久通路               │
│  LatticeIntegrityChecker — BCC晶格完整性验证           │
│  InsightAggregator     — 洞察聚合与事件推送           │
│  RuntimeObserver       — 自省观察层（v8.0）            │
│    ├ 语义分类器          error/anomaly/behavior/noise  │
│    ├ 聚合窗口            300s 滑动窗口 + 去重合并       │
│    ├ 轨迹叙述器          日志→可记忆结构压缩            │
│    ├ 速率硬上限          30 stores/min, 溢出丢低权      │
│    └ 循环隔离            自动丢弃 observer 自身日志     │
└─────────────────────────────────────────────────────────┘
```

---

## 2. 安装后必须配置（v8.0 新增）

### 2.1 RuntimeObserver — 自省观察层

**这是 v8.0 的核心能力。** 安装后必须确认以下配置，否则记忆系统无法观察自身运行轨迹。

#### 工作原理

```
Agent 运行日志 → TetraMemLogHandler 自动捕获
                    ↓
              语义分类器（6 类）
              ├ error     → 立即存储, 权重 2.0
              ├ anomaly   → 立即存储, 权重 1.8
              ├ behavior  → 聚合窗口, 权重 0.3
              ├ performance → 聚合窗口, 权重 0.5
              ├ system    → 聚合窗口, 权重 0.8
              └ noise     → 丢弃
                    ↓
              聚合窗口（300s）+ 去重合并
                    ↓
              轨迹叙述器（日志→可记忆结构）
                    ↓
              速率控制（30 stores/min 硬上限）
                    ↓
              store(低权重, source="self-observation")
                    ↓
              暗位面引擎自动拾取
              ├ 权重 0.2~0.3 → surface/shallow 层（行为轨迹）
              ├ 权重 0.8     → shallow/deep 层（系统警告）
              └ 权重 1.5~2.0 → deep/abyss 层（创伤记忆）
```

#### 三重安全阀

| 安全阀 | 作用 |
|--------|------|
| **循环隔离** | `source="self-observation"` 的日志自动丢弃，防止无限递归 |
| **噪声过滤** | DEBUG 级别 + heartbeat/health-check 直接丢弃 |
| **速率硬上限** | 30 stores/min，溢出按权重丢弃最低的 |

#### 隐私脱敏

以下模式在存储前自动替换为 `[REDACTED]`：
- `api_key=...` / `X-API-Key: ...`
- `password=...` / `token=...` / `secret=...`
- `Bearer ...`

#### 自动行为

RuntimeObserver 在 `initialize()` 时自动启动，无需手动配置。具体：
1. **自动捕获**：挂载到 `tetramem` Python logger，所有 WARNING+ 级别日志自动进入观察管线
2. **定时 flush**：后台线程每 30s 检查过期聚合窗口，自动存储
3. **ERROR/CRITICAL**：立即入队存储，不等聚合窗口
4. **自动进入暗位面**：低权重轨迹由暗位面引擎在 `dark_plane_flow()` 中自然拾取和整合

#### 确认 Observer 正常运行

```bash
# 检查 observer 状态
curl http://localhost:8000/api/v1/observer/stats -H "X-API-Key: your-key"
# 返回示例:
# {
#   "enabled": true,
#   "total_events_received": 142,
#   "memories_stored": 8,
#   "events_dropped_noise": 95,
#   "events_dropped_loop": 0,
#   "events_dropped_rate": 0,
#   "active_windows": {},
#   "uptime_seconds": 3600.0,
#   "category_counts": {"error": 3, "behavior": 35, "system": 9},
#   "config": {
#     "window_seconds": 300,
#     "max_stores_per_minute": 30,
#     "queue_max_size": 200
#   }
# }
```

**必须确认：**
- `enabled` = `true`
- `events_dropped_loop` 不在疯狂增长（否则循环隔离失效）
- `memories_stored` 在缓慢增长（系统在记忆自己的轨迹）

#### 临时禁用（如需调试）

```bash
# 禁用
curl -X POST http://localhost:8000/api/v1/observer/disable -H "X-API-Key: your-key"

# 重新启用
curl -X POST http://localhost:8000/api/v1/observer/enable -H "X-API-Key: your-key"
```

#### 手动注入观察事件（可选）

```bash
curl -X POST http://localhost:8000/api/v1/observer/observe \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-key" \
  -d '{"level": "ERROR", "module": "my-agent", "message": "Failed to connect to external service"}'
# → {"accepted": true}
```

#### 手动 flush 聚合窗口（可选）

```bash
curl -X POST http://localhost:8000/api/v1/observer/flush -H "X-API-Key: your-key"
# → {"flushed": 3}
```

#### Observer 轨迹记忆的标签

所有 observer 存入的记忆自动带有以下标签：
- `self-observation` — 标识来源
- `meta` — 元认知标记
- `trajectory` — 轨迹类型
- `error` / `anomaly` / `behavior` / `performance` / `system` — 具体分类

在 query 时可以用这些标签过滤：
```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-key" \
  -d '{"query": "trajectory error", "k": 10}'
```

#### Agent 接入 Observer 的三种方式

**方式一：文件 tail（推荐 — 外部 AI，日志写文件）**

```python
from tetrahedron_memory.runtime_observer import attach_file_observer

observer = attach_file_observer(field, "/var/log/your-agent.log")
```

**方式二：程序注入（内嵌 Agent，无日志文件）**

```python
from tetrahedron_memory.runtime_observer import attach_callback_observer, LogEvent
import time

observer = attach_callback_observer(field)
observer.ingest(LogEvent(time.time(), "ERROR", "my_agent", "Planning failed"))
```

**方式三：零接触自动挂载（最简单）**

```python
from tetrahedron_memory.observer_config import auto_attach

# 首次调用生成 ./observer_config.json 模板
observer = auto_attach(field, config_path="./observer_config.json")
```

首次运行后，编辑 `observer_config.json`：
- `log_sources.file_tail[0].path` → 改成你自己的日志文件路径
- `rules` → 按需增删规则
- 重启或重新调用 `auto_attach()` 生效

**环境变量覆盖：**

| 变量 | 作用 |
|------|------|
| `TETRAMEM_OBSERVER_ENABLED` | `false` 禁用 |
| `TETRAMEM_OBSERVER_WINDOW` | 聚合窗口秒数 |
| `TETRAMEM_OBSERVER_MAX_STORES` | 每分钟最大存储数 |
| `TETRAMEM_OBSERVER_LOG_PATH` | 日志文件路径 |

**日志格式：** 默认匹配 `YYYY-MM-DD HH:MM:SS LEVEL module message`。非标准格式整行作为 INFO 存入。

**接入检查清单：**

```
□ 安装 TetraMem-XL v8.0+ 并启动 API
□ curl .../observer/stats → "enabled": true
□ 选择接入方式并配置日志源
□ 运行一段时间 → "memories_stored" > 0
□ query "trajectory" → 看到 self-observation 标签的记忆
```

### 3.1 一键安装（推荐）

```bash
curl -sSL https://raw.githubusercontent.com/sunormesky-max/Tetrahedron-Memory-Hive-System/main/install.sh | bash
# 自定义密码和端口:
# curl -sSL ... | bash -s -- --password MySecret123 --port 8000
```

### 3.2 环境要求

| 依赖 | 版本 | 说明 |
|------|------|------|
| Python | 3.8+ | API 服务器 |
| numpy | any | 数值计算 |
| FastAPI + uvicorn | any | REST API |
| Node.js | 18+ | MCP 工具服务器（可选） |

### 3.3 手动部署 TetraMem API

```bash
# 克隆项目
git clone https://github.com/sunormesky-max/Tetrahedron-Memory-Hive-System.git
cd Tetrahedron-Memory-Hive-System

# 安装依赖
pip install fastapi uvicorn numpy

# 启动 API（默认端口 8000）
export TETRAMEM_STORAGE=./tetramem_data_v2
export TETRAMEM_UI_PASSWORD=your_password_here
python -m uvicorn start_api_v2:app --host 127.0.0.1 --port 8000
```

**环境变量：**

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `TETRAMEM_STORAGE` | `./tetramem_data_v2` | 持久化目录 |
| `TETRAMEM_UI_PASSWORD` | `CHANGE_ME` | UI 登录密码（**必须修改**） |

### 3.4 Docker 部署

```bash
git clone https://github.com/sunormesky-max/Tetrahedron-Memory-Hive-System.git
cd Tetrahedron-Memory-Hive-System
TETRAMEM_PASSWORD=mypass docker compose up -d
```

### 3.5 Systemd 服务（推荐）

```ini
# /etc/systemd/system/tetramem-api.service
[Unit]
Description=TetraMem-XL v8.0 API Server
After=network.target

[Service]
Type=simple
WorkingDirectory=/path/to/Tetrahedron-Memory-Hive-System
Environment=TETRAMEM_STORAGE=/path/to/tetramem_data_v2
Environment=TETRAMEM_UI_PASSWORD=your_password_here
ExecStart=/usr/bin/python3 -m uvicorn start_api_v2:app --host 127.0.0.1 --port 8000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

---

## 4. API 端点完整参考

### 4.1 基础记忆操作 (routers/memory)

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/v1/store` | POST | 存入记忆 |
| `/api/v1/query` | POST | 查询记忆 |
| `/api/v1/associate` | POST | 关联查询 |
| `/api/v1/browse` | GET | 时间线浏览 |
| `/api/v1/stats` | GET | 系统统计 |
| `/api/v1/health` | GET | 健康检查 |
| `/api/v1/export` | GET/POST | 导出 |
| `/api/v1/import` | POST | 导入 |

### 4.2 Agent 驱动端点 (routers/agent)

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/v1/agent/context` | POST | 话题记忆上下文组装 |
| `/api/v1/agent/reasoning` | POST | 多跳推理链 |
| `/api/v1/agent/suggest` | POST | 主动行动建议 |
| `/api/v1/navigate` | POST | 拓扑路径导航 |

### 4.3 反馈闭环 (routers/agent)

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/v1/feedback/record` | POST | 记录决策反馈 |
| `/api/v1/feedback/learn` | POST | 从行动中学习 |
| `/api/v1/feedback/stats` | GET | 反馈统计 |
| `/api/v1/feedback/insights` | GET | 学习洞察 |

### 4.4 会话管理 (routers/agent)

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/v1/session/create` | POST | 创建会话 |
| `/api/v1/session/{id}/add` | POST | 添加对话轮次 |
| `/api/v1/session/{id}/recall` | GET | 回溯会话 |
| `/api/v1/session/{id}/consolidate` | POST | 整合会话记忆 |
| `/api/v1/session/list` | GET | 列出活跃会话 |
| `/api/v1/session/{id}` | GET | 获取会话详情 |

### 4.5 系统端点 (routers/system)

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/v1/health` | GET | 健康检查 |
| `/api/v1/stats` | GET | 系统统计 |
| `/api/v1/login` | POST | 登录认证（接受 api_key 或 password） |

### 4.6 认知引擎 (routers/neural)

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/v1/dream` | POST | 触发梦境周期 |
| `/api/v1/dream/status` | GET | 梦境引擎状态 |
| `/api/v1/dream/history` | GET | 梦境历史 |
| `/api/v1/pulse-status` | GET | 脉冲状态 |
| `/api/v1/pcnn/hebbian` | GET | 赫布路径统计 |
| `/api/v1/cascade/trigger` | POST | 触发级联脉冲 |
| `/api/v1/crystallized/force` | POST | 强制结晶化 |
| `/api/v1/crystallized/status` | GET | 结晶通路状态 |

### 4.7 空间拓扑 (routers/spatial)

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/v1/lattice-integrity/check` | GET | BCC 晶格完整性检查 |
| `/api/v1/lattice-integrity/status` | GET | 完整性检查状态 |
| `/api/v1/honeycomb/analysis` | GET | 蜂巢结构分析 |
| `/api/v1/honeycomb/cells` | GET | 四面体单元列表 |
| `/api/v1/self-organize` | POST | 拓扑自组织 |
| `/api/v1/self-check/run` | POST | 运行自检 |
| `/api/v1/clusters` | GET | 语义簇 |
| `/api/v1/shortcuts` | GET | 拓扑捷径 |

### 4.8 暗位面与调节 (routers/darkplane)

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/v1/dark-plane/stats` | GET | 暗位面统计（位面分布/温度/阈值） |
| `/api/v1/dark-plane/flow` | POST | 触发暗位面流动周期 |
| `/api/v1/dark-plane/node/{node_id}` | GET | 查看指定节点暗位面状态 |
| `/api/v1/regulation/trigger` | POST | 触发自我调节周期 |
| `/api/v1/regulation/status` | GET | PID 控制器状态 + 激素水平 |
| `/api/v1/attention/focus` | POST | 设置注意力焦点 |
| `/api/v1/attention/clear` | POST | 清除注意力焦点 |
| `/api/v1/attention/status` | GET | 查询注意力状态 |

### 4.9 运行观察器 (routers/observer) — v8.0 新增

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/v1/observer/stats` | GET | 观察器状态（事件数/存储数/分类统计/自适应限速/共振检测） |
| `/api/v1/observer/flush` | POST | 手动 flush 所有聚合窗口 |
| `/api/v1/observer/enable` | POST | 启用观察器 |
| `/api/v1/observer/disable` | POST | 禁用观察器 |
| `/api/v1/observer/observe` | POST | 手动注入观察事件 |
| `/api/v1/observer/batch` | POST | 批量推送 ≤100 日志事件 |
| `/api/v1/observer/ingest-json` | POST | 原始日志行 ≤200 行，自动格式检测 |
| `/api/v1/observer/trajectories` | GET | SSE 实时轨迹流 |
| `/api/v1/observer/register-classifier` | POST | 远程正则分类器注册 |

### 4.10 OpenClaw 兼容端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/v1/search` | POST | 语义搜索（映射到 query） |
| `/api/v1/read` | POST | 读取记忆 |
| `/api/v1/status` | GET | 系统状态 |
| `/api/v1/sync` | POST | 触发持久化同步 |
| `/api/v1/capabilities/embeddings` | GET | 向量嵌入能力（返回不可用） |
| `/api/v1/capabilities/vectors` | GET | 向量引擎能力（返回不可用） |

### 4.11 SSE 事件流

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/v1/events` | GET | SSE 实时事件订阅 |

事件类型：`feedback_recorded`, `feedback_learned`, `session_created`, `session_consolidated`, `insight_high_priority`, `heartbeat`

---

## 5. MCP 工具完整列表（22 个）

### 5.1 核心记忆操作

| 工具名 | 必填参数 | 说明 |
|--------|----------|------|
| `tetramem_store` | content | 存入记忆（可选: labels, weight, metadata） |
| `tetramem_query` | query | 查询记忆（可选: k, labels） |
| `tetramem_associate` | tetra_id | 关联查询（可选: max_depth） |
| `tetramem_navigate` | seed_id | 拓扑导航（可选: max_steps, strategy） |
| `tetramem_export` | — | 导出所有记忆为 Markdown |
| `tetramem_stats` | — | 系统统计 |
| `tetramem_topology_health` | — | 拓扑健康报告 |
| `tetramem_seed_by_label` | labels | 按标签查找种子节点 |

### 5.2 认知周期

| 工具名 | 说明 |
|--------|------|
| `tetramem_dream` | 触发梦境（跨域创意重组） |
| `tetramem_self_organize` | 拓扑自组织 |
| `tetramem_abstract_reorganize` | 概念抽象重组 |
| `tetramem_closed_loop` | 完整认知闭环（Dream+SelfOrg+Abstract） |

### 5.3 Agent 智能

| 工具名 | 必填参数 | 说明 |
|--------|----------|------|
| `tetramem_agent_context` | topic | 获取话题记忆上下文（可选: max_memories） |
| `tetramem_agent_reasoning` | source_id, target_query | 多跳推理链（可选: max_hops） |
| `tetramem_agent_suggest` | — | 获取行动建议（可选: context） |
| `tetramem_agent_navigate` | source_id, target_id | A*路径导航（可选: max_hops） |

### 5.4 反馈闭环

| 工具名 | 必填参数 | 说明 |
|--------|----------|------|
| `tetramem_feedback_record` | action, context_id, outcome | 记录反馈（可选: confidence, reasoning） |
| `tetramem_feedback_insights` | — | 获取学习洞察 |

### 5.5 会话管理

| 工具名 | 必填参数 | 说明 |
|--------|----------|------|
| `tetramem_session_create` | — | 创建会话（可选: agent_id, metadata） |
| `tetramem_session_add` | session_id, content | 添加对话（可选: role） |
| `tetramem_session_recall` | session_id | 回溯会话（可选: n） |
| `tetramem_session_consolidate` | session_id | 整合临时记忆为永久 |

---

## 6. OpenClaw 集成指南

### 6.1 集成架构

OpenClaw 通过两个并行机制使用 TetraMem：
1. **原生记忆后端** — 替换内置 SQLite 后端，Agent 的 `memory search` 直接查询 TetraMem
2. **MCP 工具** — 22 个工具，Agent 可主动调用高级功能

### 6.2 配置 OpenClaw

在 `openclaw.json` 中配置：

```json
{
  "memory": {
    "backend": "tetramem",
    "tetramem": {
      "apiUrl": "http://127.0.0.1:8000",
      "defaultCollection": "default",
      "maxResults": 10
    }
  },
  "mcp": {
    "servers": {
      "tetramem": {
        "command": "/bin/bash",
        "args": ["/path/to/mcp-tetramem-safe.sh"],
        "env": {
          "TETRA_API_URL": "http://127.0.0.1:8000"
        }
      }
    }
  },
  "agents": {
    "defaults": {
      "systemPromptOverride": "[你的系统提示，指示 Agent 优先使用 TetraMem]",
      "memorySearch": {
        "enabled": true,
        "sources": ["memory"],
        "extraPaths": ["/path/to/tetramem_export.md"]
      }
    }
  }
}
```

**重要：tetramem 配置只接受以下键：**
- `apiUrl` — API 地址
- `apiKey` — API 密钥（可选）
- `defaultCollection` — 集合名（可选）
- `maxResults` — 最大返回数（可选）

**不接受 `timeout` 等其他键，会导致配置校验失败。**

### 6.3 修补 OpenClaw 源码

OpenClaw 需要修补 4 个文件才能识别 `"tetramem"` 后端：

| 文件 | 修补内容 |
|------|----------|
| `zod-schema-*.js` | 在 `MemorySchema` 的 `backend` 联合类型中添加 `"tetramem"`；添加 `TetraMemConfigSchema` |
| `runtime-schema-*.js` | 在 JSON Schema `anyOf` 中添加 `"tetramem"` |
| `backend-config-*.js` | 在 `resolveMemoryBackendConfig()` 中添加 `tetramem` 分支 |
| `memory-*.js`（主加载器） | 添加 `backend === "tetramem"` 分支，`import` 并调用 `createTetraMemManager()` |

**部署 `tetramem-manager.js`** 到 `dist/` 目录，实现以下接口：

```javascript
export async function createTetraMemManager(config, params) {
  return {
    async search(query, opts) { ... },
    async readFile(params) { ... },
    status() { ... },
    async sync(opts) { ... },
    async probeEmbeddingAvailability() {},
    async probeVectorAvailability() {},
    async close() {},
  };
}
```

**API 端点映射：**

| Manager 方法 | TetraMem API |
|-------------|-------------|
| `search` | `POST /api/v1/search` body: `{query, limit}` |
| `readFile` | `POST /api/v1/read` body: `{path}` |
| `status` | `GET /api/v1/status` |
| `sync` | `POST /api/v1/sync` body: `{paths, fullSync}` |
| `probeEmbeddingAvailability` | `GET /api/v1/capabilities/embeddings` |
| `probeVectorAvailability` | `GET /api/v1/capabilities/vectors` |

### 6.4 系统提示模板

在 `systemPromptOverride` 中注入以下指令，让 Agent 优先使用 TetraMem：

```
[记忆系统规则]
1. 回答任何问题前，先用 tetramem_query 搜索相关记忆
2. 新知识必须用 tetramem_store 存入
3. 重要对话结束前用 tetramem_export 备份
4. 获取上下文用 tetramem_agent_context
5. 做出决策后用 tetramem_feedback_record 记录结果
6. 重要对话用 tetramem_session_create 跟踪
7. 定期用 tetramem_dream 触发创意关联
8. 优先使用记忆系统而非临时回忆
```

---

## 7. 自动学习系统

### 7.1 Cron 定时任务

```bash
# 每30分钟自动学习
*/30 * * * * /path/to/auto_learn_cron.sh >> /var/log/auto_learn.log 2>&1

# 每30分钟记忆同步导出
*/30 * * * * python3 /path/to/tetramem_sync.py >> /var/log/tetramem-sync.log 2>&1
```

### 7.2 自动学习脚本模板

```bash
#!/bin/bash
TOPIC_FILE="/path/to/next_topic.txt"
TOPIC=$(cat "$TOPIC_FILE" 2>/dev/null || echo "随机选择一个学习主题")

openclaw agent --message "执行学习任务：
1. 用 tetramem_agent_context 获取「$TOPIC」的已有记忆
2. 深度学习并搜索资料
3. 用 tetramem_store 按四部分结构存入（怎么读到的/学会了什么/卡在哪里/感悟怎么来的）
4. 用 tetramem_feedback_record 记录学习结果 (positive)
5. 用 tetramem_dream 触发一次梦境
6. 更新 next_topic.txt 为下一个主题"
```

### 7.3 主题池示例

```
# next_topic.txt — 每行一个主题
量子力学 双缝实验 观察者效应 哲学意义
康德 纯粹理性批判 物自体 认知边界
斯多葛哲学 爱比克泰德 控制能控制的
```

---

## 8. 调试经验

### 8.1 PowerShell 通过 SSH 的引号陷阱

Windows PowerShell 通过 SSH 执行含引号的命令会被破坏。解决方案：

```python
# 写 Python 脚本上传到服务器执行
import subprocess
subprocess.run(["scp", "script.py", "root@SERVER:/tmp/"])
subprocess.run(["ssh", "root@SERVER", "python3 /tmp/script.py"])
```

### 8.2 OpenClaw 配置校验严格

OpenClaw 使用 `.strict()` 的 Zod schema，**任何未定义的键都会导致整个配置被拒绝**。调试方法：

```bash
journalctl --user -u openclaw-gateway --since '1 min ago' | grep "Unrecognized key"
```

### 8.3 MCP 服务器防重复启动

```bash
#!/bin/bash
# mcp-tetramem-safe.sh — 防止 MCP 重复拉起
LOCKFILE="/tmp/tetramem-mcp.lock"
if [ -f "$LOCKFILE" ]; then
    PID=$(cat "$LOCKFILE")
    if kill -0 "$PID" 2>/dev/null; then exit 0; fi
fi
echo $$ > "$LOCKFILE"
exec node /path/to/mcp-tetramem/index.js
```

### 8.4 SelfCheckEngine 死锁排查

v5.3 曾报告自检一直卡住。排查发现：
- `SelfCheckEngine.run_full_check()` 持有 `field._lock`（RLock）
- 内部调用 `_emit_pulse()` 也需要 `_lock`
- RLock 理论上可重入，但如果其他线程同时竞争可能导致长时间阻塞
- **解决方案**：将脉冲发射移到锁外执行，或使用独立的锁

### 8.5 暗位面调参

暗位面引擎使用自适应分位数阈值，无需手动调整。如需覆盖：

```python
# dark_plane_engine.py 中可调参数
BOLTZMANN_BETA = 1.0 / temperature  # 温度越高，跃迁越活跃
TUNNEL_KAPPA = 0.5                   # 隧道穿透系数
METASTABLE_COOLDOWN = 30.0           # 亚稳态冷却时间(秒)
ENERGY_INJECTION_BASE = 0.1          # 查询能量注入基础值
```

### 8.6 PID 自我调节参数

```python
# self_regulation.py PID 参数
bridge_rate:    kp=0.15, ki=0.02, kd=0.03, integral_limit=1.0
crystal_ratio:  kp=0.10, ki=0.01, kd=0.02, integral_limit=0.8
field_entropy:  kp=0.08, ki=0.01, kd=0.02, integral_limit=0.5
activation:     kp=0.05, ki=0.005, kd=0.01, integral_limit=1.0
emergence:      kp=0.12, ki=0.015, kd=0.025, integral_limit=1.5
```

---

## 9. API 请求/响应示例

### 存入记忆
```bash
curl -X POST http://127.0.0.1:8000/api/v1/store \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-key" \
  -d '{"content": "学习内容...", "labels": ["哲学", "康德"], "weight": 5.0}'
# → {"id": "a1b2c3d4e5f6"}
```

### 查询记忆
```bash
curl -X POST http://127.0.0.1:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-key" \
  -d '{"query": "康德哲学", "k": 5}'
# → {"results": [{"id": "...", "content": "...", "distance": 0.3, "weight": 5.0, ...}]}
```

### 暗位面统计
```bash
curl http://127.0.0.1:8000/api/v1/dark-plane/stats -H "X-API-Key: your-key"
# → {"plane_distribution": {"surface": 0.42, "shallow": 0.14, "deep": 0.19, "abyss": 0.24},
#     "temperature": 1.23, "thresholds": [-0.75, -0.04, 0.75], ...}
```

### 自我调节状态
```bash
curl http://127.0.0.1:8000/api/v1/regulation/status -H "X-API-Key: your-key"
# → {"pid_controllers": {...}, "hormones": {"dopamine": 0.49, "cortisol": 0.15, ...},
#     "autonomic_mode": "normal", "stress_level": 0.12}
```

### Agent 上下文
```bash
curl -X POST http://127.0.0.1:8000/api/v1/agent/context \
  -H "X-API-Key: your-key" \
  -d '{"topic": "量子力学", "max_memories": 15}'
# → {"topic": "量子力学", "context_count": 7, "context": [...], "reasoning": "..."}
```

### 反馈记录
```bash
curl -X POST http://127.0.0.1:8000/api/v1/feedback/record \
  -H "X-API-Key: your-key" \
  -d '{"action": "query", "context_id": "a1b2c3d4", "outcome": "positive", "confidence": 0.8}'
# → {"recorded": true, "action_taken": "weight_boosted:+0.160"}
```

---

## 10. 核心设计原则

| 原则 | 实现方式 |
|------|----------|
| **记忆永不删除** | FeedbackLoop 中 negative 不降权重，只标记 `__low_priority__` |
| **整合优先于覆盖** | 相同内容的 store 只增加权重 +0.1 |
| **主动感知** | SSE 推送让 Agent 不需要轮询 |
| **学习型闭环** | 每个决策都成为学习信号 |
| **渐进式记忆** | ephemeral → short-term → long-term → eternal |
| **暗位面热力学** | 自适应分位数阈值 + 玻尔兹曼重分布 + WKB 隧道穿透 |
| **PID 自我调节** | 5 控制器 + 迟滞带 + 暗位面反馈 + 内分泌激素 |

---

## 11. 文件结构

```
Tetrahedron-Memory-Hive-System/
├── tetrahedron_memory/
│   ├── honeycomb_neural_field.py    # 核心编排器 (~4320 行)
│   ├── runtime_observer.py          # 运行观察器 (v8.0)
│   ├── dark_plane_engine.py         # 暗位面热力学引擎
│   ├── self_regulation.py           # PID 自我调节引擎
│   ├── self_check.py                # 自检脉冲引擎
│   ├── insight_aggregator.py        # 洞察聚合器
│   ├── pcnn_types.py                # PCNN 类型与配置常量
│   ├── app_state.py                 # AppState 全局状态容器
│   ├── routers/                     # API 路由 (6 模块)
│   │   ├── memory.py                # store, query, browse, export
│   │   ├── agent.py                 # context, reasoning, feedback, session
│   │   ├── system.py                # health, stats, login
│   │   ├── neural.py                # dream, pulse, cascade, crystal
│   │   ├── spatial.py               # lattice, honeycomb, integrity
│   │   └── darkplane.py             # 暗位面, 调节, 注意力
│   │   └── observer.py             # 运行观察器, 自省轨迹 (v8.0)
│   └── ...
├── start_api_v2.py                   # FastAPI 入口 (~95 行)
├── tests/
│   └── test_integration.py           # 20 个集成测试
│   └── test_runtime_observer.py      # 25 个观察器单元测试 (v8.0)
├── ui/
│   ├── index.html                    # 可视化 UI (后端登录)
│   └── dashboard.html                # 独立仪表板
├── tetramem_data_v2/
│   └── mesh_index.json               # 持久化数据
├── tetramem_mcp_server.py             # MCP 工具服务器 (Python)
├── install.sh                        # 一键安装脚本
├── Dockerfile                        # Docker 构建
├── docker-compose.yml                # Docker Compose
├── pyproject.toml                    # Python 项目配置
└── requirements.txt                  # Python 依赖
```

---

## 12. 性能基准（v8.0）

| 指标 | 服务器 | 本地 |
|------|--------|------|
| Store 吞吐 | 103 mem/s | 378 mem/s |
| Query 延迟 | 92ms avg | — |
| API 路由数 | 129 | — |
| 集成测试 | 20/20 pass (1.7s) | — |

### 暗位面实测

- 自适应阈值: `-0.75 / -0.04 / 0.75`
- 位面分布: ~42% surface, ~14% shallow, ~19% deep, ~24% abyss
- 温度范围: 1.0 → 1.6 (随活动率上升)
- 玻尔兹曼/隧道/注入事件: 正常触发

### 自我调节实测

- 5 PID 控制器: 同时运行，积分项收敛
- 激素水平: dopamine~0.49, cortisol~0.15, serotonin~0.50, acetylcholine~0.57
- 自主神经模式: 正常 (迟滞带防抖动)
