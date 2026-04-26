# Tetrahedron-Memory-Hive-System

**四面体记忆蜂巢系统** | Tetrahedral Memory Hive System

BCC 晶格蜂巢 + PCNN 脉冲耦合神经网络 + 暗位面热力学引擎 — 纯 Python 实现的永恒记忆系统

[中文文档](./README.cn.md) | [集成指南](./INTEGRATION_GUIDE.md) | [变更日志](./CHANGELOG.md)

## 核心原则

| 原则 | 含义 |
|------|------|
| **永恒** | 所有记忆永久保留，不删除、不遗忘，只整合 |
| **整合** | 记忆持续抽象重组，形成更高阶概念 |
| **自涌现** | 无外部输入时也能自发产生新记忆和新关联 |
| **闭环** | Store → Query → Dream → Self-Organize → Regulate → 循环 |

## 快速开始

### 一键安装（Linux 服务器）

```bash
curl -sSL https://raw.githubusercontent.com/sunormesky-max/Tetrahedron-Memory-Hive-System/main/install.sh | bash

# 自定义密码和端口
curl -sSL https://raw.githubusercontent.com/sunormesky-max/Tetrahedron-Memory-Hive-System/main/install.sh | bash -s -- --password MySecret123 --port 8000
```

### Docker

```bash
git clone https://github.com/sunormesky-max/Tetrahedron-Memory-Hive-System.git
cd Tetrahedron-Memory-Hive-System
TETRAMEM_PASSWORD=mypass docker compose up -d
```

### Python 包

```bash
pip install -e .
```

```python
import requests

BASE = "http://localhost:8000/api/v1"
H = {"X-API-Key": "your-key", "Content-Type": "application/json"}

# 存储记忆
r = requests.post(f"{BASE}/store", json={
    "content": "AI memory should be eternal",
    "labels": ["principle", "ai"],
    "weight": 1.5,
}, headers=H)
print(r.json())  # {"id": "abc123..."}

# 查询记忆
r = requests.post(f"{BASE}/query", json={"query": "eternal memory", "k": 5}, headers=H)
for m in r.json()["results"]:
    print(f"  {m['content'][:50]}... score={m['distance']:.3f}")

# 触发梦境循环（创造性重组）
r = requests.post(f"{BASE}/dream", headers=H)

# 触发自组织
r = requests.post(f"{BASE}/self-organize", headers=H)

# 暗位面统计
r = requests.get(f"{BASE}/dark-plane/stats", headers=H)
print(r.json())
```

## 系统架构

```
HoneycombNeuralField (核心编排器)
  ├── BCC Lattice (体心立方晶格拓扑)
  │     └── HoneycombNode (记忆节点: content, labels, weight, activation)
  ├── PCNN Pulse Engine (脉冲耦合神经网络)
  │     ├── 脉冲类型: 探索/强化/级联/自检/结构/深层强化
  │     └── 自适应脉冲间隔 (基于占用率和桥接率)
  ├── Hebbian Path Memory (赫布路径记忆 — "一起激发，一起连接")
  ├── Crystallized Pathway (晶体化路径 — 零衰减永久快通道)
  ├── Dream Engine (梦境引擎 — 跨域创造性记忆重组)
  ├── Self-Organize Engine (自组织引擎 — 自动聚类和迁移)
  ├── Dark Plane Engine (暗位面引擎 — 热力学能量景观)
  │     ├── 自适应分位数阈值 (根据深度分布自动调整)
  │     ├── 玻尔兹曼重分布 (热噪声随机跃迁)
  │     ├── WKB 隧道穿透 (能量势垒逃逸)
  │     └── 查询能量注入 (命中深层节点时注入能量)
  ├── Self-Regulation Engine (自我调节引擎 — 六层生理控制系统)
  │     ├── Layer 1: PID 稳态控制器 (桥接率/晶体比/熵/激活值/涌现质量)
  │     ├── Layer 2: 昼夜节律 (工作/整理期交替)
  │     ├── Layer 3: 自主神经 (交感/副交感/平衡模式)
  │     ├── Layer 4: 免疫系统 (死边清理/一致性修复)
  │     ├── Layer 5: 内分泌系统 (多巴胺/皮质醇/血清素/乙酰胆碱)
  │     └── Layer 6: 压力响应 (过载保护+紧急节流)
  ├── Spatial Reflection Field (空间反射场 — 能量景观+相位检测)
  ├── Attention Mechanism (注意力机制 — 焦点+扩散+衰减)
  └── Persistence Engine (持久化引擎 — WAL 日志+检查点)
```

## API 参考

基础地址: `http://<host>:8000/api/v1/`

### 核心操作

| 端点 | 方法 | 说明 |
|------|------|------|
| `/store` | POST | 存储记忆 `{"content": "...", "labels": [...], "weight": 1.0}` |
| `/query` | POST | 查询记忆 `{"query": "...", "k": 5}` |
| `/browse` | GET | 浏览记忆 `?direction=newest&limit=20` |
| `/tetrahedra/{id}` | GET/DELETE | 获取/删除单个记忆 |
| `/stats` | GET | 系统统计 |
| `/health` | GET | 健康检查 |

### 自主系统

| 端点 | 方法 | 说明 |
|------|------|------|
| `/dark-plane/flow` | POST | 运行暗位面能量流动 |
| `/dark-plane/stats` | GET | 暗位面统计 |
| `/dark-plane/node/{id}` | GET | 节点能量报告 |
| `/regulation/trigger` | POST | 触发自我调节 |
| `/regulation/status` | GET | 调节状态 |
| `/dream` | POST | 触发梦境循环 |
| `/self-organize` | POST | 触发自组织 |
| `/cascade/trigger` | POST | 触发级联脉冲 |

### 注意力与反射

| 端点 | 方法 | 说明 |
|------|------|------|
| `/attention/focus` | POST | 设置注意力焦点 |
| `/attention/clear` | POST | 清除注意力 |
| `/reflection-field/run` | POST | 运行空间反射 |
| `/reflection-field/status` | GET | 反射场状态 |

## 关键特性

- **纯 Python 实现** — 零外部引擎依赖，标准库 + NumPy
- **BCC 晶格拓扑** — 体心立方晶体结构作为记忆空间
- **PCNN 脉冲传播** — 8 种脉冲类型自主传播
- **20 维查询评分** — 文本/三字符/标签/激活/权重/赫布/晶体/脉冲/空间质量/几何/单元密度等
- **暗位面热力学** — 自由能景观 + 自适应阈值 + 玻尔兹曼重分布 + 量子隧道穿透
- **六层自我调节** — PID 控制 + 昼夜节律 + 自主神经 + 免疫 + 内分泌 + 压力响应
- **注意力机制** — 用户可控焦点 + 自动扩散 + 时间衰减
- **永恒记忆** — 不删除、不遗忘，只整合和重组
- **实时 3D 可视化** — Three.js 蜂巢晶格浏览器

## 性能基准

测试环境: 1核 2GB 云服务器

| 指标 | 数值 |
|------|------|
| 写入吞吐 (500 节点) | **103 mem/s** |
| 写入吞吐 (本地) | **378 mem/s** |
| 查询延迟 (平均) | **92ms** |
| 查询相关度 | **0.595 avg** |
| 涌现质量 | **0.689 (emerging)** |
| API 路由数 | **129** |

## 项目结构

```
tetrahedron_memory/
  ├── honeycomb_neural_field.py   # 核心编排器 (~4300 行)
  ├── honeycomb_node.py           # 记忆节点 (四面体单元)
  ├── dark_plane_engine.py        # 暗位面热力学引擎
  ├── self_regulation.py          # 六层自我调节引擎
  ├── dream_engine.py             # 梦境引擎
  ├── self_organize.py            # 自组织引擎
  ├── hebbian_memory.py           # 赫布路径记忆
  ├── crystallized_pathway.py     # 晶体化路径
  ├── spatial_reflection.py       # 空间反射场
  ├── pcnn_types.py               # PCNN 配置和类型
  ├── geometry.py                 # 文本到几何映射
  ├── persistence_engine.py       # 持久化 (WAL + 检查点)
  ├── app_state.py                # 应用状态容器
  ├── auth.py                     # 认证管理
  ├── routers/                    # FastAPI 路由模块
  │     ├── memory.py             # 存储/查询/浏览
  │     ├── agent.py              # 代理集成
  │     ├── system.py             # 系统/健康/登录
  │     ├── neural.py             # 脉冲/梦境/自组织
  │     ├── spatial.py            # 空间/反射
  │     └── darkplane.py          # 暗位面/调节/注意力
  └── ...
start_api_v2.py                   # FastAPI 入口
tests/test_integration.py         # 20 个集成测试
ui/                               # 前端 (Vue 3 + Three.js)
```

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `TETRAMEM_STORAGE` | `./tetramem_data_v2` | 数据目录 |
| `TETRAMEM_UI_PASSWORD` | `CHANGE_ME` | UI 登录密码 |
| `TETRAMEM_PORT` | `8000` | API 端口 |
| `TETRAMEM_CORS_ORIGINS` | `http://localhost:3000` | CORS 允许源 |

## UI

- 嵌入式: `http://<host>:8000/ui/`
- 独立仪表盘: 浏览器打开 `ui/dashboard.html`，输入 API 地址

## 管理命令

```bash
systemctl restart tetramem-api   # 重启
systemctl stop tetramem-api      # 停止
journalctl -u tetramem-api -f    # 查看日志
```

## 许可证

AGPL-3.0-or-later (GNU Affero General Public License v3)

**作者**: sunorme (刘启航)
