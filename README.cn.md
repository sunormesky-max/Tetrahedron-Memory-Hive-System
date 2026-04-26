# TetraMem-XL 四面体记忆蜂巢系统

**BCC 晶格蜂巢 + PCNN 脉冲神经网络 + 暗位面热力学引擎**

[英文文档](./README.md) | [集成指南](./INTEGRATION_GUIDE.md) | [变更日志](./CHANGELOG.md)

## 项目愿景

TetraMem-XL 以体心立方（BCC）晶格为记忆空间拓扑，结合脉冲耦合神经网络（PCNN）实现记忆的自主激活、传播和整合。系统包含暗位面热力学引擎和六层自我调节系统，无需外部干预即可持续优化自身。

记忆不是扁平向量，而是生长在三维晶体结构中的动态拓扑节点，具有自适应脉冲传播、赫布路径强化、晶体化永久通道、梦境创造性重组、暗位面能量景观等能力。

## 五大永恒原则

| 原则 | 实现 |
|------|------|
| **永恒** | 无衰减/遗忘，所有记忆永久保留 |
| **整合** | 梦境联想 + 自组织迁移 + 暗位面整合 |
| **自涌现** | 脉冲引擎自主运行，无外部输入时也能产生新关联 |
| **闭环** | Store → Query → Dream → Self-Organize → Regulate 循环 |
| **空间结构** | BCC 晶格四面体蜂巢立体记忆 |

## 快速安装

### 一键安装（Linux）

```bash
curl -sSL https://raw.githubusercontent.com/sunormesky-max/Tetrahedron-Memory-Hive-System/main/install.sh | bash
```

### Docker

```bash
git clone https://github.com/sunormesky-max/Tetrahedron-Memory-Hive-System.git
cd Tetrahedron-Memory-Hive-System
TETRAMEM_PASSWORD=mypass docker compose up -d
```

### Python

```bash
pip install -e .
```

```python
import requests

BASE = "http://localhost:8000/api/v1"
H = {"X-API-Key": "your-key", "Content-Type": "application/json"}

# 存储
r = requests.post(f"{BASE}/store", json={
    "content": "记忆内容", "labels": ["标签"], "weight": 1.5
}, headers=H)

# 查询
r = requests.post(f"{BASE}/query", json={"query": "关键词", "k": 5}, headers=H)
for m in r.json()["results"]:
    print(m["content"], m["distance"])

# 梦境循环
requests.post(f"{BASE}/dream", headers=H)

# 自组织
requests.post(f"{BASE}/self-organize", headers=H)

# 暗位面统计
r = requests.get(f"{BASE}/dark-plane/stats", headers=H)
print(r.json())
```

## 系统架构

```
HoneycombNeuralField (核心编排器, ~4300 行)
  ├── BCC 晶格拓扑 (体心立方, 2331 节点/分辨率5)
  ├── PCNN 脉冲引擎 (8种脉冲类型, 自适应间隔)
  ├── 赫布路径记忆 ("一起激发，一起连接")
  ├── 晶体化路径 (零衰减永久快通道)
  ├── 梦境引擎 (跨域创造性重组)
  ├── 自组织引擎 (语义聚类+迁移)
  ├── 暗位面引擎 (热力学能量景观)
  │     ├── 自适应分位数阈值
  │     ├── 玻尔兹曼重分布
  │     ├── WKB 隧道穿透
  │     └── 查询能量注入
  ├── 自我调节引擎 (六层生理控制)
  │     ├── PID 稳态控制
  │     ├── 昼夜节律
  │     ├── 自主神经 (交感/副交感)
  │     ├── 免疫系统 (死边清理)
  │     ├── 内分泌 (多巴胺/皮质醇/血清素/乙酰胆碱)
  │     └── 压力响应 (过载保护)
  ├── 空间反射场 (能量景观+相位检测)
  ├── 注意力机制 (焦点+扩散+衰减)
  └── 持久化引擎 (WAL 日志+检查点)
```

## 性能

| 指标 | 数值 |
|------|------|
| 写入吞吐 (服务器) | 103 mem/s |
| 写入吞吐 (本地) | 378 mem/s |
| 查询延迟 | 92ms (平均) |
| API 路由 | **129 条** |
| 集成测试 | 20/20 通过 |

## 项目结构

```
tetrahedron_memory/
  ├── honeycomb_neural_field.py   # 核心编排器
  ├── honeycomb_node.py           # 记忆节点
  ├── dark_plane_engine.py        # 暗位面引擎
  ├── self_regulation.py          # 自我调节引擎
  ├── dream_engine.py             # 梦境引擎
  ├── self_organize.py            # 自组织引擎
  ├── hebbian_memory.py           # 赫布路径
  ├── crystallized_pathway.py     # 晶体化路径
  ├── spatial_reflection.py       # 空间反射场
  ├── pcnn_types.py               # PCNN 配置
  ├── persistence_engine.py       # 持久化
  ├── routers/                    # API 路由
  │     ├── memory.py             # 存储/查询/浏览
  │     ├── agent.py              # 代理集成
  │     ├── system.py             # 系统/登录
  │     ├── neural.py             # 脉冲/梦境
  │     ├── spatial.py            # 空间/反射
  │     └── darkplane.py          # 暗位面/调节
  └── ...
start_api_v2.py                   # FastAPI 入口
tests/test_integration.py         # 集成测试
ui/                               # 前端 (Vue 3 + Three.js)
```

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `TETRAMEM_STORAGE` | `./tetramem_data_v2` | 数据目录 |
| `TETRAMEM_UI_PASSWORD` | `CHANGE_ME` | UI 密码 |
| `TETRAMEM_PORT` | `8000` | API 端口 |

## UI

- 嵌入式: `http://<host>:8000/ui/`
- 独立: 浏览器打开 `ui/dashboard.html`

## 许可证

AGPL-3.0-or-later (GNU Affero General Public License v3)

**作者**: sunorme (刘启航)
