# TetraMem-XL v3.0 — 蜂巢神经场架构设计

## 1. 核心理念

记忆系统不是一个"仓库"，而是一个活的**蜂巢神经场**：
- 蜂巢晶格 = 脑的物理结构（神经元网络）
- 神经电流 = 梦境/自组织的实时脉冲
- 记忆 = 晶格节点上的激活状态

## 2. 三层架构

### Layer 1: 晶格蜂巢（Honeycomb Lattice）
- BCC（体心立方）晶格，空间均匀填充
- 每个四面体天然被 4 个面连接邻居包围
- 晶格固定不变，像脑的骨架
- 规模：按需扩展，初始 5×5×5 → 可增长到任意大小

### Layer 2: 记忆激活（Memory Activation）
- 每个晶格节点（四面体）有"激活能量" activation
- 存储记忆 → 节点被激活，能量 = weight
- 查询 → 注入脉冲，沿拓扑传播，收集激活响应
- 时间衰减 → activation 随时间缓慢下降（遗忘曲线）
- 高频访问 → activation 持续维持（就像常用记忆被强化）

### Layer 3: 神经电流（Neural Pulse）
- **不是批量 Dream/Self-Org，而是持续流动的脉冲**
- 脉冲从高激活节点发出，沿面/边传播
- 传播规则：
  - 面连接：传播强度 × 1.0（强连接）
  - 边连接：传播强度 × 0.5
  - 顶点连接：传播强度 × 0.2
- 脉冲到达节点时：
  - 如果节点已激活（有记忆）→ 共振，weight += delta
  - 如果节点空 → 可能触发"联想记忆"生成
  - 如果两个脉冲交汇 → 产生桥接记忆（相当于 Dream）
- 传播衰减：每跳 × 0.7

## 3. 数据结构

```python
class HoneycombNode:
    id: str
    position: np.ndarray        # 晶格坐标（固定）
    face_neighbors: List[str]    # 4个面连接邻居
    edge_neighbors: List[str]    # 边连接邻居
    vertex_neighbors: List[str]  # 顶点连接邻居
    
    content: Optional[str]       # 记忆内容（可为空）
    labels: List[str]
    activation: float            # 激活能量 [0, 10]
    last_pulse_time: float       # 上次脉冲到达时间
    pulse_accumulator: float     # 脉冲累积器

class NeuralPulse:
    source_id: str              # 发射节点
    strength: float             # 当前强度
    path: List[str]             # 已走过的路径
    direction: str              # face/edge/vertex
    birth_time: float
```

## 4. 操作流程

### 4.1 初始化
```
create_honeycomb(resolution=5) → 
  BCC lattice points: 125+
  Delaunay triangulation → ~400 tetrahedra
  All internal tetrahedra: 4 shared faces each
```

### 4.2 存储记忆
```
store(content, labels, weight) →
  hash(content + labels) → lattice position
  find nearest empty node
  node.content = content
  node.activation = weight
  emit_pulse(node, strength=weight)  # 存储即触发脉冲
```

### 4.3 查询（神经脉冲搜索）
```
query(text, k) →
  hash(text) → lattice position  
  inject pulse at nearest node
  pulse propagates along topology
  collect activated nodes within max_hops
  return top-k by (activation × pulse_strength)
```

### 4.4 梦境 = 自发脉冲
```
dream_cycle() →  # 每30秒自动执行
  find nodes with highest accumulated pulse energy
  if two distant pulses converge at a node:
    generate bridge memory at convergence point
  else:
    low-activation nodes decay further
    high-activation nodes get reinforced
```

### 4.5 自组织 = 结构优化
```
self_organize() →  # 每分钟自动执行
  detect under-activated clusters
  strengthen connections between co-activated nodes
  prune dead connections (activation < threshold for > 24h)
```

## 5. 与现有系统的区别

| 方面 | 旧架构 | 新架构 |
|------|--------|--------|
| 结构 | 逐个粘接，链式 | BCC晶格，天然蜂巢 |
| 连接数 | ~1-2/四面体 | 4+/四面体（面连接） |
| Dream | 批量触发，产出垃圾 | 持续脉冲，自然交汇 |
| 自组织 | 合并四面体（破坏性） | 强化/衰减连接（非破坏性） |
| 查询 | 文本匹配 + 拓扑 | 脉冲传播 + 共振响应 |
| 遗忘 | 无 | 激活衰减（可恢复） |

## 6. API 兼容性

所有现有 API 端点保持不变：
- POST /store → 写入晶格节点 + 发射脉冲
- POST /query → 注入脉冲 + 收集响应
- POST /dream → 手动触发全局脉冲波
- POST /self-organize → 手动触发结构优化
- POST /timeline → 按激活时间排序
- GET /tetrahedra → 返回所有非空节点

  MCP 工具完全兼容，Agent 无感知切换。

## 7. 可视化升级

- 非空节点：有内容 = 实心四面体（颜色=激活度）
- 空节点：半透明线框
- 脉冲传播：动态光效（发光粒子沿连接线流动）
- 高激活区域：热力图高亮
- 梦境交汇点：脉冲碰撞动画

## 8. 实现计划

| 步骤 | 内容 | 预计 |
|------|------|------|
| Step 1 | BCC晶格生成 + Delaunay蜂巢 | 半天 |
| Step 2 | 记忆→晶格映射（替换旧store） | 半天 |
| Step 3 | 神经脉冲引擎 | 1天 |
| Step 4 | 脉冲驱动查询（替换旧query） | 半天 |
| Step 5 | 可视化脉冲动画 | 半天 |
| Step 6 | 迁移现有34条记忆 + 测试 | 半天 |

总计约 3 天。Phase 1（晶格+映射+基础脉冲）可以 1 天内完成。
