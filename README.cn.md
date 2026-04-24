# TetraMem-XL

**三角体空间几何记忆体** — 纯几何驱动、拓扑自组织的下一代 AI 记忆系统

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19429105.svg)](https://doi.org/10.5281/zenodo.19429105)

## 项目愿景

TetraMem-XL 以 3-simplex（四面体）为基本记忆单元，彻底抛弃传统向量嵌入和余弦相似度，采用纯几何原语 + 拓扑数据分析（TDA）实现记忆的存储、检索和联想。

记忆不再是扁平向量，而是生长在三维几何空间中的动态拓扑结构，具有多尺度联想、动态自组织和分布式扩展能力。

## 五大永恒原则

| 原则 | 实现 |
|------|------|
| **永恒** | 无衰减/遗忘，所有记忆永久保留，去留由AI闭环决定 |
| **整合** | 梦境自我联想 + 定期自发整合，噪声通过转化而非删除处理 |
| **自涌现** | 无外部输入时也能自发产生新记忆、新关联、新洞见 |
| **完整闭环** | 记忆→思考→执行→反思→整合→梦境 循环往复 |
| **空间结构** | 四面体蜂巢立体记忆系统 |

## 安装

```bash
pip install tetrahedron-memory
```

可选依赖组：

```bash
pip install tetrahedron-memory[visualization]   # plotly, matplotlib
pip install tetrahedron-memory[distributed]      # ray
pip install tetrahedron-memory[api]              # fastapi, uvicorn, prometheus-client
pip install tetrahedron-memory[all]              # 全部
```

## 快速开始

```python
from tetrahedron_memory import GeoMemoryBody

memory = GeoMemoryBody(dimension=3, precision="fast")

# 存储
memory.store(content="AI记忆架构", labels=["ai", "memory"], weight=1.0)

# 查询
results = memory.query("AI记忆", k=5)

# 多参数过滤查询
results = memory.query_multiparam("AI记忆", k=10, labels_required=["ai"])

# 分辨率金字塔查询（快速粗→细）
memory.build_pyramid()
results = memory.query_pyramid("AI记忆", k=5)

# Zigzag持久性跟踪
snapshot = memory.record_zigzag_snapshot()
prediction = memory.predict_topology()

# 映射锥体：追踪梦境周期变换
cones = memory.get_mapping_cone_history()
guidance = memory.get_dream_guidance()

# 永恒审计：证明没有任何记忆被删除
report = memory.verify_eternity()
chain = memory.get_eternity_trail("some_id")

# 全局整合催化剂（无衰减）
result = memory.global_catalyze_integration(strength=1.0)
```

## 理论创新（v2.2）

### 迭代映射锥体建模
每个梦境周期在 Zigzag 持久性框架中构建**映射锥体** C(f): X_pre → X_post：
- **正向映射**：哪些特征在梦境中存活、诞生或消亡
- **反向映射**：溯源追踪——梦境前哪些特征产生了梦境后的每个特征
- **稳定性认证**：每次锥体中特征被分类为稳定/诞生/消亡/合并
- **迭代累积**：锥体跨梦境周期链接，形成累积稳定性分析
- **梦境引导**：历史锥体指导未来梦境聚焦于不稳定或未充分探索的区域

### 动态自适应分辨率金字塔
分辨率金字塔现在具有**闭环反馈**机制：
- 梦境熵变化 → 反馈 → 金字塔自动调整最大层级数和粗化比
- 各层级查询命中率 → 反馈 → 金字塔自适应调整粒度
- 正反馈增加分辨率；负反馈增加粗化度
- 金字塔随系统演化，而非使用静态参数

### 永恒原则严格审计
形式化验证系统，**证明没有任何记忆被删除**：
- 每次操作对每个记忆计算 SHA-256 内容哈希
- 传递性保留链追踪（s1 → m1 → t1）
- `verify()` 扫描全部历史并证明无违规
- 完整审计轨迹：每次 store/merge/transform/dream/reintegration 均带内容证明记录

## REST API（21端点）

```bash
pip install tetrahedron-memory[api]
```

| 方法 | 端点 | 功能 |
|------|------|------|
| POST | `/api/v1/store` | 存储记忆 |
| POST | `/api/v1/query` | 文本查询 |
| POST | `/api/v1/query-multiparam` | 多参数过滤查询 |
| POST | `/api/v1/query-pyramid` | 金字塔查询 |
| POST | `/api/v1/build-pyramid` | 构建金字塔 |
| GET | `/api/v1/associate/{id}` | 联想检索 |
| POST | `/api/v1/self-organize` | 自组织 |
| POST | `/api/v1/dream` | 梦境周期 |
| POST | `/api/v1/closed-loop` | 闭环认知周期 |
| POST | `/api/v1/batch-store` | 批量存储 |
| POST | `/api/v1/weight-update` | 权重更新 |
| POST | `/api/v1/persist` | 持久化刷盘 |
| GET | `/api/v1/stats` | 统计信息 |
| GET | `/api/v1/health` | 健康检查 |
| GET | `/api/v1/health/topology` | 拓扑健康检查 |
| GET | `/api/v1/consistency` | 一致性状态 |
| POST | `/api/v1/zigzag-snapshot` | Zigzag快照 |
| GET | `/api/v1/zigzag-status` | Zigzag状态 |
| GET | `/api/v1/predict-topology` | 拓扑预测 |
| GET | `/api/v1/dynamic-barcode` | 动态Barcode |
| GET | `/metrics` | Prometheus指标 |

## CLI（14命令）

```bash
tetramem store / query / label / stats / clear / persist
tetramem dream / self-org / catalyze / status
tetramem mquery / build-pyramid / pyquery / zigzag / predict
```

## LLM工具（15工具）

OpenAI function calling 兼容：`tetramem_store`、`tetramem_query`、`tetramem_associate`、`tetramem_self_organize`、`tetramem_stats`、`tetramem_dream`、`tetramem_closed_loop`、`tetramem_weight_update`、`tetramem_batch_store`、`tetramem_persist`、`tetramem_query_multiparam`、`tetramem_build_pyramid`、`tetramem_query_pyramid`、`tetramem_zigzag_snapshot`、`tetramem_predict_topology`

## 模块结构（24模块）

| 模块 | 功能 |
|------|------|
| `core.py` | GeoMemoryBody 主引擎（~1450行） |
| `tetra_mesh.py` | TetraMesh 动态四面体网格（slots + float32） |
| `tetra_dream.py` | TetraDreamCycle PH加权梦境融合 |
| `tetra_self_org.py` | TetraSelfOrganizer H0/H1/H2几何手术 |
| `persistent_entropy.py` | 持久熵计算 + EntropyTracker |
| `closed_loop.py` | ClosedLoopEngine 完整认知闭环 |
| `emergence.py` | EmergencePressure + AdaptiveThreshold |
| `zigzag_persistence.py` | ZigzagTracker 动态拓扑特征跟踪 |
| `resolution_pyramid.py` | ResolutionPyramid 多尺度层次聚类 |
| `multiparameter_filter.py` | MultiParameterQuery 6维组合过滤 |
| `consistency.py` | VectorClock + ConsistencyManager + 冲突自动解决 |
| `persistence.py` | ParquetPersistence 两阶段提交 + S3 |
| `monitoring.py` | Prometheus 12指标 + Grafana 15面板 + 告警规则 |
| `structured_log.py` | 分布式追踪 + JSON结构化日志 |
| `router.py` | FastAPI REST API（21端点） |
| `llm_tool.py` | LLM工具（15工具） |
| `cli.py` | 有状态CLI + 持久化（14命令） |
| `geometry.py` | 文本→几何映射 |
| `partitioning.py` | Octree + M3NO + BucketActor + GhostCell |
| `multimodal.py` | PixHomology 图像/音频/视频 |
| `multimodal_bridge.py` | MultimodalBridge |
| `global_coarse_mesh.py` | GlobalCoarseMesh 反馈回路 |
| `tetra_router.py` | TetraMeshRouter 分布式路由 |
| `hooks.py` | 会话启动钩子 |

## 性能指标

| 指标 | 目标 | 实测 |
|------|------|------|
| 插入吞吐（Mesh） | >12,000条/秒 | ~14-20K ops/sec |
| 查询延迟（本地） | <8ms p99 | <5ms |
| 持久熵下降 | ≥18% | 梦境周期后显著下降 |
| 长期运行 | 无内存泄漏 | 390+测试通过 |

## 生产部署

详见 [deployment_guide.md](deployment_guide.md)

```bash
bash deploy.sh
```

## 测试

```bash
pytest                                    # 413+ 测试
pytest -m "not slow"                      # 跳过慢速测试
pytest tests/test_production_final.py     # 生产级验证
```

## Citation

```bibtex
@misc{liu2026pyramidmemory,
  author       = {Liu, Qihang},
  title        = {Pyramid Memory: A Tetrahedron-Based Memory System with O(1) Retrieval Complexity for AI Agents},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.19429105},
  url          = {https://doi.org/10.5281/zenodo.19429105}
}
```

## License

CC BY-NC 4.0 - Personal Learning & Non-Commercial Use Only

## Author

Liu Qihang (sunorme) - sunormesky@gmail.com
