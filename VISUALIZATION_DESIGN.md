# TetraMem-XL 可视化界面设计文档

## 1. 愿景

四面体蜂巢记忆系统的可视化界面，让用户**看见记忆的几何结构**，直接观察和操控三维记忆网格。

核心理念：**记忆不是一行行数据，是三维空间中生长的晶体结构。**

## 2. 技术方案

### 2.1 架构

```
浏览器 (Three.js SPA)
    ↓ HTTP REST + WebSocket
FastAPI (现有 start_api_v2.py, 端口 8000)
    ↓
TetraMesh 核心
    ↓
SQLite 持久化
```

- **零额外后端**：复用现有 FastAPI，补充 WebSocket 端点
- **前端纯静态**：单页应用，打包后放在 `static/` 目录
- **服务器零负担**：所有 3D 渲染在用户浏览器执行

### 2.2 前端技术栈

| 技术 | 用途 | 选择理由 |
|------|------|----------|
| Vue 3 | UI 框架 | 轻量，与服务器已有技术栈一致 |
| Three.js | 3D 渲染 | 四面体网格可视化必需 |
| OrbitControls | 3D 交互 | 旋转/缩放/平移 |
| Element Plus | UI 组件 | 现成的表格/表单/对话框 |
| ECharts | 统计图表 | 与服务器已有技术栈一致 |

### 2.3 后端补充（在 start_api_v2.py 中）

| 端点 | 类型 | 用途 |
|------|------|------|
| `GET /api/v1/tetrahedra` | REST | 返回所有四面体完整数据（位置、内容、标签、权重） |
| `GET /api/v1/tetrahedra/{id}` | REST | 单个四面体详情 |
| `PUT /api/v1/tetrahedra/{id}` | REST | 编辑记忆内容/标签/权重 |
| `DELETE /api/v1/tetrahedra/{id}` | REST | 删除四面体 |
| `GET /api/v1/topology-graph` | REST | 返回拓扑图（节点+边+连接类型） |
| `POST /api/v1/import` | REST | 批量导入记忆 |
| `WS /ws/mesh` | WebSocket | 实时推送 mesh 变化事件 |

## 3. 界面布局

```
┌─────────────────────────────────────────────────────────────┐
│ TetraMem-XL                                    [stats bar] │
├──────────────────────────┬──────────────────────────────────┤
│                          │                                  │
│                          │  记忆详情面板                     │
│                          │  ┌────────────────────────────┐  │
│    3D 四面体网格视图       │  │ ID: abc123               │  │
│                          │  │ 权重: 3.0   [编辑]        │  │
│    (Three.js Canvas)     │  │ 标签: [退税] [外贸] [+]   │  │
│                          │  │ 创建: 2026-04-15 14:32    │  │
│    - 旋转/缩放/点击       │  │ 访问: 5次  集成: 2次      │  │
│    - 颜色 = 权重          │  ├────────────────────────────┤  │
│    - 大小 = filtration    │  │ 内容:                     │  │
│    - 连线 = 拓扑连接      │  │ BOSS发来首次退税资料...    │  │
│    - 标签筛选             │  │                           │  │
│                          │  ├────────────────────────────┤  │
│                          │  │ 拓扑邻居:                  │  │
│                          │  │ [face] xyz789 安全扫描...  │  │
│                          │  │ [edge] def456 退税模板...  │  │
│                          │  └────────────────────────────┘  │
├──────────────────────────┴──────────────────────────────────┤
│ 底部工具栏                                                  │
│ [时间线滑块] [Dream触发] [自组织] [导出] [批量导入] [查询]  │
└─────────────────────────────────────────────────────────────┘
```

## 4. 功能模块

### 4.1 Phase 1：核心 3D 视图 + 记忆管理

**3D 网格视图**
- 每个四面体渲染为半透明四面体体积
- 颜色映射：权重 1.0=蓝色 → 5.0=红色 → 10.0=金色（热力图）
- 大小映射：filtration 值越大体积越大（越老/越稳定）
- 拓扑连接：面连接=粗白线，边连接=细灰线，顶点连接=虚线
- Dream 四面体：特殊发光效果，半透明绿色
- 交互：OrbitControls 旋转缩放 + Raycaster 点击选中
- 筛选：按标签过滤显示，按权重范围过滤

**记忆详情面板**
- 点击四面体显示完整信息
- 编辑：内容文本框、标签编辑、权重滑块
- 拓扑邻居列表（区分 face/edge/vertex 连接）
- 删除按钮（带确认）

**统计栏**
- 总记忆数、总顶点数、总面数
- 平均权重、持久熵
- Dream 统计：合成次数、平均质量

### 4.2 Phase 2：时间线 + 查询 + Dream 可视化

**时间线浏览器**
- 底部水平滑块，拖动显示不同时间点的 mesh 状态
- 播放按钮：自动从旧到新回放记忆生长过程
- 每个时间点高亮新插入的四面体

**查询界面**
- 搜索框输入查询文本
- 3D 视图中高亮匹配结果（脉冲发光动画）
- 结果排名显示混合分数（text/topo/weight 各通道分数）

**Dream Cycle 可视化**
- 触发 dream 后实时动画：
  1. 随机游走路径高亮（黄色轨迹）
  2. 聚类分组闪烁
  3. 新合成四面体从桥接点"生长"出来
- Dream 质量评分仪表盘

### 4.3 Phase 3：高级功能

**批量导入**
- JSON/CSV 文件上传
- 预览 → 确认导入
- 进度条

**拓扑健康监控**
- 持久同调条形图（Betti numbers）
- 熵变化趋势线
- 边界面比例（mesh 封闭度）

**记忆关系图谱**
- 2D 力导向图（D3.js 风格）
- 节点=记忆，边=拓扑连接
- 边粗细=连接强度

**WebSocket 实时更新**
- 其他用户/API 操作时实时刷新 3D 视图
- Dream cycle 进度推送

## 5. 视觉设计

### 5.1 配色方案

```
背景：深色 #0a0a1a（深空感）
普通四面体：权重渐变 蓝(#4fc3f7) → 橙(#ff9800) → 金(#ffd700)
Dream四面体：绿色发光 #00e676，半透明
选中：白色描边 + 脉冲动画
拓扑连线：面=白(0.6), 边=灰(0.3), 顶点=虚线(0.15)
标签颜色：每个标签分配唯一色相
```

### 5.2 四面体渲染

- 使用 Three.js `BufferGeometry` 自定义四面体
- 半透明材质 `MeshPhongMaterial({ transparent: true, opacity: 0.7 })`
- 内部线框 `EdgesGeometry` 显示四面体棱线
- 选中时 `OutlinePass` 发光效果

## 6. 文件结构

```
tetrahedron_memory/
├── web/                        # 前端源码
│   ├── index.html
│   ├── src/
│   │   ├── App.vue
│   │   ├── main.js
│   │   ├── components/
│   │   │   ├── MeshView3D.vue      # Three.js 3D 视图
│   │   │   ├── MemoryPanel.vue     # 记忆详情/编辑
│   │   │   ├── StatsBar.vue        # 统计栏
│   │   │   ├── TimelineSlider.vue  # 时间线
│   │   │   ├── QueryBar.vue        # 查询
│   │   │   ├── DreamPanel.vue      # Dream 控制
│   │   │   └── ImportDialog.vue    # 批量导入
│   │   ├── services/
│   │   │   └── api.js              # API 调用封装
│   │   ├── composables/
│   │   │   ├── useMeshRenderer.js  # Three.js 渲染逻辑
│   │   │   └── useWebSocket.js     # WS 连接
│   │   └── stores/
│   │       └── meshStore.js        # Pinia 状态
│   ├── package.json
│   └── vite.config.js
├── static/                     # 构建产物（git tracked）
│   ├── index.html
│   └── assets/
```

## 7. 后端改动清单

### 7.1 start_api_v2.py 新增端点

```python
# 静态文件服务
app.mount("/ui", StaticFiles(directory="static", html=True))

# 四面体 CRUD
@app.get("/api/v1/tetrahedra")
@app.get("/api/v1/tetrahedra/{tetra_id}")
@app.put("/api/v1/tetrahedra/{tetra_id}")
@app.delete("/api/v1/tetrahedra/{tetra_id}")

# 拓扑图
@app.get("/api/v1/topology-graph")

# 批量导入
@app.post("/api/v1/import")

# WebSocket
@app.websocket("/ws/mesh")
```

### 7.2 数据量评估

当前 30 条记忆 → 拓扑图约 30 节点 + 100 边
预期增长到 500 条 → 约 500 节点 + 2000 边
Three.js 可流畅渲染 1000+ 半透明对象，无需优化

## 8. 开发计划

| 阶段 | 内容 | 预计工时 |
|------|------|----------|
| Phase 1 | 3D 视图 + 记忆面板 + CRUD API | 2-3 天 |
| Phase 2 | 时间线 + 查询高亮 + Dream 动画 | 1-2 天 |
| Phase 3 | 批量导入 + 拓扑监控 + WS 实时 | 1-2 天 |

Phase 1 交付后即可用于论文演示和用户测试。

## 9. 论文价值

可视化界面是 TetraMem 论文的**关键差异化亮点**：
- 截图/视频可直接用于论文 Figure
- 演示时 live 交互比静态图表震撼 10 倍
- "记忆系统可视化"在 AI Memory 领域几乎没有先例
- 证明几何方法的直观性优势——向量嵌入无法可视化

## 10. 与现有系统的关系

- 不影响 MCP adapter（李思琪继续通过 MCP 使用）
- 不影响 API 稳定性（新增端点，不修改现有端点）
- 可选功能——不需要可视化时系统照常运行
- 未来可扩展为多用户协作视图
