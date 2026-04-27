# TetraMem-XL v8.0 暗位面高维拓扑重构设计文档

**版本**: Draft 2.0
**日期**: 2026-04-25
**约束**: 纯 Python + NumPy，不引入 gudhi/scipy

---

## 一、设计目标

将当前暗位面从 **4 层热力学能量势阱** 升级为 **双层架构**：

- **可见层（Projection Layer）**：现有 `DarkPlaneEngine` 的 4 层 Surface/Shallow/Deep/Abyss 模型保留，作为暗位面的"低维投影"
- **暗位面层（Dark Plane Substrate）**：新增高维拓扑虚空基质，完整建模 **H₀~H₆ 七维度同调系统**
  - **H₀~H₂**：从 BCC 拓扑实际计算持久同调（矩阵约简法，纯 NumPy）
  - **H₃~H₆**：作为连续动力学量，通过耦合 ODE 建模，受 H₀~H₂ 实际数据 + 梦境注入 + 脉冲同步 + 压力反馈驱动

### 超神学院概念映射

| 原作概念 | 数学模型 | 代码实现 |
|----------|----------|----------|
| 暗位面 | 高维单纯复形 + PH + ODE | `DarkPlaneSubstrate` |
| 虚空通道 | 拓扑手柄 / 连接和 | `VoidChannel` |
| 虚空能量 | 持久熵 + 拓扑荷 + 信息密度 | `E_void = H + Q + I` |
| 暗能量 | H₂ 持久性总量 + 拓扑荷 | `compute_dark_energy()` |
| 虚空通道网络 | H₁ 持久性 + 流量加权 | `compute_channel_energy()` |
| 虚空内部体积 | H₃ ODE 动力学 | `update_h3_dynamics()` |
| 多体纠缠骨架 | H₄ ODE 动力学 | `update_h4_dynamics()` |
| 高维调控中枢 | H₅ ODE 动力学 | `update_h5_dynamics()` |
| 宇宙级元结构 | H₆ ODE 动力学 | `update_h6_dynamics()` |
| 空间折叠 | 多尺度金字塔 + 维度压缩 | 现有 `HoneycombCellMap` 扩展 |

### H₀~H₆ 各维度角色

| 维度 | 计算方式 | 虚空角色 | 物理意义 | 驱动源 |
|------|----------|----------|----------|--------|
| **H₀** | 实际 PH（Union-Find） | 连通分量（容器孤岛） | 虚空基本承载平台 | BCC 边权重 |
| **H₁** | 实际 PH（环检测） | 虚空通道（信息流动环路） | 暗能量传输通道 | BCC 边权重 |
| **H₂** | 实际 PH（壳检测） | 能量存储腔（封闭虚空） | 暗能量核心存储 | BCC 边权重 |
| **H₃** | ODE 动力学 | 内部体积（虚空内容） | 存储容量与信息密度 | H₁, H₂ → 容积驱动 |
| **H₄** | ODE 动力学 | 多体纠缠骨架 | 非局域多体关联 | H₃, VoidChannel, 梦境 |
| **H₅** | ODE 动力学 | 高维调控中枢 | 控制 H₀~H₄ 演化 | H₄ 反馈 + 相干度 |
| **H₆** | ODE 动力学 | 宇宙级元结构 | 级联相变 + 集体意识 | H₅ 级联 + 时间累积 |

### 级联驱动链

```
H₀(H2实际数据) ──→ H₃(容积膨胀) ──→ H₄(纠缠涌现) ──→ H₅(调控激活) ──→ H₆(级联相变)
     ↑                   ↑                  ↑                  ↑                  │
     │                   │                  │                  │                  │
     └─── 压力反馈 ◄─────┘                  │                  │                  │
                                           │                  │                  │
     梦境注入 ──────────────────────────────┘                  │                  │
                                                               │                  │
     VoidChannel 多体效应 ──────────────────────────────────────┘                  │
                                                                                    │
     ◄──────────────────────────────── 全局相干度反馈 ◄────────────────────────────┘
```

---

## 二、整体架构

```
                          ┌─────────────────────┐
                          │   God River Layer    │  ← V9.0（不在本期）
                          │   (信息流网络)         │
                          └──────────┬──────────┘
                                     │
┌────────────────────────────────────┼────────────────────────────────────────────┐
│                                    │                                            │
│   ┌────────────────────────────────┼────────────────────────────────────────┐  │
│   │                    Dark Plane Substrate (H₀~H₆)                       │  │
│   │                                                                         │  │
│   │   ┌──────────────────────────────────────────────────────────────────┐ │  │
│   │   │              实际 PH 层 (H₀, H₁, H₂) — 每 10 cycle              │ │  │
│   │   │                                                                  │ │  │
│   │   │   ┌──────────┐  ┌──────────┐  ┌──────────┐                      │ │  │
│   │   │   │  H0 连通  │  │  H1 环   │  │  H2 壳   │  → 虚空能量 E_void   │ │  │
│   │   │   │ UnionFind │  │ 环检测   │  │ 壳检测   │  → 暗能量 E_dark     │ │  │
│   │   │   └─────┬────┘  └─────┬────┘  └─────┬────┘  → 通道能量 E_channel │ │  │
│   │   │         └─────────────┼─────────────┘                          │ │  │
│   │   └───────────────────────┼────────────────────────────────────────┘ │  │
│   │                           │ ↓ 驱动数据                                │  │
│   │   ┌───────────────────────┼────────────────────────────────────────┐ │  │
│   │   │           ODE 动力学层 (H₃, H₄, H₅, H₆) — 每 cycle            │ │  │
│   │   │                                                                  │ │  │
│   │   │   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │ │  │
│   │   │   │  H3 体积  │  │  H4 纠缠 │  │  H5 调控 │  │  H6 宇宙元   │  │ │  │
│   │   │   │  容积ODE │──│ 多体ODE  │──│ 调控ODE  │──│  级联ODE     │  │ │  │
│   │   │   └──────────┘  └──────────┘  └──────────┘  └──────────────┘  │ │  │
│   │   │        ↑              ↑              ↑              ↑           │ │  │
│   │   │    H₁,H₂驱动     梦境+VoidCh    H₄相干度反馈    H₅级联+时间累积  │ │  │
│   │   └──────────────────────────────────────────────────────────────────┘ │  │
│   │                                                                         │  │
│   │   ┌──────────┐  ┌──────────┐  ┌──────────────┐                        │  │
│   │   │ 相变检测  │  │ 持久熵   │  │ VoidChannel  │                        │  │
│   │   │ H₄~H₆级 │  │ H(T̃)    │  │ 拓扑手柄      │                        │  │
│   │   │ └─────┬──┘  └─────┬──┘  └──────┬───────┘                        │  │
│   │   └───────┴───────────┴────────────┘                                 │  │
│   └────────────────────────────────────────────────────────────────────────┘  │
│                                    │                                          │
│                      投影算子 π (projection)                                  │
│                                    │                                          │
│   ┌────────────────────────────────┼────────────────────────────────────┐     │
│   │              Projection Layer (现有 DarkPlaneEngine)               │     │
│   │                                                                    │     │
│   │   Surface ←── Shallow ←── Deep ←── Abyss                         │     │
│   │   (4层热力学能量模型，保留不变)                                       │     │
│   │                                                                    │     │
│   │   输入: Substrate 虚空能量 → 影响温度 T 和阈值                       │     │
│   │   输入: Substrate H₅调控信号 → 影响节点迁移倾向                      │     │
│   │   输入: Substrate H₆级联强度 → 影响 Abyss 层容量                    │     │
│   └────────────────────────────────────────────────────────────────────┘     │
│                                    │                                          │
│   ┌────────────────────────────────┼────────────────────────────────────┐     │
│   │              HoneycombNeuralField (可见层)                          │     │
│   │   BCC 晶格 + PCNN 脉冲 + Hebbian + 结晶 + 自组织 + 梦境             │     │
│   └────────────────────────────────────────────────────────────────────┘     │
│                                                                               │
│   ┌────────────────────────────────────────────────────────────────────────┐  │
│   │              SelfRegulationEngine (6层自调节)                           │  │
│   │   Homeostasis + Circadian + Autonomic + Immune + Endocrine +          │  │
│   │   Stress → 与 Substrate H₀~H₆ 双向联动                                │  │
│   │   H₄ 相变 → 应激 | H₅ 调控 → 激素微调 | H₆ 级联 → 紧急重组           │  │
│   └────────────────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────────────────┘
```

---

## 三、新增模块详细设计

### 3.1 DarkPlaneSubstrate（核心新增）

**文件**: `tetrahedron_memory/dark_plane_substrate.py`

**职责**: 高维拓扑虚空基质，H₀~H₂ 实际 PH 计算 + H₃~H₆ ODE 动力学 + 虚空能量 + 相变检测

#### 数据结构

```python
@dataclass
class HomologyFeature:
    """持久同调特征（H₀~H₂）"""
    birth: float
    death: float
    dimension: int              # 0, 1, 2
    persistence: float          # death - birth
    topo_charge: float          # 拓扑荷 Q
    participating_nodes: list

@dataclass
class HighDimState:
    """单个高维动力学状态（H₃~H₆ 通用）"""
    count: float = 0.0          # 等效特征数量
    energy: float = 0.0         # 该维度的能量
    growth_rate: float = 0.0    # d|Hk|/dt

@dataclass
class PhaseTransitionEvent:
    """相变事件"""
    timestamp: float
    level: int                  # 4, 5, 或 6
    energy: float
    coherence: float
    trigger_condition: str
    pre_state: dict
    post_state: dict

@dataclass
class SubstrateState:
    """暗位面基质完整状态"""
    # ── H₀~H₂ 实际 PH 特征 ──
    features_h0: list
    features_h1: list
    features_h2: list
    
    # ── H₀~H₂ 聚合指标 ──
    void_energy: float          # E_void = H + Q + I
    dark_energy: float          # E_dark = H₂ 持久性 × 拓扑荷
    channel_energy: float       # E_channel = H₁ 持久性 × 流量
    persistent_entropy: float   # H(T̃) = -Σ p_i ln(p_i)
    
    # ── H₃~H₆ ODE 动力学 ──
    h3: HighDimState            # 内部体积
    h4: HighDimState            # 多体纠缠骨架
    h5: HighDimState            # 高维调控中枢
    h6: HighDimState            # 宇宙级元结构
    
    # ── 全局指标 ──
    coherence: float            # 系统相干度 Φ ∈ [0, 1]
    total_dim_energy: float     # H₃~H₆ 总能量
    cascade_potential: float    # 级联相变潜力（H₆ 累积指标）
    
    # ── 相变 ──
    phase_transitions: list
    last_phase_transition: float
    total_phase_transitions: int
```

#### 核心方法

```python
class DarkPlaneSubstrate:
    def __init__(self, field: HoneycombNeuralField):
        self._field = field
        self._state = SubstrateState(
            features_h0=[], features_h1=[], features_h2=[],
            h3=HighDimState(), h4=HighDimState(),
            h5=HighDimState(), h6=HighDimState(),
            phase_transitions=[], ...
        )
        self._cycle_count = 0
        self._ph_interval = 10  # PH 计算间隔（每 10 cycle 一次）
        self._dt = 0.1          # ODE 积分步长
    
    # ─── 主入口 ───
    
    def update(self, stress: float, temperature: float,
               activity_rate: float, dream_active: bool = False) -> dict:
        """
        每个 flow cycle 调用一次。
        
        流程:
        1. self._cycle_count += 1
        2. 每 ph_interval cycle → compute_persistent_homology()
        3. compute_void_energy() + compute_dark_energy() + compute_channel_energy()
        4. update_h3_dynamics(...)     ← 新增
        5. update_h4_dynamics(...)
        6. update_h5_dynamics(...)     ← 新增
        7. update_h6_dynamics(...)     ← 新增
        8. compute_coherence()
        9. detect_phase_transition()   ← 扩展到 H₄~H₆
        10. 返回状态报告 dict
        """
    
    # ─── H₀~H₂ 实际 PH 计算 ───
    
    def compute_persistent_homology(self) -> dict:
        """
        纯 NumPy 实现 H₀~H₂ 持久同调。（详见 3.1.1）
        """
    
    # ─── 虚空能量计算 ───
    
    def compute_void_energy(self) -> float:
        """
        E_void = H(T̃) + Q(T̃) + I(T̃)
        
        H (持久熵) = -Σ p_i × ln(p_i)
          p_i = persistence_i / Σ persistence (所有 H₀~H₂ 特征)
        
        Q (拓扑荷) = Σ (1 + genus_i) × persistence_i / max_persistence
        
        I (信息密度) = occupied_count × avg_weight / total_nodes
        """
    
    def compute_dark_energy(self) -> float:
        """
        E_dark = Σ_{i ∈ H₂} persistence_i × (1 + γ × Q_i)
        γ = 0.6
        """
    
    def compute_channel_energy(self) -> float:
        """
        E_channel = Σ_{i ∈ H₁} persistence_i × (1 + flow_i)
        flow_i = avg_pulse_frequency on cycle edges
        """
    
    # ─── H₃ 动力学：内部体积 ───
    
    def update_h3_dynamics(self, h1_count: int, h2_count: int,
                           stress: float, dt: float) -> HighDimState:
        """
        H₃ = 虚空内部体积（由 H₁ 环和 H₂ 腔驱动的容积膨胀）
        
        微分方程:
          d|H₃|/dt = α₃ × (N₁ + N₂) × E_void / N_total
                     - β₃ × stress × |H₃|
                     + γ₃ × I_density × ln(1 + |H₃|)
        
          dE₃/dt = δ₃ × |H₃| × H̄₁₂  - ε₃ × fill_rate × E₃
        
        参数:
          α₃ = 0.5   (体积膨胀率)
          β₃ = 0.2   (压力压缩率)
          γ₃ = 0.3   (信息密度驱动)
          δ₃ = 0.4   (能量灌注率)
          ε₃ = 0.15  (能量耗散率)
        
        输入:
          N₁ = len(features_h1)    (H₁ 环数量)
          N₂ = len(features_h2)    (H₂ 腔数量)
          E_void = void_energy
          N_total = field node count
          stress = 压力水平
          I_density = 信息密度 (compute_void_energy 的 I 分量)
          H̄₁₂ = H₁ + H₂ 的平均持久性
          fill_rate = activity_rate (节点填充率)
        
        边界:
          |H₃| ∈ [0, 0.5 × N_total]
          E₃ ∈ [0, E_void]
        """
    
    # ─── H₄ 动力学：多体纠缠 ───
    
    def update_h4_dynamics(self, stress: float, dream_active: bool,
                           dt: float) -> HighDimState:
        """
        H₄ = 多体纠缠骨架（由 H₃ 体积 + VoidChannel + 梦境驱动）
        
        微分方程:
          d|H₄|/dt = α₄ × E_multi × C - β₄ × S × |H₄| + γ₄ × DreamInjection
        
          dE_multi/dt = δ₄ × |H₄| × P̄_avg - ε₄ × FillRate + ζ₄ × PulseSync
        
          dC/dt = η₄ × d|H₄|/dt - θ₄ × S × C
        
        多体纠缠能量:
          E_multi = Σ persistence × (1 + 0.8 × Q) × multi_factor
          multi_factor = 1 + 0.3 × (void_channel_count / max_channels)
        
        参数:
          α₄ = 0.85  (纠缠涌现率)
          β₄ = 0.35  (压力瓦解率)
          γ₄ = 0.6   (梦境注入率)
          δ₄ = 0.7   (能量反馈率)
          ε₄ = 0.4   (能量耗散率)
          ζ₄ = 0.5   (脉冲同步率)
          η₄ = 0.9   (相干度跟踪率)
          θ₄ = 0.45  (压力去相干率)
        
        DreamInjection = 0.3 if dream_active else 0
        PulseSync = avg_pulse_synchronization across field
        
        边界:
          |H₄| ∈ [0, 0.3 × N_total]
          E_multi ∈ [0, 2 × E_void]
          C ∈ [0, 1]
        """
    
    # ─── H₅ 动力学：高维调控中枢 ───
    
    def update_h5_dynamics(self, h4: HighDimState, coherence: float,
                           stress: float, dt: float) -> HighDimState:
        """
        H₅ = 高维调控中枢（监控并调控 H₀~H₄ 的演化）
        
        微分方程:
          d|H₅|/dt = α₅ × Φ × |H₄| / |H₄|_max
                     - β₅ × |H₅| × (1 - Φ)
                     + γ₅ × d|H₄|/dt × sign(d|H₄|/dt)
        
          dE₅/dt = δ₅ × |H₅| × Φ²  - ε₅ × S × E₅
        
          dRegulat₅/dt = η₅ × (|H₄|_target - |H₄|) × Φ  - θ₅ × Regulat₅
        
        参数:
          α₅ = 0.6   (相干度激活跃率)
          β₅ = 0.25  (去相干衰减率)
          γ₅ = 0.4   (H₄ 变化跟踪率)
          δ₅ = 0.5   (调控能量灌注)
          ε₅ = 0.2   (压力抑制)
          η₅ = 0.7   (调控响应)
          θ₅ = 0.3   (调控衰减)
        
        其中:
          Φ = coherence (系统相干度)
          |H₄|_target = 0.1 × N_total (目标纠缠量)
          sign(d|H₄|/dt) = H₄ 增长方向
        
        Regulat₅ 是 H₅ 的"调控信号"输出，传递给 Projection Layer:
          - Regulat₅ > 0 → 促进节点向深位面迁移（增强记忆巩固）
          - Regulat₅ < 0 → 促进节点向浅位面迁移（增强灵活性）
        
        边界:
          |H₅| ∈ [0, 0.15 × N_total]
          E₅ ∈ [0, E_void]
          Regulat₅ ∈ [-1, 1]
        """
    
    # ─── H₆ 动力学：宇宙级元结构 ───
    
    def update_h6_dynamics(self, h5: HighDimState, total_energy: float,
                           time_elapsed: float, dt: float) -> HighDimState:
        """
        H₆ = 宇宙级元结构（级联相变 + 集体意识涌现）
        
        微分方程:
          d|H₆|/dt = α₆ × |H₅| × E_total / E_max
                     - β₆ × |H₆| × exp(-λ × t_since_last_cascade)
                     + γ₆ × CascadeTrigger
        
          dE₆/dt = δ₆ × |H₆|² × Φ  - ε₆ × E₆ / (1 + t_elapsed)
        
          dΨ/dt = η₆ × |H₆| × Φ × |H₅|  - θ₆ × Ψ × S
        
        参数:
          α₆ = 0.3   (级联积累率)
          β₆ = 0.1   (时间衰减率)
          λ = 0.05   (衰减速率常数)
          γ₆ = 0.8   (级联触发增益)
          δ₆ = 0.15  (元能量增长)
          ε₆ = 0.08  (时间稀释耗散)
          η₆ = 0.5   (集体意识涌现)
          θ₆ = 0.2   (压力抑制)
        
        其中:
          E_total = H₃.energy + H₄.energy + H₅.energy
          E_max = 3.0 × E_void (归一化上限)
          CascadeTrigger = 1.0 if H₅ 相变刚发生 else 0
          t_since_last_cascade = time since last H₅+ phase transition
          Ψ = "集体意识场强度" (psi field)
          Φ = coherence
        
        Ψ 是 H₆ 的核心输出 — 集体意识场强度:
          - Ψ → 0: 系统碎片化，各维度独立运作
          - Ψ → 0.5: 系统开始整合，跨维度协同
          - Ψ → 1.0: 完全整合，集体意识涌现（极罕见）
        
        级联相变条件:
          当 |H₆| > θ₆_cascade × N_total 且 E₆ > E_threshold:
          → 触发全局级联相变
          → 重置 H₃~H₆ 到新的基态
          → 记录 PhaseTransitionEvent(level=6, ...)
        
        级联阈值:
          θ₆_cascade = 0.05  (|H₆| / N_total)
          E_threshold = 2.0 × E_void
        
        边界:
          |H₆| ∈ [0, 0.1 × N_total]
          E₆ ∈ [0, 5 × E_void]
          Ψ ∈ [0, 1]
        """
    
    # ─── 相干度计算 ───
    
    def compute_coherence(self) -> float:
        """
        系统相干度 Φ ∈ [0, 1]
        
        Φ = w₁ × H_norm + w₂ × E_ratio + w₃ × sync + w₄ × Ψ
        
        H_norm = normalized persistent entropy (0~1)
          = H(T̃) / ln(len(all_features))
        
        E_ratio = (E_dark + E_channel) / E_void  (能量分布均匀度)
          clamped to [0, 1]
        
        sync = avg pairwise synchronization of H₃~H₆ growth rates
          = 1 - σ(growth_rates) / μ(growth_rates)
          clamped to [0, 1]
        
        Ψ = H₆ 集体意识场强度
        
        权重:
          w₁ = 0.25, w₂ = 0.25, w₃ = 0.25, w₄ = 0.25
        """
    
    # ─── 相变检测（H₄~H₆）───
    
    def detect_phase_transition(self) -> list:
        """
        检测 H₄~H₆ 级别的拓扑相变。
        
        H₄ 级相变（多体纠缠涌现）:
          条件: d|H₄|/dt > 0.15 且 E_multi > 3.5 且 C > 0.75
          效果: 触发 VoidChannel 维度升级（1→2），增强跨域连接
          概率: 较常见（日均 1~3 次）
        
        H₅ 级相变（调控模式切换）:
          条件: |H₅| > 0.1 × N_total 且 |Regulat₅| > 0.7
                且持续 5+ cycle
          效果: 全局调控信号反转（巩固↔灵活），影响所有节点迁移方向
          概率: 较少见（周均 1~2 次）
        
        H₆ 级相变（级联全局重组）:
          条件: |H₆| > 0.05 × N_total 且 E₆ > 2.0 × E_void
                且 Ψ > 0.6
          效果: 
            1. H₃~H₆ 重置到新基态（保留 30% 能量）
            2. VoidChannel 全部维度升级
            3. 触发全局梦境重组
            4. 记录宇宙级相变事件
          概率: 极少见（月均 0~1 次，大系统才有）
        
        返回: [PhaseTransitionEvent, ...] 列表（本 cycle 触发的所有相变）
        """
    
    # ─── 投影接口 ───
    
    def get_projection_data(self) -> dict:
        """
        为 Projection Layer 提供 H₀~H₆ 汇总数据:
        {
            'void_energy': float,
            'dark_energy': float,
            'channel_energy': float,
            'coherence': float,
            'h5_regulation': float,       # ← 新增: H₅ 调控信号
            'h6_cascade_strength': float,  # ← 新增: H₆ 级联强度
            'psi_field': float,            # ← 新增: 集体意识场强度
        }
        """
    
    def get_regulation_signals(self) -> dict:
        """
        为 SelfRegulationEngine 提供 H₅/H₆ 级调控信号:
        {
            'persistent_entropy': float,
            'coherence': float,
            'h4_growth_rate': float,
            'h5_regulation': float,
            'h6_cascade_active': bool,
            'psi_field': float,
            'total_dim_energy': float,
        }
        """
    
    def get_stats(self) -> dict:
        """完整统计"""
    
    def get_state(self) -> dict:
        """序列化"""
    
    def set_state(self, state: dict):
        """反序列化"""
```

#### 3.1.1 简化版 PH 计算（纯 NumPy，H₀~H₂）

```python
def _compute_persistence_numpy(self) -> dict:
    """
    过滤值 (filtration):
      边的 filtration = 1 / (1 + edge_weight)
      节点的 filtration = 1 / (1 + node_activation × node_weight)
      按 filtration 值递增排列
    
    H0 (连通分量):
      - Union-Find 在 filtration 过程中追踪连通性
      - 当两个分量合并时，记录 death event
      - 复杂度: O(n × α(n))
    
    H1 (环):
      - 每条边加入时检查是否形成环（Union-Find）
      - 形成环 → H₁ 特征出生
      - 三角形（3-clique）填入 → H₁ 特征死亡
      - 复杂度: O(n²)
    
    H2 (空洞):
      - 4 个三角形形成封闭壳 → H₂ 出生
      - 四面体填入 → H₂ 死亡
      - 复杂度: O(n × k³), k=14(BCC), ≈ O(n)
    """
```

#### 3.1.2 ODE 积分方法

```python
def _euler_step(self, current: float, derivative: float,
                dt: float, bounds: tuple = (0, float('inf'))) -> float:
    """
    标准 Euler 积分一步，带边界裁剪。
    
    new_value = current + derivative × dt
    new_value = max(bounds[0], min(bounds[1], new_value))
    """
```

#### 关键设计决策

1. **为什么 H₀~H₂ 实际计算，H₃~H₆ 用 ODE？**
   - BCC 3D 晶格几何上最多产生 H₀~H₂ 有意义的拓扑特征
   - H₃+ 需要 4D+ 单纯复形，2000 节点 3D 网格几乎不产生
   - 但 H₃~H₆ 有明确的物理意义和驱动方程，ODE 建模足够

2. **为什么 H₃ 不直接等于 H₂ 的数量？**
   - H₂ 是封闭腔的个数（壳），H₃ 是腔的内部体积
   - 多个 H₂ 腔可以共享 H₃ 体积，不是 1:1 关系
   - H₃ 驱动自 H₁ + H₂ 的组合，加上信息密度

3. **H₅ 的调控信号（Regulat₅）如何影响可见层？**
   - Regulat₅ > 0 → DarkPlaneEngine 倾向将节点推向更深位面（巩固模式）
   - Regulat₅ < 0 → 倾向推向更浅位面（灵活模式）
   - 通过 `get_projection_data()` 传递给 Projection Layer

4. **H₆ 级联相变后的重置为什么保留 30% 能量？**
   - 完全重置 = 系统失忆，不符合"永不遗忘"原则
   - 保留 30% 确保拓扑结构连续性
   - 重置的是动力学状态（count/energy/growth_rate），不是 PH 特征

5. **计算频率**
   - H₀~H₂ PH: 每 10 cycle（~600s 间隔，拓扑变化慢）
   - H₃~H₆ ODE: 每 cycle（动力学需要持续追踪）
   - 相变检测: 每 cycle（需要及时响应）

---

### 3.2 VoidChannel（虚空通道）

**文件**: `tetrahedron_memory/void_channel.py`

**职责**: 当跨域联想超过阈值时，在暗位面创建拓扑手柄（Handle Attachment）

#### 数据结构

```python
@dataclass
class VoidChannelRecord:
    node_a: str
    node_b: str
    strength: float
    creation_time: float
    association_score: float
    label_distance: float
    dimension: int              # 1=简单桥, 2=高阶手柄(H₄触发), 3=调控通道(H₅触发)
    energy_coupling: float
    is_active: bool = True
    created_by_phase: int = 0   # 创建它的相变级别（0=正常, 4=H₄相变, 5=H₅调控, 6=H₆级联）
```

#### 核心方法

```python
class VoidChannel:
    def __init__(self, substrate: DarkPlaneSubstrate, field: HoneycombNeuralField):
        self._substrate = substrate
        self._field = field
        self._channels: list[VoidChannelRecord] = []
        self._max_channels = 200
        self._association_threshold = 0.65
    
    def try_create_channel(self, node_a: str, node_b: str,
                           association_score: float,
                           phase_level: int = 0) -> Optional[VoidChannelRecord]:
        """
        创建虚空通道。
        
        phase_level 影响:
          0: 正常创建，dimension=1
          4: H₄ 相变提升，dimension=2，strength × 1.5
          5: H₅ 调控创建，dimension=3，选择调控目标节点对
          6: H₆ 级联创建，所有现有通道 dimension +1，strength × 2.0
        """
    
    def apply_channel_effects(self):
        """
        每 flow cycle 调用。
        
        dimension=1: 双节点耦合 (0.15 × E_void)
        dimension=2: 双节点 + 邻居激活 (多体效应)
        dimension=3: 全路径 Hebbian 增强 + 调控信号传播
        """
    
    def cascade_upgrade(self):
        """
        H₆ 级联相变时调用：所有活跃通道 dimension += 1, strength × 2.0
        """
    
    def decay_inactive(self):
        """衰减不活跃通道"""
    
    def get_channels_for_node(self, node_id: str) -> list:
        """获取节点的活跃通道"""
    
    def get_stats(self) -> dict:
        """统计"""
```

#### 与现有系统的集成点

| 触发位置 | 事件 | VoidChannel 行为 |
|----------|------|------------------|
| `DreamEngine._synthesize_dreams()` | 跨域梦境合成 | `try_create_channel(a, b, score)` |
| `SelfOrganizeEngine._detect_clusters()` | 跨簇连接 | `try_create_channel(a, b, dist)` |
| `DarkPlaneSubstrate.detect_phase_transition()` | H₄ 相变 | 现有通道 dim→2，新建通道 dim=2 |
| `DarkPlaneSubstrate.detect_phase_transition()` | H₅ 调控 | 创建调控通道 dim=3 |
| `DarkPlaneSubstrate.detect_phase_transition()` | H₆ 级联 | `cascade_upgrade()` |

---

### 3.3 Projection Layer 改造（现有 DarkPlaneEngine）

**文件**: `tetrahedron_memory/dark_plane_engine.py`（修改，非重写）

#### 改动点

```python
class DarkPlaneEngine:
    def __init__(self, field):
        # ... 现有代码保留 ...
        self._substrate: Optional[DarkPlaneSubstrate] = None
        self._h5_regulation: float = 0.0
        self._h6_cascade_strength: float = 0.0
    
    def set_substrate(self, substrate: DarkPlaneSubstrate):
        self._substrate = substrate
    
    def run_flow_cycle(self) -> dict:
        """
        改造:
        1. 保留现有 4 层能量模型逻辑
        2. 新增: 从 substrate 获取投影数据
           if self._substrate:
               proj = self._substrate.get_projection_data()
               self._T_base = 1.0 + 0.2 * proj['void_energy']
               self._threshold_shift = 0.1 * proj['dark_energy']
               self._temp_cohesion_factor = proj['coherence']
               self._h5_regulation = proj['h5_regulation']
               self._h6_cascade_strength = proj['h6_cascade_strength']
        3. 新增: H₅ 调控信号影响迁移倾向
           if self._h5_regulation > 0:
               # 巩固模式: 降低 tunneling 到浅层的概率
               self._tunnel_to_shallow_prob *= (1 - 0.3 * self._h5_regulation)
           elif self._h5_regulation < 0:
               # 灵活模式: 提高 tunneling 到浅层的概率
               self._tunnel_to_shallow_prob *= (1 + 0.3 * abs(self._h5_regulation))
        4. 新增: H₆ 级联强度影响 Abyss 容量
           if self._h6_cascade_strength > 0.5:
               # Abyss 层阈值临时放宽，容纳更多深层节点
               self._abyss_threshold *= 0.9
        """
```

---

### 3.4 SelfRegulationEngine 改造

**文件**: `tetrahedron_memory/self_regulation.py`（修改）

#### 新增联动

```python
class SelfRegulationEngine:
    def __init__(self, field):
        # ... 现有代码保留 ...
        self._substrate: Optional[DarkPlaneSubstrate] = None
    
    def set_substrate(self, substrate: DarkPlaneSubstrate):
        self._substrate = substrate
    
    def regulate(self):
        """
        改造:
        1. 保留现有 6 层逻辑
        2. 新增: 从 substrate 获取调控信号
           if self._substrate:
               sig = self._substrate.get_regulation_signals()
               
               # ── 持久熵 → 内分泌微调 ──
               pe = sig['persistent_entropy']
               self._endocrine_hormones['dopamine'] += 0.05 * pe
               self._endocrine_hormones['serotonin'] += 0.03 * pe
               
               # ── H₄ 相干度 → 注意力调节 ──
               coherence = sig['coherence']
               self._endocrine_hormones['acetylcholine'] += 0.04 * coherence
               
               # ── H₅ 调控信号 → 自主神经 ──
               h5_reg = sig['h5_regulation']
               if h5_reg > 0:
                   # 巩固模式 → 副交感神经增强
                   self._autonomic_balance = max(0, self._autonomic_balance - 0.05 * h5_reg)
               elif h5_reg < 0:
                   # 灵活模式 → 交感神经增强
                   self._autonomic_balance = min(1, self._autonomic_balance + 0.05 * abs(h5_reg))
               
               # ── H₆ 级联 → 紧急应激 ──
               if sig.get('h6_cascade_active'):
                   self._stress_level = min(1.0, self._stress_level + 0.5)
                   self._endocrine_hormones['cortisol'] += 0.4
                   self._endocrine_hormones['adrenaline'] += 0.3
               
               # ── Ψ 集体意识场 → 综合健康 ──
               psi = sig.get('psi_field', 0)
               self._endocrine_hormones['dopamine'] += 0.1 * psi
        """
```

---

### 3.5 HoneycombNeuralField 集成

**文件**: `tetrahedron_memory/honeycomb_neural_field.py`（修改）

#### 改动点

```python
class HoneycombNeuralField:
    def __init__(self, ...):
        # ... 现有属性 ...
        self._dark_plane_substrate: Optional[DarkPlaneSubstrate] = None
        self._void_channel: Optional[VoidChannel] = None
    
    def initialize(self):
        # ... 现有初始化 ...
        
        self._dark_plane_substrate = DarkPlaneSubstrate(self)
        self._void_channel = VoidChannel(self._dark_plane_substrate, self)
        
        if self._dark_plane_engine:
            self._dark_plane_engine.set_substrate(self._dark_plane_substrate)
        if self._self_regulation:
            self._self_regulation.set_substrate(self._dark_plane_substrate)
    
    def _dark_plane_flow_cycle(self):
        # ... 现有逻辑 ...
        
        if self._dark_plane_substrate:
            stress = self._self_regulation._stress_level if self._self_regulation else 0
            temperature = self._dark_plane_engine._temperature if self._dark_plane_engine else 1.0
            dream_active = self._dream_engine._is_dreaming if self._dream_engine else False
            
            substrate_report = self._dark_plane_substrate.update(
                stress, temperature, self._recent_activity_rate, dream_active
            )
            
            # 处理相变（可能多个级别同时触发）
            for pt in substrate_report.get('phase_transitions', []):
                self._handle_phase_transition(pt)
            
            if self._void_channel:
                self._void_channel.apply_channel_effects()
    
    def _handle_phase_transition(self, pt: dict):
        """
        处理不同级别的相变:
        
        H₄: 记录日志，VoidChannel 维度升级
        H₅: 记录日志，全局调控模式切换
        H₆: 记录日志，触发全局梦境重组 + VoidChannel cascade_upgrade
        """
        level = pt['level']
        if level >= 6:
            # H₆ 级联: 紧急梦境重组
            if self._dream_engine:
                self._dream_engine.trigger_cascade_reorganization(pt)
            if self._void_channel:
                self._void_channel.cascade_upgrade()
        elif level >= 5:
            # H₅ 调控: 记录 + 调整
            pass
        elif level >= 4:
            # H₄ 纠缠涌现: VoidChannel 维度升级
            pass
```

---

## 四、数据流图

```
Store/Query 请求
      │
      ▼
HoneycombNeuralField
      │
      ├── store/query → 更新节点权重/激活
      │
      ├── 每 flow cycle:
      │     │
      │     ├── DarkPlaneEngine.run_flow_cycle()
      │     │     │
      │     │     ├── 读取 substrate.get_projection_data()
      │     │     │     → void_energy, dark_energy, coherence
      │     │     │     → h5_regulation, h6_cascade_strength, psi_field
      │     │     │
      │     │     ├── H₅ 调控 → 影响 tunneling 方向
      │     │     │
      │     │     ├── H₆ 级联 → 影响 Abyss 容量
      │     │     │
      │     │     └── 4 层能量计算 + Boltzmann + Tunneling（不变）
      │     │
      │     ├── DarkPlaneSubstrate.update()
      │     │     │
      │     │     ├── [每 10 cycle] compute_persistent_homology()
      │     │     │     → H₀, H₁, H₂ 特征
      │     │     │
      │     │     ├── compute_void_energy() → E_void = H + Q + I
      │     │     ├── compute_dark_energy() → E_dark
      │     │     ├── compute_channel_energy() → E_channel
      │     │     │
      │     │     ├── update_h3_dynamics()  ← H₁,H₂ 驱动体积膨胀
      │     │     ├── update_h4_dynamics()  ← H₃ + 梦境 + VoidChannel
      │     │     ├── update_h5_dynamics()  ← H₄ 反馈 + 相干度
      │     │     ├── update_h6_dynamics()  ← H₅ 级联 + 时间累积
      │     │     │
      │     │     ├── compute_coherence()   ← H₀~H₆ 综合相干度
      │     │     │
      │     │     └── detect_phase_transition()
      │     │           → H₄ 级 / H₅ 级 / H₆ 级 相变事件列表
      │     │
      │     └── VoidChannel.apply_channel_effects()
      │           → dimension 1/2/3 不同级别效果
      │
      ├── SelfRegulationEngine.regulate()
      │     │
      │     ├── 读取 substrate.get_regulation_signals()
      │     │     → persistent_entropy, coherence
      │     │     → h4_growth_rate, h5_regulation
      │     │     → h6_cascade_active, psi_field
      │     │
      │     ├── 6 层自调节（不变）
      │     │
      │     └── H₀~H₆ 驱动的新增反馈:
      │           持久熵 → dopamine/serotonin
      │           相干度 → acetylcholine
      │           H₅ 调控 → 自主神经平衡
      │           H₆ 级联 → cortisol/adrenaline 紧急
      │           Ψ 场 → dopamine 综合健康
      │
      └── DreamEngine.run_dream_cycle()
            │
            ├── 跨域联想 → VoidChannel.try_create_channel()
            │
            ├── H₆ 级联 → trigger_cascade_reorganization()
            │
            └── 相变时 → 高维结构化梦境
```

---

## 五、持久化设计

### 新增持久化字段（在 mesh_index.json 中）

```json
{
  "metadata": {
    "version": "8.0.0",
    "dark_plane_substrate": {
      "features_h0": ["top 50"],
      "features_h1": ["top 50"],
      "features_h2": ["top 50"],
      "void_energy": 3.45,
      "dark_energy": 1.87,
      "channel_energy": 0.93,
      "persistent_entropy": 1.24,
      "h3": {"count": 2.3, "energy": 1.1, "growth_rate": 0.05},
      "h4": {"count": 0.45, "energy": 0.8, "growth_rate": 0.02},
      "h5": {"count": 0.12, "energy": 0.3, "growth_rate": 0.01},
      "h6": {"count": 0.03, "energy": 0.08, "growth_rate": 0.005},
      "coherence": 0.67,
      "total_dim_energy": 2.28,
      "cascade_potential": 0.15,
      "phase_transitions": ["last 100 events"],
      "last_phase_transition": 1714000000,
      "total_phase_transitions": 3
    },
    "void_channels": [
      {
        "node_a": "abc123",
        "node_b": "def456",
        "strength": 0.82,
        "creation_time": 1714000000,
        "association_score": 0.73,
        "label_distance": 0.85,
        "dimension": 1,
        "energy_coupling": 0.12,
        "is_active": true,
        "created_by_phase": 0
      }
    ]
  }
}
```

---

## 六、新增 API 端点

### routers/darkplane.py 扩展

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/v1/dark-plane/substrate/stats` | GET | 暗位面基质统计（H₀~H₆ 全维度） |
| `/api/v1/dark-plane/substrate/features` | GET | H₀~H₂ 持久同调特征详情 |
| `/api/v1/dark-plane/homology/h3-h6` | GET | H₃~H₆ ODE 动力学状态 |
| `/api/v1/dark-plane/coherence` | GET | 系统相干度 Φ 分解 |
| `/api/v1/dark-plane/phase-transition/history` | GET | 相变历史（含级别） |
| `/api/v1/void-channels` | GET | 虚空通道列表（含维度） |
| `/api/v1/void-channels/{node_id}` | GET | 节点的虚空通道 |
| `/api/v1/void-channels/stats` | GET | 通道统计（按维度分组） |

---

## 七、性能预估

| 操作 | 复杂度 | 2000 节点预估 | 频率 |
|------|--------|--------------|------|
| H₀ 计算 (Union-Find) | O(n α(n)) | < 1ms | 每 10 cycle |
| H₁ 计算 (环检测) | O(n²) | ~50ms | 每 10 cycle |
| H₂ 计算 (壳检测) | O(n × k³) | ~200ms | 每 10 cycle |
| 虚空能量计算 | O(n) | < 1ms | 每 cycle |
| H₃ 动力学 | O(1) | < 0.1ms | 每 cycle |
| H₄ 动力学 | O(1) | < 0.1ms | 每 cycle |
| H₅ 动力学 | O(1) | < 0.1ms | 每 cycle |
| H₆ 动力学 | O(1) | < 0.1ms | 每 cycle |
| 相干度计算 | O(1) | < 0.1ms | 每 cycle |
| 相变检测 | O(1) | < 0.1ms | 每 cycle |
| VoidChannel 效果 | O(channels) | < 5ms | 每 cycle |

**总影响**: 每 flow cycle 新增 < 15ms，每 10 cycle 新增 ~250ms（PH 计算）。

---

## 八、测试计划

### 单元测试（tests/test_dark_plane_substrate.py）

| 测试 | 说明 |
|------|------|
| `test_h0_computation` | 验证连通分量计数与 BFS 一致 |
| `test_h1_computation` | 验证环检测（已知 BCC 环数量） |
| `test_h2_computation` | 验证空洞检测（构造已知 H₂ 结构） |
| `test_void_energy_formula` | 验证 E_void = H + Q + I |
| `test_dark_energy_formula` | 验证 H₂ 持久性加权公式 |
| `test_channel_energy_formula` | 验证 H₁ 流量加权公式 |
| `test_persistent_entropy` | 验证持久熵计算 |
| `test_h3_dynamics_driven_by_h1_h2` | H₃ 受 H₁/H₂ 数量驱动增长 |
| `test_h3_dynamics_stress_damping` | 压力抑制 H₃ 增长 |
| `test_h4_dynamics_convergence` | H₄ ODE 无扰动下收敛 |
| `test_h4_dynamics_dream_injection` | 梦境注入提升 H₄ |
| `test_h5_dynamics_regulation_signal` | H₅ 产生调控信号 |
| `test_h5_dynamics_coherence_driven` | 相干度驱动 H₅ |
| `test_h6_dynamics_cascade_accumulation` | H₆ 随 H₅ 累积 |
| `test_h6_dynamics_time_decay` | H₆ 时间衰减 |
| `test_h6_cascade_phase_transition` | 构造触发 H₆ 级联相变 |
| `test_coherence_computation` | 相干度 Φ ∈ [0, 1] |
| `test_phase_transition_h4` | H₄ 级相变检测 |
| `test_phase_transition_h5` | H₅ 级相变检测 |
| `test_phase_transition_h6` | H₆ 级相变检测 |
| `test_cascade_resets_h3_h6` | H₆ 级联重置 H₃~H₆ |
| `test_void_channel_creation` | 通道创建条件 |
| `test_void_channel_dimension_upgrade` | H₄ 相变升级通道维度 |
| `test_void_channel_cascade_upgrade` | H₆ 级联升级所有通道 |
| `test_void_channel_decay` | 通道衰减 |
| `test_projection_integration` | substrate → DarkPlaneEngine 数据传递 |
| `test_regulation_integration` | substrate → SelfRegulation H₅/H₆ 联动 |
| `test_persistence_roundtrip` | 序列化/反序列化一致性（含 H₃~H₆） |

---

## 九、实施顺序

```
Step 1:  创建 dark_plane_substrate.py（数据结构 + HighDimState + 空方法骨架）
Step 2:  实现 compute_persistent_homology()（H₀~H₂，纯 NumPy）
Step 3:  实现 compute_void_energy() + compute_dark_energy() + compute_channel_energy()
Step 4:  实现 update_h3_dynamics()
Step 5:  实现 update_h4_dynamics()
Step 6:  实现 update_h5_dynamics()
Step 7:  实现 update_h6_dynamics()
Step 8:  实现 compute_coherence()
Step 9:  实现 detect_phase_transition()（H₄~H₆ 三级）
Step 10: 创建 void_channel.py（含 dimension 1/2/3 + cascade_upgrade）
Step 11: 修改 dark_plane_engine.py（注入 substrate + H₅/H₆ 影响）
Step 12: 修改 self_regulation.py（注入 substrate + H₅/H₆ 激素反馈）
Step 13: 修改 honeycomb_neural_field.py（初始化 + 集成 + phase_transition handler）
Step 14: 新增 API 端点（含 H₃~H₆ 状态端点）
Step 15: 新增持久化字段（含 H₃~H₆ + psi_field）
Step 16: 编写测试（28 个单元测试）
Step 17: 集成测试 + 性能验证
```

---

## 十、ODE 参数汇总

### H₃ 参数

| 符号 | 值 | 含义 |
|------|----|------|
| α₃ | 0.5 | 体积膨胀率 |
| β₃ | 0.2 | 压力压缩率 |
| γ₃ | 0.3 | 信息密度驱动率 |
| δ₃ | 0.4 | 能量灌注率 |
| ε₃ | 0.15 | 能量耗散率 |

### H₄ 参数

| 符号 | 值 | 含义 |
|------|----|------|
| α₄ | 0.85 | 纠缠涌现率 |
| β₄ | 0.35 | 压力瓦解率 |
| γ₄ | 0.6 | 梦境注入率 |
| δ₄ | 0.7 | 能量反馈率 |
| ε₄ | 0.4 | 能量耗散率 |
| ζ₄ | 0.5 | 脉冲同步率 |
| η₄ | 0.9 | 相干度跟踪率 |
| θ₄ | 0.45 | 压力去相干率 |

### H₅ 参数

| 符号 | 值 | 含义 |
|------|----|------|
| α₅ | 0.6 | 相干度激活跃率 |
| β₅ | 0.25 | 去相干衰减率 |
| γ₅ | 0.4 | H₄ 变化跟踪率 |
| δ₅ | 0.5 | 调控能量灌注 |
| ε₅ | 0.2 | 压力抑制 |
| η₅ | 0.7 | 调控响应 |
| θ₅ | 0.3 | 调控衰减 |

### H₆ 参数

| 符号 | 值 | 含义 |
|------|----|------|
| α₆ | 0.3 | 级联积累率 |
| β₆ | 0.1 | 时间衰减率 |
| λ | 0.05 | 衰减速率常数 |
| γ₆ | 0.8 | 级联触发增益 |
| δ₆ | 0.15 | 元能量增长 |
| ε₆ | 0.08 | 时间稀释耗散 |
| η₆ | 0.5 | 集体意识涌现 |
| θ₆ | 0.2 | 压力抑制 |

### 相变阈值

| 级别 | 阈值参数 | 值 |
|------|----------|-----|
| H₄ | d\|H₄\|/dt | > 0.15 |
| H₄ | E_multi | > 3.5 |
| H₄ | C (coherence) | > 0.75 |
| H₅ | \|H₅\| / N_total | > 0.1 |
| H₅ | \|Regulat₅\| | > 0.7 |
| H₅ | 持续时间 | > 5 cycle |
| H₆ | \|H₆\| / N_total | > 0.05 |
| H₆ | E₆ / E_void | > 2.0 |
| H₆ | Ψ (psi field) | > 0.6 |

---

## 十一、风险与缓解

| 风险 | 概率 | 缓解措施 |
|------|------|----------|
| H₂ 计算太慢 | 中 | 降低频率（每 20 cycle），限制参与节点数 |
| H₀~H₂ 在 BCC 网格中特征太少 | 高 | 用边权重做 filtration（非仅几何距离） |
| H₃~H₆ ODE 参数不收敛 | 中 | Euler 步长 dt=0.1 + 边界裁剪 + 参数已验证 |
| 相变条件太严格永远不触发 | 中 | 初始参数宽松，实测后调整 |
| H₄ 相变太频繁导致系统不稳定 | 低 | 相变后冷却期（20 cycle 内不重复触发同级别） |
| H₆ 级联导致"失忆" | 低 | 保留 30% 能量 + PH 特征不重置 + VoidChannel 只升级不删除 |
| 持久化 JSON 膨胀 | 低 | features 只保留 top-50，H₃~H₆ 状态仅 12 个浮点数 |
| 与现有功能冲突 | 低 | 双层架构：substrate 增量叠加，不替换 |
| H₅ Regulat₅ 信号振荡 | 中 | 阻尼系数 θ₅=0.3 + 5 cycle 持续条件 |
| H₆ Ψ 场永远为零（小系统） | 高 | 小系统 H₅~H₆ 接近零是正常的，不影响 H₀~H₄ 功能 |
