from .consistency import (
    CompensationLog,
    ConflictRecord,
    ConsistencyManager,
    VectorClock,
    VersionedNode,
)
from .core import GeoMemoryBody, MemoryNode, QueryResult
from .topology_organizer import DreamCycle, TopologySelfOrganizer
from .tetra_mesh import FaceRecord, MemoryTetrahedron, TetraMesh
from .tetra_dream import TetraDreamCycle, default_synthesis
from .tetra_self_org import TetraSelfOrganizer
from .persistent_entropy import (
    EntropyTracker,
    compute_persistent_entropy,
    compute_entropy_by_dimension,
    compute_entropy_delta,
    should_trigger_integration,
)
from .closed_loop import (
    ClosedLoopEngine,
    LoopPhase,
)
from .llm_tool import (
    TOOL_DEFINITIONS,
    create_tool_response,
    execute_tool_call,
    get_tool_definitions,
)
from .geometry import (
    GeometryPrimitives,
    TextToGeometryMapper,
    SemanticEmbedder,
    weighted_tetra_power_radius,
)
from .multimodal import PixHomology
from .partitioning import (
    BoundingBox,
    BucketActor,
    GhostCell,
    M3NOPartitioner,
    Octree,
    SpatialBucketRouter,
    TetraMemRayController,
    global_coarse_grid_sync,
)
from .persistence import (
    MemoryPersistence,
    ParquetPersistence,
    RayController,
    RemoteBucketActor,
    S3Storage,
)

# Lazy imports for optional dependencies -- these succeed even when
# prometheus_client / fastapi are not installed.
try:
    from .monitoring import (
        ASSOCIATE_COUNTER,
        ERROR_COUNTER,
        NODE_COUNT_GAUGE,
        QUERY_COUNTER,
        SELF_ORGANIZE_COUNTER,
        STORE_COUNTER,
        WEIGHT_HISTOGRAM,
        DREAM_COUNTER,
        ENTROPY_GAUGE,
        INTEGRATION_COUNTER,
        QUERY_LATENCY,
        STORE_LATENCY,
        get_grafana_dashboard_json,
        get_metrics_registry,
        get_ray_cluster_status,
        health_check,
        increment_counter,
        observe_histogram,
        record_error,
        set_gauge,
    )
    from .monitoring import (  # noqa: F401
        generate_metrics as generate_prometheus_metrics,
    )
except ImportError:
    pass

try:
    from .router import create_app  # noqa: F401
    from .router import create_app as create_router
except ImportError:
    pass

__all__ = [
    "GeoMemoryBody",
    "MemoryNode",
    "QueryResult",
    "MemoryPersistence",
    "ParquetPersistence",
    "RayController",
    "S3Storage",
    "RemoteBucketActor",
    "PixHomology",
    "TextToGeometryMapper",
    "Octree",
    "M3NOPartitioner",
    "BoundingBox",
    "GhostCell",
    "BucketActor",
    "TetraMemRayController",
    "SpatialBucketRouter",
    "global_coarse_grid_sync",
    "VersionedNode",
    "VectorClock",
    "CompensationLog",
    "ConsistencyManager",
    "get_tool_definitions",
    "execute_tool_call",
    "create_tool_response",
    "TOOL_DEFINITIONS",
    "TetraMesh",
    "increment_counter",
    "set_gauge",
    "observe_histogram",
    "record_error",
    "get_metrics_registry",
    "get_ray_cluster_status",
    "get_grafana_dashboard_json",
    "health_check",
    "STORE_COUNTER",
    "QUERY_COUNTER",
    "ASSOCIATE_COUNTER",
    "SELF_ORGANIZE_COUNTER",
    "NODE_COUNT_GAUGE",
    "WEIGHT_HISTOGRAM",
    "ERROR_COUNTER",
    "STORE_LATENCY",
    "QUERY_LATENCY",
    "ENTROPY_GAUGE",
    "INTEGRATION_COUNTER",
    "DREAM_COUNTER",
    "create_app",
    "create_router",
    "TopologySelfOrganizer",
    "DreamCycle",
    "MemoryTetrahedron",
    "FaceRecord",
    "TetraSelfOrganizer",
    "MultimodalBridge",
    "TetraBucket",
    "TetraMeshRouter",
    "TetraDistributedController",
    "GlobalCoarseMesh",
    "default_synthesis",
    "TetraDreamCycle",
    "weighted_tetra_power_radius",
    "GeometryPrimitives",
    "compute_persistent_entropy",
    "compute_entropy_by_dimension",
    "compute_entropy_delta",
    "should_trigger_integration",
    "EntropyTracker",
    "ClosedLoopEngine",
    "LoopPhase",
    "ZigzagTracker",
    "PersistenceSnapshot",
    "TopologicalTransition",
    "ResolutionPyramid",
    "PyramidNode",
    "MultiParameterQuery",
    "FilterCriteria",
    "MultiParamResult",
    "EternityAudit",
    "MappingConeRecord",
    "SemanticEmbedder",
]

from .circuit_breaker import CircuitBreaker, EmergenceProtector, RateLimiter
from .global_coarse_mesh import GlobalCoarseMesh
from .tetra_router import TetraBucket, TetraMeshRouter
from .tetra_distributed import TetraDistributedController
from .multimodal_bridge import MultimodalBridge
from .emergence import AdaptiveThreshold, EmergencePressure
from .zigzag_persistence import (
    ZigzagTracker,
    PersistenceSnapshot,
    TopologicalTransition,
    MappingConeRecord,
)
from .resolution_pyramid import ResolutionPyramid, PyramidNode
from .multiparameter_filter import MultiParameterQuery, FilterCriteria, MultiParamResult
from .eternity_audit import EternityAudit
