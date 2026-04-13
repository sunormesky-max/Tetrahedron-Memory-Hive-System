import threading
import time

import numpy as np
import pytest

from tetrahedron_memory.core import GeoMemoryBody
from tetrahedron_memory.partitioning import SpatialBucketRouter

slow = pytest.mark.slow


def _unique_content(i: int) -> str:
    return f"stress-test-memory-item-{i:07d}-payload-" + "x" * 40


@slow
def test_sequential_store_1m():
    body = GeoMemoryBody(dimension=3, precision="fast")
    start = time.perf_counter()
    n = 1_000_000
    for i in range(n):
        body.store(content=_unique_content(i), weight=np.random.rand())
    elapsed = time.perf_counter() - start
    ops_per_sec = n / elapsed
    stats = body.get_statistics()
    print(
        f"\n[1M sequential store] {elapsed:.2f}s, {ops_per_sec:,.0f} ops/sec, nodes={stats['total_nodes']}"
    )
    assert stats["total_nodes"] == n
    assert ops_per_sec > 100


@slow
def test_batch_query_100k():
    body = GeoMemoryBody(dimension=3, precision="fast")
    n = 100_000
    for i in range(n):
        body.store(content=_unique_content(i), labels=[f"batch{i % 100}"])
    start = time.perf_counter()
    queries = 100
    for i in range(queries):
        body.query(_unique_content(i * 100), k=5)
    elapsed = time.perf_counter() - start
    qps = queries / elapsed
    print(f"\n[100K batch query] {queries} queries in {elapsed:.2f}s, {qps:,.0f} qps")
    assert qps > 1


@slow
def test_bucket_router_100k():
    router = SpatialBucketRouter(max_points_per_bucket=10000)
    router.initialize()
    start = time.perf_counter()
    n = 100_000
    for i in range(n):
        geom = np.random.uniform(-1.0, 1.0, size=3)
        router.route_store(geom, content=_unique_content(i), weight=np.random.rand())
    elapsed = time.perf_counter() - start
    ops_per_sec = n / elapsed
    stats = router.get_statistics()
    print(
        f"\n[100K bucket router] {elapsed:.2f}s, {ops_per_sec:,.0f} ops/sec, buckets={stats['total_buckets']}"
    )
    assert stats["total_nodes"] == n


@slow
def test_self_organize_10k():
    body = GeoMemoryBody(dimension=3, precision="fast")
    n = 10_000
    for i in range(n):
        body.store(content=_unique_content(i), weight=np.random.uniform(0.1, 2.0))
    start = time.perf_counter()
    result = body.self_organize()
    elapsed = time.perf_counter() - start
    print(
        f"\n[10K self-organize] {elapsed:.2f}s, actions={result.get('actions', 0)}, "
        f"repulsions={result.get('repulsions', 0)}, caves={result.get('cave_growths', 0)}"
    )
    assert isinstance(result, dict)


@slow
def test_concurrent_store_50k():
    body = GeoMemoryBody(dimension=3, precision="fast")
    n = 50_000
    errors: list = []
    start = time.perf_counter()

    def worker(offset: int, count: int) -> None:
        try:
            for i in range(offset, offset + count):
                body.store(content=_unique_content(i), weight=np.random.rand())
        except Exception as e:
            errors.append(str(e))

    threads = []
    num_threads = 4
    per_thread = n // num_threads
    for t in range(num_threads):
        th = threading.Thread(target=worker, args=(t * per_thread, per_thread))
        threads.append(th)
        th.start()
    for th in threads:
        th.join()

    elapsed = time.perf_counter() - start
    ops_per_sec = n / elapsed
    stats = body.get_statistics()
    print(
        f"\n[50K concurrent store] {elapsed:.2f}s, {ops_per_sec:,.0f} ops/sec, "
        f"nodes={stats['total_nodes']}, errors={len(errors)}"
    )
    assert len(errors) == 0
    assert stats["total_nodes"] == n


@slow
def test_memory_growth_profile():
    body = GeoMemoryBody(dimension=3, precision="fast")
    import sys

    checkpoints = [1_000, 10_000, 50_000, 100_000]
    sizes = []
    prev = 0
    for target in checkpoints:
        for i in range(prev, target):
            body.store(content=_unique_content(i), weight=1.0)
        size_bytes = sys.getsizeof(body._nodes)
        for node in list(body._nodes.values())[: min(100, len(body._nodes))]:
            size_bytes += sys.getsizeof(node.content) + sys.getsizeof(node.geometry)
        sizes.append((target, size_bytes))
        prev = target

    print("\n[Memory growth profile]")
    for count, sz in sizes:
        print(f"  {count:>8,} nodes -> {sz / 1024 / 1024:>8.2f} MB (est.)")
    assert len(sizes) == len(checkpoints)
