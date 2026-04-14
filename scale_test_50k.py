import time, numpy as np, tracemalloc, resource

tracemalloc.start()

from tetrahedron_memory.tetra_mesh import TetraMesh
from tetrahedron_memory.tetra_dream import TetraDreamCycle

def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

print("=" * 70)
print("TetraMem-Hive 50K Scale Test (1-core 2GB server)")
print("=" * 70)

mesh = TetraMesh()
dream = TetraDreamCycle(mesh)

BATCH = 1000
DREAM_EVERY = 5000
MAX = 50000
t0_total = time.time()
t0_batch = t0_total

np.random.seed(42)

for i in range(1, MAX + 1):
    seed_pt = np.random.randn(3)
    mesh.store(
        f"Memory item {i}: exploring eternal geometric integration and self-emergence",
        seed_point=seed_pt,
        labels=[f"t{i % 10}", "scale_test"],
        weight=1.0 + (i % 5) * 0.2,
    )

    if i % BATCH == 0:
        elapsed = time.time() - t0_batch
        rate = BATCH / elapsed
        cur, peak = tracemalloc.get_traced_memory()
        print(f"  {i:5d}/{MAX} | {rate:6.0f} items/s | heap {cur/1e6:.0f}MB | rss {rss_mb():.0f}MB")
        t0_batch = time.time()

    if i % DREAM_EVERY == 0:
        try:
            result = dream.trigger_now()
            fused = result.get("fused", [])
            n_new = len(fused) if isinstance(fused, list) else fused
            avg_q = result.get("avg_quality", 0)
            print(f"  >>> Dream #{i//DREAM_EVERY}: fused={n_new} avg_quality={avg_q:.3f}")
        except Exception as e:
            print(f"  >>> Dream skipped: {e}")

total_time = time.time() - t0_total
cur, peak = tracemalloc.get_traced_memory()
print("\n" + "=" * 70)
print(f"DONE: {MAX} items in {total_time:.1f}s ({MAX/total_time:.0f} items/s avg)")
print(f"Final heap: {cur/1e6:.0f}MB | Peak heap: {peak/1e6:.0f}MB | RSS: {rss_mb():.0f}MB")
print("=" * 70)
tracemalloc.stop()
