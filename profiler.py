import sys, os, time, random, string, gc
import psutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tetrahedron_memory import HoneycombNeuralField

def mem_mb():
    return psutil.Process(os.getpid()).memory_info().rss / (1024*1024)

LABELS_POOL = [
    ["ai","ml"], ["db","sql"], ["web","react"], ["api","rest"],
    ["devops","k8s"], ["sec","auth"], ["math","calc"], ["phys","quantum"],
]

def random_content(i):
    topics = ["neural net", "database", "frontend", "backend", "devops", "security", "math", "physics"]
    return f"Research note {i}: findings on {random.choice(topics)} with metric_{random.uniform(0,1):.4f} and key insight {random.randint(0,9999)}"

def main():
    N = 2000
    gc.collect()

    field = HoneycombNeuralField(resolution=5)
    field.initialize()
    print(f"Field ready: {len(field._nodes)} nodes")

    # Pre-fill
    for i in range(N):
        content = random_content(i)
        labels = LABELS_POOL[i % len(LABELS_POOL)]
        field.store(content=content, labels=labels, weight=random.uniform(0.5, 3.0))
    print(f"Pre-filled {N} memories. Lattice: {len(field._nodes)} nodes, {field._occupied_count} occupied")
    print(f"  Label index sizes: {[(k, len(v)) for k, v in list(field._label_index.items())[:8]]}")

    # Profile _find_nearest_empty_node phases
    import time

    t_total = 0
    t_cell_map = 0
    t_face_loop = 0
    t_edge_loop = 0
    t_fallback = 0

    n_profile = 100
    for i in range(n_profile):
        content = f"profile_{i}_{random.randint(0,99999)}"
        labels = LABELS_POOL[i % len(LABELS_POOL)]
        label_set = set(labels)

        # Phase 1: find_optimal_placement_cells
        t0 = time.time()
        optimal_cells = field._cell_map.find_optimal_placement_cells(
            field._nodes, label_set, field._label_index, count=20
        )
        t1 = time.time()
        t_cell_map += (t1 - t0)

        # Phase 2: face candidates from label-matched occupied
        related_occ_ids = set()
        for lbl in label_set:
            related_occ_ids.update(field._label_index.get(lbl, set()))
        related_occ_ids = related_occ_ids & field._occupied_ids

        t0 = time.time()
        face_candidates = []
        for oid in related_occ_ids:
            on = field._nodes.get(oid)
            if not on:
                continue
            for fnid in on.face_neighbors:
                fn = field._nodes.get(fnid)
                if fn and not fn.is_occupied:
                    vacancy = field._vacancy_attraction_fast(fnid) * 0.5
                    cell_q = field._cell_quality_factor(fnid)
                    face_candidates.append((fnid, 10 + cell_q * 3 + vacancy))
        t1 = time.time()
        t_face_loop += (t1 - t0)

        # Full store call
        t0 = time.time()
        field.store(content=content, labels=labels, weight=1.0)
        t1 = time.time()
        t_total += (t1 - t0)

    print(f"\nProfiled {n_profile} calls at {N} memories:")
    print(f"  Full store avg:           {t_total/n_profile*1000:.2f}ms")
    print(f"  find_optimal_cells avg:   {t_cell_map/n_profile*1000:.2f}ms ({t_cell_map/t_total*100:.1f}%)")
    print(f"  face_candidates avg:      {t_face_loop/n_profile*1000:.2f}ms ({t_face_loop/t_total*100:.1f}%)")
    print(f"  other (emit_pulse etc):   {(t_total-t_cell_map-t_face_loop)/n_profile*1000:.2f}ms ({(t_total-t_cell_map-t_face_loop)/t_total*100:.1f}%)")

    # Profile cell_map internals
    print(f"\nCell map stats:")
    print(f"  Total cells: {len(field._cell_map._cells)}")
    t0 = time.time()
    for _ in range(10):
        field._cell_map.find_optimal_placement_cells(
            field._nodes, {"ai", "ml"}, field._label_index, count=20
        )
    t1 = time.time()
    print(f"  find_optimal_placement_cells x10: {(t1-t0)*1000:.2f}ms")

    # Test _vacancy_attraction_fast
    sample_id = random.choice(list(field._occupied_ids))
    node = field._nodes[sample_id]
    empty_neighbors = [fnid for fnid in node.face_neighbors if fnid in field._nodes and not field._nodes[fnid].is_occupied]
    if empty_neighbors:
        t0 = time.time()
        for _ in range(100):
            field._vacancy_attraction_fast(empty_neighbors[0])
        t1 = time.time()
        print(f"  _vacancy_attraction_fast x100: {(t1-t0)*1000:.2f}ms ({(t1-t0)/100*1000:.3f}ms each)")

    # Test _cell_quality_factor
    if empty_neighbors:
        t0 = time.time()
        for _ in range(100):
            field._cell_quality_factor(empty_neighbors[0])
        t1 = time.time()
        print(f"  _cell_quality_factor x100: {(t1-t0)*1000:.2f}ms ({(t1-t0)/100*1000:.3f}ms each)")

if __name__ == "__main__":
    main()
