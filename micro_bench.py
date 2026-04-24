import sys, os, time, gc, random, string, json
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
    N = 500
    Q = 100

    gc.collect()
    m0 = mem_mb()
    print(f"Init field (res=5)...")
    t0 = time.time()
    field = HoneycombNeuralField(resolution=5)
    field.initialize()
    print(f"  init: {time.time()-t0:.2f}s, mem={mem_mb():.0f}MB")

    ids = []
    store_times = []
    errors = 0

    print(f"\nStoring {N} memories...")
    t0 = time.time()
    for i in range(N):
        content = random_content(i)
        labels = LABELS_POOL[i % len(LABELS_POOL)]
        weight = random.uniform(0.5, 3.0)
        try:
            t1 = time.time()
            nid = field.store(content=content, labels=labels, weight=weight)
            t2 = time.time()
            store_times.append(t2 - t1)
            ids.append(nid)
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  ERROR at i={i}: {e}")

        if (i+1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i+1) / elapsed
            st = field.stats(force=True)
            print(f"  [{i+1}/{N}] rate={rate:.0f}/s store_avg={sum(store_times[-50:])/len(store_times[-50:])*1000:.1f}ms nodes={st['total_nodes']} occ={st['occupied_nodes']} mem={mem_mb():.0f}MB")

    total_store = time.time() - t0
    store_times.sort()
    print(f"\nSTORE RESULTS ({N} ops, {errors} errors, {total_store:.2f}s):")
    print(f"  throughput:    {N/total_store:.1f} ops/s")
    print(f"  avg latency:   {sum(store_times)/len(store_times)*1000:.2f}ms")
    print(f"  p50 latency:   {store_times[len(store_times)//2]*1000:.2f}ms")
    print(f"  p95 latency:   {store_times[int(len(store_times)*0.95)]*1000:.2f}ms")
    print(f"  p99 latency:   {store_times[int(len(store_times)*0.99)]*1000:.2f}ms")
    print(f"  max latency:   {store_times[-1]*1000:.2f}ms")

    st = field.stats(force=True)
    print(f"\nLATTICE: total={st['total_nodes']} occupied={st['occupied_nodes']} faces={st['face_edges']} edges={st['edge_edges']}")

    # Latency distribution
    buckets = [1, 5, 10, 50, 100, 500, 1000, 5000]
    print(f"\nSTORE LATENCY DISTRIBUTION:")
    prev = 0
    for b in buckets:
        count = sum(1 for t in store_times if prev <= t*1000 < b)
        pct = count/len(store_times)*100
        print(f"  {prev:>5}ms - {b:>5}ms: {count:>4} ({pct:>5.1f}%)")
        prev = b
    count = sum(1 for t in store_times if t*1000 >= prev)
    pct = count/len(store_times)*100
    print(f"  {prev:>5}ms+       : {count:>4} ({pct:>5.1f}%)")

    # Query test
    print(f"\nQuerying {Q} times...")
    query_times = []
    t0 = time.time()
    for i in range(Q):
        qt = random.choice(["neural", "database", "frontend", "backend", "security", "math"])
        k = random.randint(3, 10)
        try:
            t1 = time.time()
            res = field.query(text=qt, k=k)
            t2 = time.time()
            query_times.append(t2 - t1)
        except Exception as e:
            print(f"  QUERY ERROR: {e}")

    total_query = time.time() - t0
    query_times.sort()
    if query_times:
        print(f"\nQUERY RESULTS ({Q} ops, {total_query:.2f}s):")
        print(f"  throughput:    {Q/total_query:.1f} ops/s")
        print(f"  avg latency:   {sum(query_times)/len(query_times)*1000:.2f}ms")
        print(f"  p50 latency:   {query_times[len(query_times)//2]*1000:.2f}ms")
        print(f"  p95 latency:   {query_times[int(len(query_times)*0.95)]*1000:.2f}ms")
        print(f"  p99 latency:   {query_times[-1]*1000:.2f}ms")

        print(f"\nQUERY LATENCY DISTRIBUTION:")
        prev = 0
        for b in [1, 5, 10, 50, 100, 500, 1000]:
            count = sum(1 for t in query_times if prev <= t*1000 < b)
            pct = count/len(query_times)*100
            print(f"  {prev:>5}ms - {b:>5}ms: {count:>4} ({pct:>5.1f}%)")
            prev = b
        count = sum(1 for t in query_times if t*1000 >= prev)
        pct = count/len(query_times)*100
        print(f"  {prev:>5}ms+       : {count:>4} ({pct:>5.1f}%)")

    m1 = mem_mb()
    print(f"\nMEMORY: {m0:.0f}MB -> {m1:.0f}MB (delta={m1-m0:.0f}MB)")

    # Profile store phases - breakdown for a single store call
    print(f"\n{'='*60}")
    print(f"  PROFILING: Individual store call phases")
    print(f"{'='*60}")
    
    # Warm a fresh field
    pfield = HoneycombNeuralField(resolution=5)
    pfield.initialize()
    
    # Pre-fill some data
    for i in range(50):
        pfield.store(content=f"prefill_{i}", labels=["test"], weight=1.0)
    
    # Time a store call with internal instrumentation
    # We'll wrap the key methods
    orig_find = pfield._find_nearest_empty_node
    orig_emit = pfield._emit_pulse
    
    find_times = []
    emit_times = []
    
    def timed_find(*args, **kwargs):
        t = time.time()
        r = orig_find(*args, **kwargs)
        find_times.append(time.time() - t)
        return r
    
    def timed_emit(*args, **kwargs):
        t = time.time()
        r = orig_emit(*args, **kwargs)
        emit_times.append(time.time() - t)
        return r
    
    pfield._find_nearest_empty_node = timed_find
    pfield._emit_pulse = timed_emit
    
    n_profile = 100
    total_stores = []
    for i in range(n_profile):
        content = f"profile_{i}_{random.randint(0,99999)}"
        t = time.time()
        pfield.store(content=content, labels=["profile"], weight=1.5)
        total_stores.append(time.time() - t)
    
    print(f"  Profiled {n_profile} store calls:")
    print(f"  Total store avg:    {sum(total_stores)/len(total_stores)*1000:.2f}ms")
    print(f"  _find_nearest avg:  {sum(find_times)/len(find_times)*1000:.2f}ms ({sum(find_times)/sum(total_stores)*100:.1f}% of total)")
    print(f"  _emit_pulse avg:    {sum(emit_times)/len(emit_times)*1000:.2f}ms ({sum(emit_times)/sum(total_stores)*100:.1f}% of total)")
    print(f"  Other overhead avg: {(sum(total_stores)-sum(find_times)-sum(emit_times))/len(total_stores)*1000:.2f}ms ({(sum(total_stores)-sum(find_times)-sum(emit_times))/sum(total_stores)*100:.1f}% of total)")

if __name__ == "__main__":
    main()
