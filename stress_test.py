import sys
import os
import time
import random
import string
import threading
import traceback
import json
import gc
import psutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tetrahedron_memory import HoneycombNeuralField

LABELS_POOL = [
    ["ai", "ml", "deep-learning"],
    ["database", "sql", "nosql"],
    ["frontend", "react", "vue"],
    ["backend", "api", "microservice"],
    ["devops", "k8s", "docker"],
    ["security", "auth", "encryption"],
    ["math", "linear-algebra", "statistics"],
    ["physics", "quantum", "relativity"],
    ["biology", "genetics", "evolution"],
    ["chemistry", "organic", "catalyst"],
    ["history", "ancient", "medieval"],
    ["economics", "macro", "trade"],
    ["psychology", "cognitive", "behavioral"],
    ["music", "classical", "jazz"],
    ["art", "painting", "sculpture"],
    ["literature", "poetry", "novel"],
    ["engineering", "civil", "mechanical"],
    ["climate", "weather", "ocean"],
    ["space", "astronomy", "cosmology"],
    ["robotics", "automation", "control"],
]

CONTENT_TEMPLATES = [
    "Research findings on {topic}: {detail}. Key insight: {insight}.",
    "Meeting notes from {date}: Discussed {topic} with {person}. Action items: {action}.",
    "Bug report #{id}: In {module}, when {condition}, error {error} occurs. Root cause: {cause}.",
    "Design document for {project}: Architecture uses {arch}. Trade-offs: {tradeoff}.",
    "Code review feedback on {file}: {feedback}. Suggested fix: {fix}.",
    "Learning note: {concept} means {explanation}. Example: {example}.",
    "Deployment log: Released v{version} to {env}. Changes: {changes}. Status: {status}.",
    "User story #{id}: As a {role}, I want {feature} so that {benefit}. Priority: {priority}.",
    "Performance benchmark: {operation} took {duration}ms with {load} load. Baseline: {baseline}ms.",
    "Architecture decision record #{id}: Chose {choice} over {alternative} because {reason}.",
    "Documentation update for {module}: Added {section} covering {topic}. Reviewer: {reviewer}.",
    "Security audit finding: {vuln} in {component}. Severity: {severity}. Mitigation: {mitigation}.",
    "Data pipeline status: Ingested {count} records from {source}. Quality: {quality}%. Latency: {latency}ms.",
    "Experiment results: Model {model} achieved {metric}={value} on {dataset}. Baseline: {baseline}.",
    "Incident report: {service} degraded at {time}. Impact: {impact}. Resolution: {resolution}.",
    "Sprint retrospective: Went well: {good}. Improve: {improve}. Actions: {actions}.",
    "API endpoint spec: {method} {path} -> {response}. Auth: {auth}. Rate limit: {rate}.",
    "Test coverage report: {module} coverage {pct}%. Missing: {missing}. Priority areas: {areas}.",
    "Configuration change: {key} from {old_val} to {new_val}. Reason: {reason}. Approved by: {approver}.",
    "Memory consolidation: {concept_a} relates to {concept_b} via {relation}. Evidence: {evidence}.",
]

TOPICS = [
    "neural network optimization", "distributed consensus", "data compression",
    "real-time processing", "memory management", "concurrent programming",
    "graph algorithms", "information retrieval", "natural language processing",
    "computer vision", "reinforcement learning", "transfer learning",
    "anomaly detection", "time series forecasting", "recommendation systems",
    "edge computing", "blockchain consensus", "quantum simulation",
    "protein folding", "climate modeling", "fluid dynamics",
    "electromagnetic fields", "signal processing", "control systems",
    "supply chain optimization", "fraud detection", "sentiment analysis",
    "autonomous driving", "medical imaging", "drug discovery",
]

PERSONS = [
    "Alice Chen", "Bob Smith", "Carol Zhang", "David Lee", "Eve Wang",
    "Frank Liu", "Grace Kim", "Henry Wu", "Iris Yang", "Jack Brown",
]


def random_id():
    return str(random.randint(10000, 99999))


def random_content(idx):
    template = CONTENT_TEMPLATES[idx % len(CONTENT_TEMPLATES)]
    return template.format(
        topic=random.choice(TOPICS),
        detail=f"observed behavior pattern {random_id()}",
        insight=f"correlation between variables alpha_{idx % 100}",
        date=f"2026-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
        person=random.choice(PERSONS),
        action=f"follow up on item_{idx}",
        id=random_id(),
        module=f"module_{random.choice(string.ascii_lowercase)}_{idx % 20}",
        condition=f"input exceeds threshold_{random.uniform(0, 1):.3f}",
        error=f"RuntimeError(code={random.randint(100, 999)})",
        cause=f"null pointer in handler_{idx % 50}",
        project=f"project_{random.choice(['alpha', 'beta', 'gamma', 'delta'])}",
        arch=random.choice(["microservices", "monolith", "event-driven", "layered"]),
        tradeoff=f"latency vs consistency at scale_{idx}",
        file=f"src/{random.choice(['core', 'api', 'util', 'model'])}/file_{idx % 100}.py",
        feedback=f"consider using pattern_{random.choice(['factory', 'observer', 'strategy'])}",
        fix=f"refactor to use async_{idx % 30}",
        concept=f"concept_{random.choice(['A', 'B', 'C', 'D'])}_{idx % 200}",
        explanation=f"it handles {random.choice(['edge cases', 'concurrency', 'errors'])}",
        example=f"when x={random.uniform(0, 100):.1f}, result is {random.choice(['optimal', 'stable'])}",
        version=f"{random.randint(1,5)}.{random.randint(0,20)}.{random.randint(0,99)}",
        env=random.choice(["staging", "production", "canary", "dark"]),
        changes=f"{random.randint(1, 20)} features, {random.randint(1, 10)} fixes",
        status=random.choice(["success", "rolling", "complete", "pending"]),
        role=random.choice(["user", "admin", "developer", "manager"]),
        feature=f"capability_{idx % 500}",
        benefit=f"productivity increases by {random.randint(5, 50)}%",
        priority=random.choice(["P0", "P1", "P2", "P3"]),
        operation=random.choice(["insert", "query", "update", "delete"]),
        duration=f"{random.uniform(0.1, 500):.1f}",
        load=f"{random.randint(1, 1000)} req/s",
        baseline=f"{random.uniform(0.1, 500):.1f}",
        choice=random.choice(["Redis", "Memcached", "SQLite", "custom"]),
        alternative=random.choice(["RabbitMQ", "Kafka", "Pulsar", "NATS"]),
        reason=f"benchmark showed {random.randint(10, 90)}% improvement",
        section=random.choice(["overview", "api", "examples", "faq"]),
        reviewer=random.choice(PERSONS),
        vuln=f"CVE-2026-{random.randint(10000, 99999)}",
        component=f"auth-service-{idx % 10}",
        severity=random.choice(["critical", "high", "medium", "low"]),
        mitigation=f"upgrade to version {random.randint(2, 5)}.{random.randint(0, 9)}",
        count=str(random.randint(1000, 1000000)),
        source=random.choice(["database", "api", "file", "stream"]),
        quality=str(random.randint(85, 99)),
        latency=str(random.randint(1, 500)),
        model=random.choice(["gpt", "bert", "t5", "llama"]),
        metric=random.choice(["accuracy", "f1", "precision", "recall"]),
        value=f"{random.uniform(0.7, 0.99):.4f}",
        dataset=random.choice(["imagenet", "cifar", "mnist", "custom"]),
        service=random.choice(["auth", "gateway", "payment", "notification"]),
        impact=f"{random.randint(100, 10000)} users affected",
        time=f"{random.randint(0,23):02d}:{random.randint(0,59):02d}",
        resolution=f"rolled back to v{random.randint(1, 3)}.{random.randint(0, 9)}",
        good=f"delivered {random.randint(5, 20)} stories",
        improve=f"reduce meeting time by {random.randint(10, 50)}%",
        actions=f"implement async standups",
        method=random.choice(["GET", "POST", "PUT", "DELETE"]),
        path=f"/api/v{random.randint(1, 3)}/{random.choice(['users', 'items', 'orders'])}",
        response=f"{{'data': '...'}}",
        auth=random.choice(["JWT", "OAuth2", "API Key", "None"]),
        rate=f"{random.randint(10, 10000)}/min",
        pct=str(random.randint(40, 99)),
        missing=random.choice(["error handlers", "edge cases", "async paths"]),
        areas=random.choice(["core logic", "API layer", "data access"]),
        key=f"config.{random.choice(['timeout', 'retries', 'batch_size'])}",
        old_val=str(random.randint(1, 100)),
        new_val=str(random.randint(1, 100)),
        approver=random.choice(PERSONS),
        concept_a=f"concept_{idx % 100}",
        concept_b=f"concept_{random.randint(0, 99)}",
        relation=random.choice(["causal", "correlational", "hierarchical", "temporal"]),
        evidence=f"experiment_{random_id()} showed p<{random.uniform(0.001, 0.05):.4f}",
    )


class StressTestResult:
    def __init__(self, name):
        self.name = name
        self.total_time = 0
        self.store_times = []
        self.query_times = []
        self.store_errors = 0
        self.query_errors = 0
        self.memory_peak_mb = 0
        self.memory_start_mb = 0
        self.lattice_stats = {}
        self.store_throughput = 0
        self.query_throughput = 0
        self.p95_store = 0
        self.p99_store = 0
        self.p95_query = 0
        self.p99_query = 0
        self.avg_store = 0
        self.avg_query = 0
        self.concurrent_store_throughput = 0
        self.concurrent_query_throughput = 0

    def compute_stats(self):
        if self.store_times:
            self.store_times.sort()
            self.avg_store = sum(self.store_times) / len(self.store_times)
            self.p95_store = self.store_times[int(len(self.store_times) * 0.95)]
            self.p99_store = self.store_times[int(len(self.store_times) * 0.99)]
            self.store_throughput = len(self.store_times) / max(self.total_time, 0.001)
        if self.query_times:
            self.query_times.sort()
            self.avg_query = sum(self.query_times) / len(self.query_times)
            self.p95_query = self.query_times[int(len(self.query_times) * 0.95)]
            self.p99_query = self.query_times[int(len(self.query_times) * 0.99)]
            self.query_throughput = len(self.query_times) / max(self.total_time, 0.001)

    def report(self):
        self.compute_stats()
        lines = [
            f"\n{'='*70}",
            f"  STRESS TEST: {self.name}",
            f"{'='*70}",
            f"  Total time:              {self.total_time:.2f}s",
            f"  Memory:                  {self.memory_start_mb:.1f}MB -> {self.memory_peak_mb:.1f}MB (delta: {self.memory_peak_mb - self.memory_start_mb:.1f}MB)",
            f"",
            f"  --- STORE ({len(self.store_times)} ops, {self.store_errors} errors) ---",
            f"  Throughput:              {self.store_throughput:.1f} ops/sec",
            f"  Avg latency:             {self.avg_store*1000:.2f}ms",
            f"  P95 latency:             {self.p95_store*1000:.2f}ms",
            f"  P99 latency:             {self.p99_store*1000:.2f}ms",
            f"",
            f"  --- QUERY ({len(self.query_times)} ops, {self.query_errors} errors) ---",
            f"  Throughput:              {self.query_throughput:.1f} ops/sec",
            f"  Avg latency:             {self.avg_query*1000:.2f}ms",
            f"  P95 latency:             {self.p95_query*1000:.2f}ms",
            f"  P99 latency:             {self.p99_query*1000:.2f}ms",
            f"",
            f"  --- CONCURRENT ---",
            f"  Concurrent store:        {self.concurrent_store_throughput:.1f} ops/sec",
            f"  Concurrent query:        {self.concurrent_query_throughput:.1f} ops/sec",
            f"",
            f"  --- LATTICE ---",
        ]
        for k, v in self.lattice_stats.items():
            if isinstance(v, float):
                lines.append(f"  {k:30s} {v:.4f}")
            else:
                lines.append(f"  {k:30s} {v}")
        lines.append(f"{'='*70}\n")
        return "\n".join(lines)


def get_memory_mb():
    proc = psutil.Process(os.getpid())
    return proc.memory_info().rss / (1024 * 1024)


def run_stress_test(n_memories, resolution=5, query_ratio=0.2, concurrent_stores=4, concurrent_queries=4, queries_per_worker=50):
    result = StressTestResult(f"N={n_memories}, res={resolution}")

    gc.collect()
    result.memory_start_mb = get_memory_mb()

    print(f"\n[*] Initializing field (resolution={resolution})...")
    field = HoneycombNeuralField(resolution=resolution)
    init_start = time.time()
    stats = field.initialize()
    init_time = time.time() - init_start
    print(f"    Field initialized in {init_time:.2f}s: {stats['total_nodes']} nodes, {stats['occupied_nodes']} occupied")

    ids = []
    t0 = time.time()

    # Phase 1: Sequential store
    print(f"[*] Phase 1: Storing {n_memories} memories sequentially...")
    for i in range(n_memories):
        content = random_content(i)
        labels = LABELS_POOL[i % len(LABELS_POOL)]
        weight = random.uniform(0.5, 3.0)
        try:
            t1 = time.time()
            nid = field.store(content=content, labels=labels, weight=weight)
            t2 = time.time()
            result.store_times.append(t2 - t1)
            ids.append(nid)
        except Exception as e:
            result.store_errors += 1
            if result.store_errors <= 3:
                print(f"    STORE ERROR #{result.store_errors} at i={i}: {e}")

        if (i + 1) % 1000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            mem = get_memory_mb()
            print(f"    [{i+1}/{n_memories}] rate={rate:.0f} ops/s, mem={mem:.0f}MB, errors={result.store_errors}")

    store_phase_time = time.time() - t0
    print(f"    Store phase done: {store_phase_time:.2f}s, {result.store_errors} errors")

    # Phase 2: Sequential query
    n_queries = max(int(n_memories * query_ratio), 100)
    print(f"[*] Phase 2: Querying {n_queries} times sequentially...")
    for i in range(n_queries):
        query_text = random.choice([
            f"research on {random.choice(TOPICS)}",
            f"bug report about {random.choice(['memory', 'network', 'cpu', 'disk'])}",
            f"deployment of {random.choice(['alpha', 'beta', 'gamma'])}",
            f"performance of {random.choice(['insert', 'query', 'update'])}",
            f"design for {random.choice(['microservices', 'monolith', 'event-driven'])}",
            f"security in {random.choice(['auth', 'encryption', 'network'])}",
            f"test coverage for {random.choice(['core', 'api', 'util'])}",
            f"meeting about {random.choice(TOPICS)}",
        ])
        k = random.randint(3, 20)
        query_labels = random.choice(LABELS_POOL)[:1]
        try:
            t1 = time.time()
            res = field.query(text=query_text, k=k, labels=query_labels)
            t2 = time.time()
            result.query_times.append(t2 - t1)
        except Exception as e:
            result.query_errors += 1
            if result.query_errors <= 3:
                print(f"    QUERY ERROR #{result.query_errors} at i={i}: {e}")

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0 - store_phase_time
            rate = (i + 1) / elapsed
            print(f"    [{i+1}/{n_queries}] rate={rate:.0f} ops/s, errors={result.query_errors}")

    query_phase_time = time.time() - t0 - store_phase_time
    print(f"    Query phase done: {query_phase_time:.2f}s, {result.query_errors} errors")

    # Phase 3: Concurrent stores
    print(f"[*] Phase 3: Concurrent stores ({concurrent_stores} threads x {queries_per_worker} ops)...")
    concurrent_store_ids = []
    concurrent_store_errors = [0]
    cs_lock = threading.Lock()
    cs_times = []

    def concurrent_store_worker(wid):
        local_ids = []
        local_times = []
        for i in range(queries_per_worker):
            content = f"concurrent_{wid}_{i}_{random_id()}_{random_content(wid * queries_per_worker + i)}"
            labels = LABELS_POOL[(wid * queries_per_worker + i) % len(LABELS_POOL)]
            try:
                t1 = time.time()
                nid = field.store(content=content, labels=labels, weight=1.0)
                t2 = time.time()
                local_ids.append(nid)
                local_times.append(t2 - t1)
            except Exception:
                with cs_lock:
                    concurrent_store_errors[0] += 1
        with cs_lock:
            concurrent_store_ids.extend(local_ids)
            cs_times.extend(local_times)

    cs_start = time.time()
    threads = [threading.Thread(target=concurrent_store_worker, args=(w,)) for w in range(concurrent_stores)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    cs_elapsed = time.time() - cs_start
    total_cs_ops = concurrent_stores * queries_per_worker
    result.concurrent_store_throughput = total_cs_ops / max(cs_elapsed, 0.001)
    result.store_times.extend(cs_times)
    result.store_errors += concurrent_store_errors[0]
    print(f"    Concurrent store: {cs_elapsed:.2f}s, {result.concurrent_store_throughput:.0f} ops/s, errors={concurrent_store_errors[0]}")

    # Phase 4: Concurrent queries
    print(f"[*] Phase 4: Concurrent queries ({concurrent_queries} threads x {queries_per_worker} ops)...")
    concurrent_query_errors = [0]
    cq_times = []

    def concurrent_query_worker(wid):
        local_times = []
        for i in range(queries_per_worker):
            query_text = random.choice(TOPICS)
            k = random.randint(3, 10)
            try:
                t1 = time.time()
                field.query(text=query_text, k=k)
                t2 = time.time()
                local_times.append(t2 - t1)
            except Exception:
                with cs_lock:
                    concurrent_query_errors[0] += 1
        with cs_lock:
            cq_times.extend(local_times)

    cq_start = time.time()
    threads = [threading.Thread(target=concurrent_query_worker, args=(w,)) for w in range(concurrent_queries)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    cq_elapsed = time.time() - cq_start
    total_cq_ops = concurrent_queries * queries_per_worker
    result.concurrent_query_throughput = total_cq_ops / max(cq_elapsed, 0.001)
    result.query_times.extend(cq_times)
    result.query_errors += concurrent_query_errors[0]
    print(f"    Concurrent query: {cq_elapsed:.2f}s, {result.concurrent_query_throughput:.0f} ops/s, errors={concurrent_query_errors[0]}")

    # Collect final stats
    result.total_time = time.time() - t0
    result.memory_peak_mb = get_memory_mb()
    lattice = field.stats(force=True)
    result.lattice_stats = {
        "total_nodes": lattice.get("total_nodes", 0),
        "occupied_nodes": lattice.get("occupied_nodes", 0),
        "face_edges": lattice.get("face_edges", 0),
        "edge_edges": lattice.get("edge_edges", 0),
        "vertex_edges": lattice.get("vertex_edges", 0),
        "avg_activation": lattice.get("avg_activation", 0),
        "crystal_nodes": lattice.get("crystal_nodes", 0),
        "pulse_count": lattice.get("pulse_count", 0),
        "bridge_count": lattice.get("bridge_count", 0),
        "cascade_count": lattice.get("cascade_count", 0),
    }

    field.stop_pulse_engine()
    del field
    gc.collect()

    return result


def main():
    print("=" * 70)
    print("  TetraMem-XL v7.0 STRESS TEST SUITE")
    print("=" * 70)
    print(f"  Python: {sys.version}")
    print(f"  PID:    {os.getpid()}")
    print(f"  Memory: {get_memory_mb():.0f}MB")
    print()

    all_results = []

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, default=0, help="1=10K, 2=50K, 3=100K, 0=all")
    args = parser.parse_args()

    phases = []
    if args.phase == 0:
        phases = [(1, 10000, "10K"), (2, 50000, "50K"), (3, 100000, "100K")]
    elif args.phase == 1:
        phases = [(1, 10000, "10K")]
    elif args.phase == 2:
        phases = [(2, 50000, "50K")]
    elif args.phase == 3:
        phases = [(3, 100000, "100K")]

    results_map = {}
    for phase_num, n, label in phases:
        print(f"\n{'#' * 70}")
        print(f"# PHASE {phase_num}: {label} MEMORIES")
        print(f"{'#' * 70}")
        r = run_stress_test(n, resolution=5)
        results_map[phase_num] = r
        all_results.append(r)
        print(r.report())

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY COMPARISON")
    print("=" * 70)

    labels_list = []
    r_list = []
    for r in all_results:
        n = r.name.split(",")[0].split("=")[1]
        labels_list.append(n)
        r_list.append(r)

    header = f"  {'Metric':<30s}"
    for l in labels_list:
        header += f" {l:>12s}"
    print(header)
    sep = f"  {'-'*30}"
    for l in labels_list:
        sep += f" {'-'*12}"
    print(sep)

    def row(label, getter):
        line = f"  {label:<30s}"
        for r in r_list:
            v = getter(r)
            line += f" {v:>12s}"
        print(line)

    row("Total time (s)", lambda r: f"{r.total_time:.2f}")
    row("Store throughput (ops/s)", lambda r: f"{r.store_throughput:.1f}")
    row("Store avg latency (ms)", lambda r: f"{r.avg_store*1000:.2f}")
    row("Store P95 latency (ms)", lambda r: f"{r.p95_store*1000:.2f}")
    row("Store P99 latency (ms)", lambda r: f"{r.p99_store*1000:.2f}")
    row("Store errors", lambda r: f"{r.store_errors:d}")
    row("Query throughput (ops/s)", lambda r: f"{r.query_throughput:.1f}")
    row("Query avg latency (ms)", lambda r: f"{r.avg_query*1000:.2f}")
    row("Query P95 latency (ms)", lambda r: f"{r.p95_query*1000:.2f}")
    row("Query P99 latency (ms)", lambda r: f"{r.p99_query*1000:.2f}")
    row("Query errors", lambda r: f"{r.query_errors:d}")
    row("Concurrent store (ops/s)", lambda r: f"{r.concurrent_store_throughput:.1f}")
    row("Concurrent query (ops/s)", lambda r: f"{r.concurrent_query_throughput:.1f}")
    row("Memory peak (MB)", lambda r: f"{r.memory_peak_mb:.1f}")
    row("Memory delta (MB)", lambda r: f"{r.memory_peak_mb-r.memory_start_mb:.1f}")
    row("Lattice total_nodes", lambda r: f"{r.lattice_stats.get('total_nodes',0):d}")
    row("Lattice occupied", lambda r: f"{r.lattice_stats.get('occupied_nodes',0):d}")
    row("Lattice crystal_nodes", lambda r: f"{r.lattice_stats.get('crystal_nodes',0):d}")
    print("=" * 70)

    # Save JSON results
    json_results = []
    for r in all_results:
        r.compute_stats()
        json_results.append({
            "name": r.name,
            "total_time": r.total_time,
            "store_ops": len(r.store_times),
            "store_errors": r.store_errors,
            "store_throughput": r.store_throughput,
            "store_avg_ms": r.avg_store * 1000,
            "store_p95_ms": r.p95_store * 1000,
            "store_p99_ms": r.p99_store * 1000,
            "query_ops": len(r.query_times),
            "query_errors": r.query_errors,
            "query_throughput": r.query_throughput,
            "query_avg_ms": r.avg_query * 1000,
            "query_p95_ms": r.p95_query * 1000,
            "query_p99_ms": r.p99_query * 1000,
            "concurrent_store_throughput": r.concurrent_store_throughput,
            "concurrent_query_throughput": r.concurrent_query_throughput,
            "memory_start_mb": r.memory_start_mb,
            "memory_peak_mb": r.memory_peak_mb,
            "memory_delta_mb": r.memory_peak_mb - r.memory_start_mb,
            "lattice_stats": r.lattice_stats,
        })
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stress_test_results.json")
    with open(out_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\n  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
