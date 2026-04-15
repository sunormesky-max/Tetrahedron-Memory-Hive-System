# TetraMem-XL v3.0 — Execution Checklist

## Phase 1: LLM Dream Synthesis (Week 1-3)

### Week 1
- [ ] Create branch `feat/phase1-llm-integration`
- [ ] Write `tetrahedron_memory/llm_integration.py` (~250 lines)
  - [ ] LLMProvider ABC + AnthropicProvider
  - [ ] OpenAIProvider, GLMProvider, OllamaProvider
  - [ ] LLMDreamExecutor (think/execute/reflect)
  - [ ] create_executor() factory
- [ ] Verify imports: `python -c "from tetrahedron_memory.llm_integration import create_executor"`

### Week 2
- [ ] Write `tests/test_llm_dream.py` (~150 lines)
  - [ ] TestExtractJson
  - [ ] TestLLMDreamExecutor (think/execute/reflect with MockProvider)
  - [ ] TestCreateExecutor
  - [ ] TestProviderInitialization
- [ ] All tests pass: `pytest tests/test_llm_dream.py -v`
- [ ] Integrate into `tetra_dream.py` (add `llm_executor` param)
- [ ] Integrate into `start_api_v2.py` (TETRAMEM_LLM_PROVIDER env var)

### Week 3
- [ ] Write `demo_llm_dream_quality.py`
- [ ] Run quality benchmark: default vs LLM
- [ ] Target: avg quality >= 0.75 with LLM
- [ ] Update README.md with LLM usage instructions
- [ ] Create PR, merge to main

**Phase 1 deliverables**:
- [ ] `llm_integration.py` complete
- [ ] `test_llm_dream.py` all passing
- [ ] Quality benchmark report
- [ ] Demo script working

---

## Phase 2: Semantic Geometry (Week 4-6)

### Week 4
- [ ] Create branch `feat/phase2-semantic-geometry`
- [ ] Write `tetrahedron_memory/semantic_geometry.py` (~150 lines)
  - [ ] SemanticGeometryMapper class
  - [ ] sentence-transformers → embedding → PCA → 3D
  - [ ] Fallback to SHA-256 hash when model unavailable

### Week 5
- [ ] Write `tests/test_semantic_similarity.py` (~200 lines)
  - [ ] Similarity preservation test (>= 85%)
  - [ ] Cluster formation test
  - [ ] Fallback test
- [ ] Make `text_to_geometry()` pluggable in `tetra_mesh.py`

### Week 6
- [ ] Write `demo_semantic_vs_hash.py`
- [ ] Benchmark: query precision, topology navigation
- [ ] Create PR, merge to main

---

## Phase 3: Persistent Storage (Week 7-9)

### Week 7
- [ ] Create branch `feat/phase3-persistence`
- [ ] Design SQLite schema
- [ ] Write `tetrahedron_memory/persistence_v2.py` (~200 lines)

### Week 8
- [ ] Write `tests/test_persistence.py` (~150 lines)
  - [ ] save/load cycle test
  - [ ] Restart recovery test
  - [ ] 50K tetrahedra stress test
- [ ] Replace JSON replay in `start_api_v2.py`

### Week 9
- [ ] Write `demo_persistence_recovery.py`
- [ ] Create PR, merge to main

---

## Phase 4: Ancestry Tracking (Week 10-11)

### Week 10
- [ ] Create branch `feat/phase4-ancestry`
- [ ] Write `tetrahedron_memory/ancestry.py` (~100 lines)
  - [ ] AncestryTracker class
  - [ ] record_creation(), record_merge(), get_lineage()

### Week 11
- [ ] Modify `edge_contraction()` to record ancestry
- [ ] Integrate with EternityAudit
- [ ] Write `tests/test_ancestry.py` (~100 lines)
- [ ] Create PR, merge to main

---

## Phase 5: Visualization (Week 12)

### Week 12
- [ ] Create branch `feat/phase5-visualization`
- [ ] Write `tetrahedron_memory/visualization.py` (~150 lines)
  - [ ] OBJ mesh export
  - [ ] Entropy trajectory plot
  - [ ] Dream quality distribution
- [ ] Add monitoring endpoints to `start_api_v2.py`
- [ ] Write `tests/test_visualization.py` (~100 lines)
- [ ] Create PR, merge to main

---

## Final Deliverables

### Code
- [ ] 5 new modules (~850 lines)
- [ ] 5 new test files (~700 lines)
- [ ] Zero breaking changes to existing API

### Documentation
- [ ] V3.0_ROADMAP.md
- [ ] PHASE1_LLM_INTEGRATION.md
- [ ] SETUP_AND_GUIDELINES.md
- [ ] Updated README.md

### Demos
- [ ] 5 working demo scripts
- [ ] Performance comparison reports

## Success Criteria

| Metric | Target |
|--------|--------|
| Dream fusion quality (LLM) | >= 0.75 |
| Semantic similarity preservation | >= 85% |
| Data recovery success | 100% |
| Ancestry chain completeness | 100% |
| Test coverage (new code) | >= 85% |
