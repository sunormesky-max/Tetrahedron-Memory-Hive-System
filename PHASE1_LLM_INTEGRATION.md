# Phase 1: LLM Dream Synthesis — Detailed Design

## Goal

Replace DreamProtocol's default template-based synthesis with LLM-driven content generation.

## Current State

`DreamProtocol` in `tetra_dream.py:266-396` already supports custom callables:

```python
protocol = DreamProtocol(
    think_fn=callable,    # (inputs) -> dict
    execute_fn=callable,  # (inputs, analysis) -> str
    reflect_fn=callable,  # (inputs, content) -> float
    quality_threshold=0.3,
)
```

The integration point in `TetraDreamCycle.__init__` (line 448-455) already constructs `DreamProtocol` based on available callbacks. We need to add an `LLMDreamExecutor` class that provides these three callables.

## Architecture

```
DreamSynthesisInput (dict)
  ├── content: str
  ├── labels: List[str]
  ├── weight: float
  ├── centroid: List[float]
  ├── integration_count: int
  └── tetra_id: str

LLMDreamExecutor
  ├── think(inputs) -> Dict
  │     dominant_concept, synthesis_strategy, confidence, reasoning
  ├── execute(inputs, analysis) -> Optional[str]
  │     150-300 word synthesized content
  └── reflect(inputs, content) -> float
        quality score 0.0-1.0

Providers
  ├── AnthropicExecutor (Claude 3.5 Sonnet)
  ├── OpenAIExecutor (GPT-4)
  ├── GLMExecutor (via OpenClaw HTTP API)
  └── OllamaExecutor (local, offline)
```

## Integration Plan

### Step 1: Create `llm_integration.py`

Base class `LLMDreamExecutor` with provider abstraction. Each provider implements `_call_llm(prompt) -> str`.

### Step 2: Add to `TetraDreamCycle.__init__`

```python
# In tetra_dream.py, add parameter:
def __init__(self, ..., llm_executor=None):
    if llm_executor is not None:
        self._protocol = DreamProtocol(
            think_fn=llm_executor.think,
            execute_fn=llm_executor.execute,
            reflect_fn=llm_executor.reflect,
            quality_threshold=0.4,  # LLM is more reliable
        )
    elif synthesis_fn is not None:
        ...  # existing logic
```

### Step 3: Add to `start_api_v2.py`

```python
# New env var: TETRAMEM_LLM_PROVIDER
llm = os.environ.get("TETRAMEM_LLM_PROVIDER", "")
if llm:
    from tetrahedron_memory.llm_integration import create_executor
    executor = create_executor(llm)
    _dream = TetraDreamCycle(_mesh, llm_executor=executor)
```

## Provider Configuration

### Anthropic (Claude)
```bash
export TETRAMEM_LLM_PROVIDER=anthropic
export ANTHROPIC_API_KEY=sk-ant-...
```

### OpenAI (GPT)
```bash
export TETRAMEM_LLM_PROVIDER=openai
export OPENAI_API_KEY=sk-...
```

### GLM-5 (via OpenClaw)
```bash
export TETRAMEM_LLM_PROVIDER=glm
export TETRAMEM_LLM_ENDPOINT=http://127.0.0.1:8080/v1/chat/completions
```

### Ollama (local)
```bash
export TETRAMEM_LLM_PROVIDER=ollama
export TETRAMEM_LLM_MODEL=llama3
```

## Cost Estimate

| Provider | Input / 1K tokens | Output / 1K tokens | Cost per dream |
|----------|-------------------|--------------------|---------|
| Claude 3.5 Sonnet | $0.003 | $0.015 | ~$0.003 |
| GPT-4o | $0.005 | $0.015 | ~$0.004 |
| GLM-5 | Free (via OpenClaw) | Free | $0.00 |
| Ollama | Free (local) | Free | $0.00 |

Each dream cycle calls LLM 3 times (THINK + EXECUTE + REFLECT), consuming ~500-1000 tokens per call.

## Test Plan

1. `test_executor_init` — each provider initializes correctly
2. `test_think_returns_valid_analysis` — structure validation
3. `test_execute_returns_content` — non-empty, >50 chars
4. `test_reflect_returns_score` — 0.0-1.0 range
5. `test_full_dream_cycle_with_llm` — end-to-end with TetraDreamCycle
6. `test_fallback_on_llm_failure` — graceful degradation to default
7. `test_quality_improvement` — avg quality >= 0.75 over 10 cycles

## Demo Script

`demo_llm_dream_quality.py`:
1. Store 20 AI-related memories
2. Run 10 dream cycles with default synthesizer → record quality
3. Run 10 dream cycles with LLM synthesizer → record quality
4. Print comparison table
