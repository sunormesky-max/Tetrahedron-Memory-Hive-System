# TetraMem-XL v3.0 — Development Setup & Guidelines

## Environment

### Prerequisites
- Python 3.10+
- pip

### Install

```bash
git clone https://github.com/sunormesky-max/TetraMem-XL-v3.git
cd TetraMem-XL-v3
pip install -e ".[all]"
```

### Phase 1 extras
```bash
pip install anthropic    # Claude
# or
pip install openai       # GPT
```

### Phase 2 extras
```bash
pip install sentence-transformers scikit-learn
```

### Phase 3 extras
```bash
pip install pyarrow pandas
```

## API Keys

### Environment variables
```bash
# LLM provider (pick one)
export TETRAMEM_LLM_PROVIDER=anthropic   # or openai, glm, ollama
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
export TETRAMEM_LLM_ENDPOINT=http://127.0.0.1:8080/v1/chat/completions  # for GLM
```

### .env file (project root)
```bash
TETRAMEM_LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
```

## Workflow

### Branch naming
```bash
feat/phase1-llm-integration
feat/phase2-semantic-geometry
feat/phase3-persistence
feat/phase4-ancestry
feat/phase5-visualization
```

### Commit format
```
type(scope): description

Types: feat, fix, refactor, perf, test, docs, chore
Scopes: phase1, phase2, phase3, phase4, phase5, core, api
```

### Tests
```bash
pytest tests/ -v                              # all
pytest tests/test_llm_dream.py -v             # specific
pytest tests/ -v -k "not throughput_25k"       # skip perf thresholds
```

### Code style
```bash
black tetrahedron_memory/ tests/
flake8 tetrahedron_memory/ --max-line-length=120
```

## LLM Cost Control

| Provider | Per dream (3 calls) | Monthly budget |
|----------|--------------------|----|
| Claude 3.5 Sonnet | ~$0.003 | Set alert at $10 |
| GPT-4o | ~$0.004 | Set alert at $10 |
| GLM-5 (OpenClaw) | $0 | Unlimited |
| Ollama (local) | $0 | Unlimited |

## Offline Mode

Set `TETRAMEM_LLM_PROVIDER=""` or omit it. System falls back to default template synthesizer.
