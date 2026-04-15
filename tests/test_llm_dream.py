"""
Tests for LLM-driven dream synthesis (TetraMem-XL v3.0 Phase 1).

Run: pytest tests/test_llm_dream.py -v
"""

import pytest
from unittest.mock import MagicMock, patch
from tetrahedron_memory.llm_integration import (
    LLMDreamExecutor,
    AnthropicProvider,
    OpenAIProvider,
    GLMProvider,
    OllamaProvider,
    create_executor,
    _extract_json,
)


class TestExtractJson:
    def test_plain_json(self):
        assert _extract_json('{"score": 5}') == {"score": 5}

    def test_json_in_code_block(self):
        text = '```json\n{"score": 7}\n```'
        assert _extract_json(text) == {"score": 7}

    def test_json_in_plain_block(self):
        text = '```\n{"score": 3}\n```'
        assert _extract_json(text) == {"score": 3}

    def test_invalid_returns_none(self):
        assert _extract_json("not json") is None


class MockProvider:
    def __init__(self, responses=None):
        self._responses = responses or []
        self._idx = 0

    def call(self, system, user, temperature=0.7, max_tokens=500):
        if self._idx < len(self._responses):
            resp = self._responses[self._idx]
            self._idx += 1
            return resp
        return '{"score": 5}'


class TestLLMDreamExecutor:
    @pytest.fixture
    def sample_inputs(self):
        return [
            {"content": "Deep learning uses multi-layer neural networks for pattern recognition", "labels": ["ai", "dl"], "weight": 1.0, "centroid": [0.5, 0.0, 0.0], "integration_count": 2, "tetra_id": "abc123"},
            {"content": "Neural networks are inspired by biological brains", "labels": ["ai", "biology"], "weight": 0.8, "centroid": [0.3, 0.1, 0.0], "integration_count": 1, "tetra_id": "def456"},
            {"content": "Reinforcement learning trains agents through reward signals", "labels": ["ai", "rl"], "weight": 0.9, "centroid": [0.6, -0.1, 0.0], "integration_count": 3, "tetra_id": "ghi789"},
        ]

    def test_think_returns_valid_structure(self, sample_inputs):
        mock_json = '{"dominant_concept": "learning architectures", "synthesis_strategy": "generalize", "bridging_paths": ["biological-artificial"], "confidence": 0.8, "reasoning": "All relate to learning systems"}'
        executor = LLMDreamExecutor(MockProvider([mock_json]))
        result = executor.think(sample_inputs)

        assert "dominant_concept" in result
        assert result["synthesis_strategy"] in ("generalize", "specialize", "analogize")
        assert 0.0 <= result["confidence"] <= 1.0
        assert executor.call_count == 1

    def test_think_fallback_on_failure(self, sample_inputs):
        executor = LLMDreamExecutor(MockProvider(["not valid json"]))
        result = executor.think(sample_inputs)

        assert result["dominant_concept"] == "unknown"
        assert result["confidence"] == 0.3

    def test_execute_returns_content(self, sample_inputs):
        content = "Learning architectures represent a unified framework spanning biological neural systems and artificial computation. The convergence of supervised, unsupervised, and reinforcement paradigms suggests a deeper principle: intelligence emerges from layered information processing with feedback loops."
        executor = LLMDreamExecutor(MockProvider([content]))
        analysis = {"synthesis_strategy": "generalize", "dominant_concept": "learning"}
        result = executor.execute(sample_inputs, analysis)

        assert result is not None
        assert len(result) > 50

    def test_execute_returns_none_on_short_output(self, sample_inputs):
        executor = LLMDreamExecutor(MockProvider(["too short"]))
        analysis = {"synthesis_strategy": "generalize", "dominant_concept": "test"}
        result = executor.execute(sample_inputs, analysis)

        assert result is None

    def test_execute_fallback_on_exception(self, sample_inputs):
        class FailProvider:
            def call(self, *a, **kw):
                raise ConnectionError("API down")
        executor = LLMDreamExecutor(FailProvider())
        result = executor.execute(sample_inputs)

        assert result is None

    def test_reflect_returns_score(self, sample_inputs):
        executor = LLMDreamExecutor(MockProvider(['{"score": 7, "reasoning": "Good synthesis"}']))
        score = executor.reflect(sample_inputs, "Some synthesized content about learning architectures")

        assert 0.0 <= score <= 1.0
        assert abs(score - 0.7) < 0.01

    def test_reflect_clamps_score(self, sample_inputs):
        executor = LLMDreamExecutor(MockProvider(['{"score": 15, "reasoning": "Too high"}']))
        score = executor.reflect(sample_inputs, "content")
        assert score == 1.0

    def test_reflect_fallback_on_parse_failure(self, sample_inputs):
        executor = LLMDreamExecutor(MockProvider(["not json"]))
        score = executor.reflect(sample_inputs, "content")
        assert score == 0.5


class TestCreateExecutor:
    def test_returns_none_for_empty_provider(self):
        assert create_executor("") is None
        assert create_executor(None) is None

    def test_returns_none_for_unknown_provider(self):
        assert create_executor("nonexistent") is None

    def test_creates_anthropic_executor_with_mock(self):
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("tetrahedron_memory.llm_integration.AnthropicProvider") as mock_cls:
                mock_cls.return_value = MagicMock()
                executor = create_executor("anthropic")
                assert executor is not None
                assert isinstance(executor, LLMDreamExecutor)


class TestProviderInitialization:
    def test_glm_provider_configurable(self):
        provider = GLMProvider(endpoint="http://localhost:9999/v1/chat/completions")
        assert provider._endpoint == "http://localhost:9999/v1/chat/completions"

    def test_ollama_provider_configurable(self):
        provider = OllamaProvider(model="mistral", host="http://192.168.1.100:11434")
        assert provider._model == "mistral"
