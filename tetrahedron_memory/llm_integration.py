"""
LLM-driven dream synthesis for TetraMem-XL v3.0.

Provides LLMDreamExecutor that implements the three DreamProtocol callables:
  think_fn(inputs) -> dict
  execute_fn(inputs, analysis) -> Optional[str]
  reflect_fn(inputs, content) -> float

Supported providers: anthropic, openai, glm (via HTTP), ollama (local)
"""

import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

logger = logging.getLogger("tetramem.llm")


class LLMProvider(ABC):
    @abstractmethod
    def call(self, system: str, user: str, temperature: float = 0.7, max_tokens: int = 500) -> str:
        ...


class AnthropicProvider(LLMProvider):
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-sonnet-4-20250514"):
        try:
            import anthropic
        except ImportError:
            raise ImportError("pip install anthropic")
        self._client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self._model = model

    def call(self, system: str, user: str, temperature: float = 0.7, max_tokens: int = 500) -> str:
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return resp.content[0].text


class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("pip install openai")
        self._client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self._model = model

    def call(self, system: str, user: str, temperature: float = 0.7, max_tokens: int = 500) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content


class GLMProvider(LLMProvider):
    def __init__(self, endpoint: Optional[str] = None, model: str = "glm-5"):
        import urllib.request
        self._endpoint = endpoint or os.environ.get("TETRAMEM_LLM_ENDPOINT", "http://127.0.0.1:8080/v1/chat/completions")
        self._model = model

    def call(self, system: str, user: str, temperature: float = 0.7, max_tokens: int = 500) -> str:
        import urllib.request
        payload = json.dumps({
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }).encode()
        req = urllib.request.Request(
            self._endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        resp = urllib.request.urlopen(req, timeout=30)
        data = json.loads(resp.read())
        return data["choices"][0]["message"]["content"]


class OllamaProvider(LLMProvider):
    def __init__(self, model: str = "llama3", host: str = "http://localhost:11434"):
        import urllib.request
        self._model = model
        self._host = host

    def call(self, system: str, user: str, temperature: float = 0.7, max_tokens: int = 500) -> str:
        import urllib.request
        payload = json.dumps({
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "stream": False,
        }).encode()
        req = urllib.request.Request(
            f"{self._host}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        resp = urllib.request.urlopen(req, timeout=120)
        data = json.loads(resp.read())
        return data["message"]["content"]


SYSTEM_THINK = """You are analyzing a group of AI memory fragments to identify synthesis opportunities.
Return JSON only, no markdown fences."""

SYSTEM_EXECUTE = """You are a creative knowledge synthesis engine. Generate new insights by combining multiple memories.
Output the synthesized content directly, no formatting wrappers."""

SYSTEM_REFLECT = """You are evaluating the quality of a synthesized memory. Score 0-10.
Return JSON only: {"score": N, "reasoning": "brief explanation"}"""


def _extract_json(text: str) -> Optional[Dict]:
    for marker in ["```json", "```"]:
        if marker in text:
            try:
                chunk = text.split(marker)[1]
                if marker == "```json":
                    chunk = chunk.split("```")[0]
                else:
                    parts = chunk.split("```")
                    chunk = parts[0] if parts else chunk
                return json.loads(chunk.strip())
            except (json.JSONDecodeError, IndexError):
                continue
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        return None


class LLMDreamExecutor:
    def __init__(self, provider: LLMProvider):
        self._provider = provider
        self._call_count = 0

    def think(self, inputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        sources = "\n".join(f"- [{i.get('labels', [])}] {i['content'][:120]}" for i in inputs[:6])
        labels = sorted(set(l for i in inputs for l in i.get("labels", []) if not l.startswith("__")))

        user = f"""## Source memories
{sources}

## Labels: {', '.join(labels) or 'none'}

## Task
Identify the dominant concept and best synthesis strategy.
Return JSON:
{{
  "dominant_concept": "string (max 20 chars)",
  "synthesis_strategy": "generalize|specialize|analogize",
  "bridging_paths": ["path1", "path2"],
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation"
}}"""

        try:
            raw = self._provider.call(SYSTEM_THINK, user, temperature=0.5, max_tokens=300)
            self._call_count += 1
            parsed = _extract_json(raw)
            if parsed and "dominant_concept" in parsed:
                return parsed
        except Exception as e:
            logger.warning("LLM think failed: %s", e)

        return {
            "dominant_concept": "unknown",
            "synthesis_strategy": "generalize",
            "bridging_paths": [],
            "confidence": 0.3,
            "reasoning": "LLM call failed, using fallback",
        }

    def execute(self, inputs: List[Dict[str, Any]], analysis: Optional[Dict[str, Any]] = None) -> Optional[str]:
        analysis = analysis or {"synthesis_strategy": "generalize", "dominant_concept": "new concept"}
        strategy = analysis.get("synthesis_strategy", "generalize")
        concept = analysis.get("dominant_concept", "new concept")

        sources = "\n".join(f"- {i['content'][:150]}" for i in inputs[:5])

        if strategy == "generalize":
            task = f'Extract a high-level principle from these memories under the concept "{concept}". Discover a non-obvious insight. 150-200 words.'
        elif strategy == "specialize":
            task = f'Under "{concept}", find a specific new application that combines these memories\' intersection. 150-200 words.'
        else:
            task = f'Find a metaphor or parallel structure in these memories under "{concept}". Reveal new understanding. 150-200 words.'

        user = f"""## Source memories
{sources}

## Synthesis task
{task}

Output the synthesized content directly."""

        try:
            raw = self._provider.call(SYSTEM_EXECUTE, user, temperature=0.7, max_tokens=500)
            self._call_count += 1
            content = raw.strip()
            if len(content) > 20:
                return content
        except Exception as e:
            logger.warning("LLM execute failed: %s", e)

        return None

    def reflect(self, inputs: List[Dict[str, Any]], content: str) -> float:
        preview = " | ".join(i["content"][:50] for i in inputs[:3])

        user = f"""## Source memory preview
{preview}

## Synthesized content
{content[:500]}

## Evaluation dimensions
1. Novelty: new concepts not in sources?
2. Coherence: logically connects sources?
3. Depth: hidden associations discovered?
4. Utility: guides future thinking?

Return JSON: {{"score": 0-10, "reasoning": "brief"}}"""

        try:
            raw = self._provider.call(SYSTEM_REFLECT, user, temperature=0.2, max_tokens=200)
            self._call_count += 1
            parsed = _extract_json(raw)
            if parsed and "score" in parsed:
                score = float(parsed["score"]) / 10.0
                return max(0.0, min(1.0, score))
        except Exception as e:
            logger.warning("LLM reflect failed: %s", e)

        return 0.5

    @property
    def call_count(self) -> int:
        return self._call_count


def create_executor(provider_name: Optional[str] = None, **kwargs) -> Optional[LLMDreamExecutor]:
    provider_name = provider_name or os.environ.get("TETRAMEM_LLM_PROVIDER", "")

    if not provider_name:
        return None

    providers = {
        "anthropic": AnthropicProvider,
        "openai": OpenAIProvider,
        "glm": GLMProvider,
        "ollama": OllamaProvider,
    }

    cls = providers.get(provider_name.lower())
    if cls is None:
        logger.warning("Unknown LLM provider: %s", provider_name)
        return None

    try:
        provider = cls(**kwargs)
        return LLMDreamExecutor(provider)
    except Exception as e:
        logger.warning("Failed to create LLM provider %s: %s", provider_name, e)
        return None
