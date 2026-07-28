"""Data models for the Router Agent configuration."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EmbeddingConfig:
    """Embedding service configuration (OpenAI-compatible API)."""

    base_url: str
    api_key: str
    model: str
    dimensions: int = 1024

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EmbeddingConfig | None:
        if not data or not data.get("base_url") or not data.get("model"):
            return None
        return cls(
            base_url=data["base_url"],
            api_key=data.get("api_key", ""),
            model=data["model"],
            dimensions=int(data.get("dimensions", 1024)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_url": self.base_url,
            "api_key": self.api_key,
            "model": self.model,
            "dimensions": self.dimensions,
        }


@dataclass
class RouteDefinition:
    """A single routing category with utterance prototypes."""

    name: str
    label: str
    target: str  # entity_id of the target backend
    threshold: float = 0.70
    utterances: list[str] = field(default_factory=list)
    priority: int = 100  # lower = higher priority, used as tiebreaker when scores are close

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RouteDefinition:
        return cls(
            name=data["name"],
            label=data.get("label", data["name"]),
            target=data["target"],
            threshold=float(data.get("threshold", 0.70)),
            utterances=list(data.get("utterances", [])),
            priority=int(data.get("priority", 100)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "target": self.target,
            "threshold": self.threshold,
            "utterances": list(self.utterances),
            "priority": self.priority,
        }


@dataclass
class RouterConfigData:
    """Complete router configuration."""

    embedding: EmbeddingConfig | None = None
    routes: list[RouteDefinition] = field(default_factory=list)
    fallback: str | None = None  # fallback backend entity_id

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RouterConfigData:
        raw_embedding = data.get("embedding")
        embedding = EmbeddingConfig.from_dict(raw_embedding) if raw_embedding else None
        raw_routes = data.get("routes", [])
        return cls(
            embedding=embedding,
            routes=[RouteDefinition.from_dict(r) for r in raw_routes],
            fallback=data.get("fallback"),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "embedding": self.embedding.to_dict() if self.embedding else None,
            "routes": [r.to_dict() for r in self.routes],
        }
        if self.fallback:
            result["fallback"] = self.fallback
        return result


@dataclass
class BackendInfo:
    """Runtime info about a discovered backend agent."""

    entity_id: str
    name: str
    model_name: str = ""


@dataclass
class RoutingResult:
    """Result of a routing decision."""

    backend: BackendInfo
    category: str | None = None
    confidence: float = 0.0
    source: str = "semantic"  # "semantic" | "fallback" | "cache"
