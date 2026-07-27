"""Router engine — embedding-based semantic router for home-llm."""
from __future__ import annotations

import logging
import time
from typing import Any

from homeassistant.core import HomeAssistant

from .router_config import BackendInfo, RouteDefinition, RouterConfigData, RoutingResult

_LOGGER = logging.getLogger(__name__)


class EmbeddingClient:
    """OpenAI-compatible embedding API client with response caching."""

    def __init__(self, base_url: str, api_key: str, model: str, dimensions: int) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.dimensions = dimensions
        self._cache: dict[str, list[float]] = {}

    async def embed(self, texts: list[str]) -> list[list[float]]:
        uncached = [(i, t) for i, t in enumerate(texts) if t not in self._cache]
        if uncached:
            vectors = await self._call_api([t for _, t in uncached])
            for (i, text), vec in zip(uncached, vectors):
                self._cache[text] = vec
        return [self._cache[t] for t in texts]

    async def _call_api(self, texts: list[str]) -> list[list[float]]:
        import httpx
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{self.base_url}/v1/embeddings",
                headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {},
                json={
                    "model": self.model,
                    "input": texts,
                    **({"dimensions": self.dimensions} if self.dimensions else {}),
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return [item["embedding"] for item in data["data"]]

    def clear_cache(self) -> None:
        self._cache.clear()


class TaskRouter:
    """Embedding-based semantic router.

    Compares user queries against prototype utterances for each route category
    using cosine similarity, then returns the best-matching backend.
    """

    def __init__(
        self,
        hass: HomeAssistant,
        embedding_client: EmbeddingClient,
        config_store: Any,
    ) -> None:
        self.hass = hass
        self.embedding_client = embedding_client
        self.config_store = config_store
        self._route_prototypes: dict[str, list[list[float]]] = {}
        self._initialized = False

    @property
    def initialized(self) -> bool:
        return self._initialized

    async def initialize(self) -> None:
        config = self.config_store.data
        routes = config.routes
        if not routes:
            self._initialized = True
            return
        all_utterances = []
        route_utterance_count: list[int] = []
        for route in routes:
            all_utterances.extend(route.utterances)
            route_utterance_count.append(len(route.utterances))
        if not all_utterances:
            self._initialized = True
            return
        vectors = await self.embedding_client.embed(all_utterances)
        idx = 0
        for i, route in enumerate(routes):
            count = route_utterance_count[i]
            self._route_prototypes[route.name] = vectors[idx: idx + count]
            idx += count
        self._initialized = True
        _LOGGER.debug("TaskRouter initialized with %d routes, %d utterances", len(routes), len(all_utterances))

    async def route(self, query: str, backends: list[BackendInfo]) -> RoutingResult | None:
        config = self.config_store.data
        routes = config.routes
        if not routes:
            return None
        try:
            query_vec = (await self.embedding_client.embed([query]))[0]
        except Exception:
            _LOGGER.warning("Embedding API call failed, using fallback")
            return None
        best_score = 0.0
        best_route: RouteDefinition | None = None
        for route in routes:
            prototypes = self._route_prototypes.get(route.name, [])
            for proto_vec in prototypes:
                score = _cosine_similarity(query_vec, proto_vec)
                if score > best_score:
                    best_score = score
                    best_route = route
        if best_route is None or best_score < best_route.threshold:
            fallback_id = config.fallback
            if fallback_id:
                backend = next((b for b in backends if b.entity_id == fallback_id), None)
                if backend:
                    return RoutingResult(backend=backend, source="fallback", confidence=best_score)
            return None
        backend = next((b for b in backends if b.entity_id == best_route.target), None)
        if not backend:
            fallback_id = config.fallback
            fb = next((b for b in backends if b.entity_id == fallback_id), backends[0] if backends else None)
            if fb:
                return RoutingResult(backend=fb, source="fallback", confidence=best_score)
            return None
        return RoutingResult(
            backend=backend,
            category=best_route.name,
            confidence=best_score,
            source="semantic",
        )


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb + 1e-10)
