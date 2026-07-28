"""Router Agent — semantic routing layer for home-llm."""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from homeassistant.components import conversation
from homeassistant.components.conversation import (
    ConversationEntity,
    ConversationInput,
    ConversationResult,
)
from homeassistant.components.conversation.const import DATA_COMPONENT
from homeassistant.const import MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_component, intent
from homeassistant.helpers.storage import Store
from homeassistant.helpers.selector import SelectOptionDict

from .const import DOMAIN, CONF_CHAT_MODEL
from .conversation import LocalLLMAgent
from .router_config import BackendInfo, RouterConfigData, RoutingResult
from .router_engine import TaskRouter

_LOGGER = logging.getLogger(__name__)

STORAGE_KEY = f"{DOMAIN}.router_config"
STORAGE_VERSION = 1
STORAGE_VERSION_MINOR = 1


# ──────────────────────────────────────────────
#  Config Store
# ──────────────────────────────────────────────

class RouterConfigStore:
    """Persistent storage for Router Agent configuration using HA Store."""

    def __init__(self, hass: HomeAssistant) -> None:
        self.hass = hass
        self._store = Store[dict[str, Any]](
            hass, STORAGE_VERSION, STORAGE_KEY, minor_version=STORAGE_VERSION_MINOR
        )
        self._data: RouterConfigData = RouterConfigData()
        self._loaded = False

    async def async_load(self) -> RouterConfigData:
        raw = await self._store.async_load()
        if raw:
            self._data = RouterConfigData.from_dict(raw)
        else:
            self._data = RouterConfigData()
        self._loaded = True
        _LOGGER.debug("Router config loaded: %d routes", len(self._data.routes))
        return self._data

    async def async_save(self, data: RouterConfigData) -> None:
        self._data = data
        await self._store.async_save(data.to_dict())
        _LOGGER.debug("Router config saved: %d routes", len(data.routes))

    @property
    def data(self) -> RouterConfigData:
        if not self._loaded:
            raise RuntimeError("RouterConfigStore not loaded. Call async_load() first.")
        return self._data


# ──────────────────────────────────────────────
#  Router Conversation Agent
# ──────────────────────────────────────────────

class RouterConversationAgent(ConversationEntity):
    """Conversation agent that routes requests to the appropriate LLM backend
    based on embedding-based semantic routing."""

    _attr_name = "Home LLM Router"
    _attr_unique_id = "home_llm_router"
    _attr_has_entity_name = True
    _attr_supported_features = conversation.ConversationEntityFeature.CONTROL

    def __init__(self, hass: HomeAssistant) -> None:
        self.hass = hass
        self.config_store = RouterConfigStore(hass)
        self.task_router: TaskRouter | None = None
        self._session_backend_cache: dict[str, str] = {}
        self._init_lock = asyncio.Lock()
        self._initialized = False

    async def async_initialize(self) -> None:
        async with self._init_lock:
            if self._initialized:
                return
            config = await self.config_store.async_load()
            emb = config.embedding
            if emb and emb.base_url and emb.model:
                from .router_engine import EmbeddingClient

                embedding_client = EmbeddingClient(
                    base_url=emb.base_url,
                    api_key=emb.api_key,
                    model=emb.model,
                    dimensions=emb.dimensions,
                )
                self.task_router = TaskRouter(
                    hass=self.hass,
                    embedding_client=embedding_client,
                    config_store=self.config_store,
                )
                try:
                    await self.task_router.initialize()
                except Exception:
                    _LOGGER.exception("TaskRouter initialization failed")
            else:
                self.task_router = None
            self._initialized = True

    @property
    def supported_languages(self) -> list[str] | str:
        return MATCH_ALL

    def _discover_backends(self) -> list[BackendInfo]:
        component: entity_component.EntityComponent = self.hass.data.get(DATA_COMPONENT)
        if not component:
            return []
        backends: list[BackendInfo] = []
        for entity in list(component.entities):
            if isinstance(entity, LocalLLMAgent) and entity.entity_id != self.entity_id:
                model_name = ""
                try:
                    model_name = entity.runtime_options.get(CONF_CHAT_MODEL, "")
                except Exception:
                    pass
                backends.append(BackendInfo(
                    entity_id=entity.entity_id,
                    name=entity.name or entity.entity_id,
                    model_name=model_name,
                ))
        return backends

    def _find_agent(self, entity_id: str) -> LocalLLMAgent | None:
        component: entity_component.EntityComponent = self.hass.data.get(DATA_COMPONENT)
        if not component:
            return None
        entity = component.get_entity(entity_id)
        if isinstance(entity, LocalLLMAgent):
            return entity
        return None

    async def async_process(self, user_input: ConversationInput) -> ConversationResult:
        await self.async_initialize()

        conv_id = user_input.conversation_id
        if conv_id and conv_id in self._session_backend_cache:
            cached_id = self._session_backend_cache[conv_id]
            _LOGGER.debug("Router cache hit for conv=%s → backend=%s", conv_id, cached_id)
            target = self._find_agent(cached_id)
            if target:
                return await target.async_process(user_input)
            del self._session_backend_cache[conv_id]

        backends = self._discover_backends()
        if not backends:
            return _error_result(
                self.hass, "No LLM backends available. Please add at least one model.",
                user_input,
            )

        route_result: RoutingResult | None = None

        if self.task_router and self.task_router.initialized:
            try:
                route_result = await self.task_router.route(user_input.text, backends)
            except Exception:
                _LOGGER.exception("Routing failed, falling back")

        if not route_result:
            fallback_id = self.config_store.data.fallback
            fallback_backend = next(
                (b for b in backends if b.entity_id == fallback_id), None
            )
            if not fallback_backend and backends:
                fallback_backend = backends[0]
            if fallback_backend:
                route_result = RoutingResult(
                    backend=fallback_backend,
                    source="fallback",
                )

        if not route_result:
            return _error_result(
                self.hass, "No suitable backend found for your request.",
                user_input,
            )

        if conv_id:
            self._session_backend_cache[conv_id] = route_result.backend.entity_id

        _LOGGER.info(
            "Router: query=%.60s category=%s confidence=%.2f source=%s backend=%s",
            user_input.text,
            route_result.category or "-",
            route_result.confidence,
            route_result.source,
            route_result.backend.entity_id,
        )

        target = self._find_agent(route_result.backend.entity_id)
        if not target:
            return _error_result(
                self.hass, f"Backend {route_result.backend.name} is no longer available.",
                user_input,
            )

        return await target.async_process(user_input)

    def get_available_backends(self) -> list[SelectOptionDict]:
        return [
            SelectOptionDict(value=b.entity_id, label=f"{b.name} ({b.model_name})" if b.model_name else b.name)
            for b in self._discover_backends()
        ]

    def invalidate_session_cache(self, conversation_id: str) -> None:
        self._session_backend_cache.pop(conversation_id, None)

    async def async_will_remove_from_hass(self) -> None:
        self._session_backend_cache.clear()
        await super().async_will_remove_from_hass()


# ──────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────

def _error_result(hass: HomeAssistant, message: str, user_input: ConversationInput) -> ConversationResult:
    intent_response = intent.IntentResponse(language=user_input.language)
    intent_response.async_set_error(
        intent.IntentResponseErrorCode.FAILED_TO_HANDLE,
        message,
    )
    return ConversationResult(
        response=intent_response,
        conversation_id=user_input.conversation_id,
    )
