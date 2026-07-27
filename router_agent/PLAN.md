# Router Agent 开发计划

> 基于 home-llm 插件扩展，版本 v0.5.0 | 2026-07

---

## 1. 架构概览

```
homeassistant/components/conversation/
  EntityComponent[ConversationEntity]  (hass.data[DATA_COMPONENT])
    ├── LocalLLMAgent A (entity_id: "conversation.llama_fast")
    │     └── client: LocalLLMClient (绑定的后端)
    ├── LocalLLMAgent B (entity_id: "conversation.llama_capable")
    │     └── client: LocalLLMClient (绑定的后端)
    └── RouterConversationAgent (entity_id: "conversation.home_llm_router")  ← [新增]
          ├── hass.data[DOMAIN] → 扫描所有 ConfigEntry
          ├── hass.data[DATA_COMPONENT].entities → 发现所有 LocalLLMAgent
          ├── TaskRouter (路由引擎)
          │     ├── EmbeddingClient (纯语义路由)
          │     └── RouteTable (路由规则 + 原型向量)
          ├── RouterConfigStore (HA Store 持久化)
          └── _session_backend_cache (对话级缓存)
```

### 核心流程

```
Voice Pipeline → RouterConversationAgent.async_process(user_input)
  │
  ├─ 1. _discover_backends() → 扫描当前可用的 LocalLLMAgent 列表
  │
  ├─ 2. 检查 _session_backend_cache[conversation_id]
  │     └─ 命中 → 直接委托给缓存的 agent
  │
  ├─ 3. TaskRouter._try_semantic(query)
  │     ├─ EmbeddingClient.embed([query]) → query_vec
  │     ├─ 与所有 Route 的 prototype vectors 计算余弦相似度
  │     ├─ 取最高分 category
  │     ├─ score >= threshold → 使用该 Route
  │     └─ score < threshold → 使用 fallback
  │
  ├─ 4. _resolve_backend(category, backends)
  │     └─ 在可用后端的 entity_id 列表中查找 Route.target
  │         ├─ 找到 → 缓存到 _session_backend_cache[conversation_id]
  │         └─ 未找到（后端已删）→ 使用 fallback
  │
  └─ 5. target_agent.async_process(user_input)
        └─ 完全委托，ChatLog 隔离机制已验证安全
```

---

## 2. 新增文件

### 2.1 `router_agent.py` — Router Agent 主体（~500 行）

```python
class RouterConversationAgent(ConversationEntity):
    """路由代理实体 — 自动注册，纳管所有 LocalLLMAgent"""

    # 注册属性
    _attr_name = "Home LLM Router"
    _attr_unique_id = "home_llm_router"
    _attr_has_entity_name = True
    _attr_supported_features = ConversationEntityFeature.CONTROL
    _attr_supported_languages = MATCH_ALL
    _attr_supports_streaming = False  # 取决于被委托的后端

    hass: HomeAssistant
    task_router: TaskRouter
    config_store: RouterConfigStore
    _session_backend_cache: dict[str, str]  # conversation_id → target_entity_id

    def __init__(self, hass: HomeAssistant):
        self.hass = hass
        self.config_store = RouterConfigStore(hass)
        self.task_router = TaskRouter(hass, self.config_store)
        self._session_backend_cache = {}

    def _discover_backends(self) -> list[BackendInfo]:
        """发现所有已注册的 LocalLLMAgent"""
        component: EntityComponent = self.hass.data[DATA_COMPONENT]
        return [
            BackendInfo(
                entity_id=entity.entity_id,
                name=entity.name or entity.entity_id,
                model_name=entity.runtime_options.get(CONF_CHAT_MODEL, ""),
            )
            for entity in component.entities
            if isinstance(entity, LocalLLMAgent) and entity.entity_id != self.entity_id
        ]

    async def async_process(self, user_input: ConversationInput) -> ConversationResult:
        """路由决策 → 委托给目标后端"""
        # 同一个对话内复用首次路由决策
        if user_input.conversation_id in self._session_backend_cache:
            target_id = self._session_backend_cache[user_input.conversation_id]
            target = self._find_agent(target_id)
            if target:
                return await target.async_process(user_input)
            # 缓存的后端不可用，降级重新路由

        backends = self._discover_backends()
        if not backends:
            return _error_result("没有可用的 LLM 后端", user_input)

        route_result = await self.task_router.route(user_input.text, backends)
        if not route_result:
            return _error_result("路由失败，请检查配置", user_input)

        self._session_backend_cache[user_input.conversation_id] = route_result.backend.entity_id
        target_id = route_result.backend.entity_id
        target = self._find_agent(target_id)
        if not target:
            return _error_result("路由目标后端不可用", user_input)

        return await target.async_process(user_input)

    def get_available_backends(self) -> list[SelectOptionDict]:
        """供 Config Flow 调用的后端列表"""
        ...


class EmbeddingClient:
    """OpenAI 兼容的 Embedding API 客户端"""

    def __init__(self, base_url: str, api_key: str, model: str, dimensions: int):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.dimensions = dimensions
        self._cache: dict[str, list[float]] = {}  # 原型向量缓存

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """批量 Embedding，带缓存"""
        ...

    async def _call_api(self, texts: list[str]) -> list[list[float]]:
        """POST {base_url}/v1/embeddings"""
        ...


class TaskRouter:
    """纯语义路由引擎"""

    hass: HomeAssistant
    config_store: RouterConfigStore
    embedding_client: EmbeddingClient | None
    _route_prototypes: dict[str, list[list[float]]]
    _route_configs: dict[str, RouteDefinition]
    _initialized: bool

    async def initialize(self):
        """预热：计算所有 Route 的原型向量"""
        ...

    async def route(self, query: str, backends: list[BackendInfo]) -> RoutingResult | None:
        """
        纯语义路由：
        1. Embedding 相似度匹配 → 取最高分
        2. score >= threshold → 返回对应后端
        3. score < threshold → 返回 fallback
        4. Embedding API 异常 → 返回 fallback
        """
        ...

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        ...


class RouterConfigStore:
    """路由规则持久化（HA Store）"""

    _store: Store
    _data: RouterConfigData

    async def async_load(self):
        ...

    async def async_save(self, data: RouterConfigData):
        ...

    @property
    def embedding_config(self) -> EmbeddingConfig | None:
        ...

    @property
    def routes(self) -> list[RouteDefinition]:
        ...

    @property
    def fallback(self) -> str | None:
        ...
```

### 2.2 `router_config.py` — 数据模型（~100 行）

```python
@dataclass
class EmbeddingConfig:
    """Embedding 服务配置"""
    base_url: str       # e.g. "https://api.openai.com"
    api_key: str        # API Key
    model: str          # e.g. "text-embedding-3-large"
    dimensions: int     # e.g. 1024

@dataclass
class RouteDefinition:
    """路由类别定义"""
    name: str               # e.g. "simple_control"
    label: str              # e.g. "简单设备控制"
    target: str             # entity_id, e.g. "conversation.llama_fast"
    threshold: float        # 匹配阈值, e.g. 0.70
    utterances: list[str]   # 例句列表, ≥3 条

@dataclass
class RouterConfigData:
    """完整路由配置"""
    embedding: EmbeddingConfig | None
    routes: list[RouteDefinition]
    fallback: str | None    # 兜底后端 entity_id

@dataclass
class BackendInfo:
    """已纳管后端信息"""
    entity_id: str
    name: str
    model_name: str

@dataclass
class RoutingResult:
    """路由决策结果"""
    backend: BackendInfo
    category: str | None
    confidence: float
    source: str  # "semantic" | "fallback"
```

---

## 3. 修改文件

### 3.1 `__init__.py` — 注册 Router Agent（+ ~40 行）

```python
# 新增模块级 async_setup
async def async_setup(hass, config):
    """集成加载时自动注册 Router Agent"""
    # 在首次 setup_entry 时延迟注册，因为需要 DATA_COMPONENT 已就绪
    return True

# 修改 async_setup_entry，在首次调用时注册 Router Agent
async def async_setup_entry(hass: HomeAssistant, entry: LocalLLMConfigEntry) -> bool:
    # ... 现有代码 ...

    # 首次 setup_entry 时注册 Router Agent
    if "router_agent" not in hass.data.get(DOMAIN, {}):
        router = RouterConversationAgent(hass)
        await hass.data[conversation.DATA_COMPONENT].async_add_entities([router])
        hass.data.setdefault(DOMAIN, {})["router_agent"] = router

    return True
```

**关键依赖**：需要 import `conversation.DATA_COMPONENT`。

### 3.2 `const.py` — 新增常量（+ ~20 行）

```python
# Router Agent 相关常量
CONF_ROUTER_EMBEDDING_BASE_URL = "router_embedding_base_url"
CONF_ROUTER_EMBEDDING_API_KEY = "router_embedding_api_key"
CONF_ROUTER_EMBEDDING_MODEL = "router_embedding_model"
CONF_ROUTER_EMBEDDING_DIMENSIONS = "router_embedding_dimensions"
CONF_ROUTER_ROUTES = "router_routes"
CONF_ROUTER_FALLBACK = "router_fallback"

DEFAULT_ROUTER_EMBEDDING_DIMENSIONS = 1024
DEFAULT_ROUTER_THRESHOLD = 0.70
```

### 3.3 `config_flow.py` — Router Agent Options Flow（+ ~200 行）

新增 `RouterOptionsFlowHandler`，包含：

```
Step 1: Embedding 服务配置
├─ Base URL (TextSelector)
├─ API Key (TextSelector, password)
├─ Model (TextSelector)
└─ Dimensions (NumberSelector, 默认 1024)

Step 2: 路由规则列表
├─ 显示已配置的 route 列表（只读摘要）
├─ [+ 添加路由] 按钮 → Step 3
└─ 每条规则的 [编辑] [删除] 按钮

Step 3: 编辑单条路由规则
├─ 名称 (TextSelector)
├─ 标签 (TextSelector)
├─ 阈值 (NumberSelector, 0.0-1.0)
├─ 目标后端 (SelectSelector, 动态从 _discover_backends() 获取)
└─ 例句 (TextSelector, multi-line)

Step 4: 兜底后端选择
└─ SelectSelector, 动态从可用后端获取
```

### 3.4 `entity.py` — 无改动

Router Agent 不继承 `LocalLLMEntity`，直接继承 `ConversationEntity`，所以 entity.py 不需要修改。

### 3.5 `manifest.json` — 版本号更新

```json
{
  "version": "0.5.0"
}
```

---

## 4. 关键设计决策

### 4.1 对话级缓存

```python
_session_backend_cache: dict[str, str]  # conversation_id → target_entity_id

# 缓存策略:
# - 创建: 每轮新对话（新 conversation_id）首次路由时设置
# - 读取: 同 conversation_id 的后续轮次直接复用
# - 失效: 被委托的 entity 不可用时清空该条目，重新路由
# - 清理: 定时清理过期条目（可选）
```

**为什么需要**：
- 保证同一次对话中回复风格一致
- 避免中间插入 tool_call 结果后 re-route 导致上下文不兼容
- 减少 embedding API 调用

### 4.2 兜底策略

```
语义匹配成功但目标后端不可用 → 尝试同 Route 的其他匹配（同 Route 可以有多个 utterance 但只有一个 target）
                             └→ 不可用 → 使用 fallback 后端
语义匹配低于阈值 → 使用 fallback 后端
Embedding API 异常 → 使用 fallback 后端
所有后端都不可用 → 返回用户错误提示
```

### 4.3 Embedding 缓存预热

```python
# 触发时机:
# 1. Router Agent 启动时
# 2. 路由规则配置更新时（用户增删改 Route 后）

# 预热流程:
routes = config_store.routes
all_utterances = []
for route in routes:
    all_utterances.extend(route.utterances)
vectors = await embedding_client.embed(all_utterances)
# 按 Route 分组存储
idx = 0
for route in routes:
    self._route_prototypes[route.name] = vectors[idx:idx + len(route.utterances)]
    idx += len(route.utterances)
self._initialized = True
```

### 4.4 维度配置

Embedding 维度是可配置参数，与 OpenAI 的 Matryoshka 表示学习兼容。推荐值：
- `text-embedding-3-large`: 1024（速度与精度的平衡点）
- `text-embedding-3-small`: 512
- 其他模型按实际输出维度

### 4.5 后端发现时机

`_discover_backends()` 每次 `async_process()` 调用时都执行，保证实时性：
- 新增后端子条目 → 自动出现在可用后端列表
- 删除后端子条目 → 自动从可用后端列表移除
- 路由规则中的 `target` 引用的是后端 entity_id，删除后尝试匹配时会失败，自然降级到下一个规则或 fallback

---

## 5. 实现顺序

### Phase 1：基础骨架（预估 1 天）

| 步骤 | 内容 | 文件 |
|------|------|------|
| 1.1 | 创建数据模型 | `router_config.py` |
| 1.2 | 创建 Config Store（HA Store 持久化） | `router_agent.py` → `RouterConfigStore` |
| 1.3 | 注册 Router Agent 实体 | `__init__.py` |
| 1.4 | 后端发现机制 | `router_agent.py` → `_discover_backends()` |
| 1.5 | 委托调用目标 agent | `router_agent.py` → `async_process()` |

**验证点**：Router Agent 出现在 HA 对话代理列表中，选中后能委托到指定后端

### Phase 2：路由引擎（预估 1.5 天）

| 步骤 | 内容 | 文件 |
|------|------|------|
| 2.1 | EmbeddingClient 实现 | `router_agent.py` |
| 2.2 | TaskRouter 初始化（原型向量预热） | `router_agent.py` |
| 2.3 | 余弦相似度匹配 | `router_agent.py` |
| 2.4 | 兜底逻辑 | `router_agent.py` |
| 2.5 | 对话级缓存 | `router_agent.py` |

**验证点**：通过单元测试验证路由匹配准确性，验证降级路径

### Phase 3：配置 UI（预估 1 天）

| 步骤 | 内容 | 文件 |
|------|------|------|
| 3.1 | Router Options Flow 骨架 | `config_flow.py` |
| 3.2 | Embedding 服务配置 UI | `config_flow.py` |
| 3.3 | 路由规则列表 + CRUD | `config_flow.py` |
| 3.4 | 后端选择器（动态发现） | `config_flow.py` |
| 3.5 | 兜底后端配置 | `config_flow.py` |

**验证点**：完整配置流程可用，保存后规则生效

### Phase 4：集成验证（预估 0.5 天）

| 步骤 | 内容 |
|------|------|
| 4.1 | 与现有后端子条目共存测试 |
| 4.2 | 后端增删的发现测试 |
| 4.3 | 多轮对话路由一致性测试 |
| 4.4 | Embedding API 异常降级测试 |
| 4.5 | 与 Voice Pipeline 集成测试 |

---

## 6. 测试策略

```python
# 单元测试
class TestTaskRouter:
    async def test_semantic_match_returns_highest_score():
        ...
    async def test_low_confidence_fallsback():
        ...
    async def test_embedding_api_down_uses_fallback():
        ...
    async def test_backend_deleted_falls_to_next():
        ...
    async def test_same_conversation_reuses_cache():
        ...

# 集成测试
class TestRouterAgent:
    async def test_discovers_backends():
        """验证能发现已注册的 LocalLLMAgent"""
        ...
    async def test_delegates_to_target():
        """验证委托调用返回正确结果"""
        ...
    async def test_config_persistence():
        """验证配置保存后重启还能正确加载"""
        ...
```

---

## 7. 风险与缓解

| 风险 | 影响 | 缓解 |
|------|------|------|
| `chat_log.llm_api` 在委托链中行为不符合预期 | 工具调用异常 | ✅ 已分析：`async_get_chat_log` 的 copy 机制隔离了 `llm_api`，每轮对话在每个 agent 上下文中独立设置 |
| Embedding API 延迟影响用户体验 | 路由增加 ~100ms | 对话级缓存可减少后续轮次的 embedding 调用；预热机制解决冷启动 |
| 后端列表在配置时和运行时不一致 | 规则引用到已删除的后端 | 降级链自动容错；用户在配置 UI 中可看到后端的实时状态 |
| Router Agent 与 DefaultAgent 冲突 | 用户困惑 | Router Agent 仅在用户主动选择后才接管，不影响 HA 默认代理 |
