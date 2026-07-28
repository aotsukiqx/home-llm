# Home LLM (Router)

> **Fork notice**: This is a fork of [acon96/home-llm](https://github.com/acon96/home-llm) with the addition of **embedding-based semantic routing**. The original integration (`Local LLMs`) focuses on direct LLM backends; this fork adds a Router Agent that automatically dispatches requests to the most appropriate backend based on intent similarity. Both integrations can coexist in the same Home Assistant instance under different domain names.

Control your Home Assistant smart home with a **completely local** Large Language Model. No cloud services and no subscriptions needed. Just privacy-focused AI running entirely on your own hardware.

[![Open your Home Assistant instance and open a repository inside the Home Assistant Community Store.](https://my.home-assistant.io/badges/hacs_repository.svg)](https://my.home-assistant.io/redirect/hacs_repository/?category=Integration&repository=home-llm&owner=aotsukiqx)

## What is Home LLM?

Home LLM is a complete solution for adding AI-powered voice and chat control to Home Assistant. It consists of two parts:

1. **Local LLM Integration** – A Home Assistant custom component that connects local language models to your smart home
2. **Home Models** – Small, efficient AI models fine-tuned specifically for smart home control

### Key Features

- 🏠 **Fully Local** – Everything runs on your hardware. Your data never leaves your control (unless you want to!)
- 🗣️ **Voice & Chat Control** – Use as a conversation agent with voice assistants or chat interfaces
- 🤖 **AI Task Automation** – Generate dynamic content and structured data for automations
- 🔀 **Semantic Routing** *(fork addition)* – Deploy multiple LLM backends and route requests by intent similarity, sending simple commands to fast models and complex tasks to capable models
- 🌍 **Multi-Language Support** – Built-in support for English, German, French, Spanish, and Polish (better translations are welcome!)
- ⚡ **Runs on Low-Power Devices** – Models work on Raspberry Pi and other modest hardware -- no GPU required!
- 🔌 **Flexible Backends** – Run models locally as part of Home Assistant **or** connect to external model providers

## Quick Start

See the [Setup Guide](./docs/Setup.md) for detailed installation instructions.

**Requirements:** Home Assistant 2026.5.0 or newer

---

## Local LLM Integration

The integration connects language models to Home Assistant, enabling them to understand your requests and control your smart devices.

### Supported Backends

Choose how and where you want to run your models:

| Backend                                                                                             | Best For                                                                      |
| --------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| **Llama.cpp** (built-in)                                                                            | Running models directly in Home Assistant                                     |
| **[Ollama](https://ollama.com/)**                                                                   | Easy setup on a separate GPU machine                                          |
| **[Generic OpenAI API](https://platform.openai.com/docs/api-reference/conversations/create)**       | LM Studio, LocalAI, vLLM, and other OpenAI-compatible servers                 |
| **[llama.cpp server](https://github.com/ggml-org/llama.cpp/tree/master/tools/server)**              | Heterogeneous (non-uniform) GPU compute setups, including CPU + GPU inference |
| **[OpenAI 'Responses' Style API](https://platform.openai.com/docs/api-reference/responses/create)** | Cloud services supporting the 'responses' style API                           |
| **[Anthropic 'Messages' Style API](https://platform.claude.com/docs/en/api/messages)**              | Cloud services supporting the 'messages' style API                            |
| **[text-generation-webui](https://github.com/oobabooga/text-generation-webui)**                     | Advanced users with existing setups                                           |

> NOTE: When utilizing **external** APIs or model providers, your data will be transmitted over the internet and shared with the respective service providers. Ensure you understand the privacy implications of using these third-party services, since they will be able to see the status of all exposed entities in your Home Assistant instance, which can potentially include your current location.

### Supported Device Types

The integration can control: **lights, switches, fans, covers, locks, climate, media players, vacuums, buttons, timers, todo lists, and scripts**

### Using the Integration

**As a Conversation Agent:**
- Chat with your assistant through the Home Assistant UI
- Connect to voice pipelines with Speech-to-Text and Text-to-Speech
- Supports voice streaming for faster responses

**As an AI Task Handler:**
- Create automations that use AI to process data and generate structured responses
- Perfect for dynamic content generation, data extraction, and intelligent decision making
- See [AI Tasks documentation](./docs/AI%20Tasks.md) for examples

---

## Home LLM Models

The "Home" models are small language models (under 5B parameters) fine-tuned specifically for smart home control. They understand natural language commands and translate them into Home Assistant service calls.

### Latest Models

| Model Family  | Size | Link                                                                                    |
| ------------- | ---- | --------------------------------------------------------------------------------------- |
| **Llama 3.2** | 3B   | [acon96/Home-Llama-3.2-3B](https://huggingface.co/acon96/Home-Llama-3.2-3B)             |
| **Gemma**     | 270M | [acon96/Home-FunctionGemma-270m](https://huggingface.co/acon96/Home-FunctionGemma-270m) |

<details>
<summary>Previous Model Versions</summary>

**Stable Models:**
- 3B v3 (StableLM-Zephyr-3B): [acon96/Home-3B-v3-GGUF](https://huggingface.co/acon96/Home-3B-v3-GGUF)
- 1B v3 (TinyLlama-1.1B): [acon96/Home-1B-v3-GGUF](https://huggingface.co/acon96/Home-1B-v3-GGUF)
- 3B v2 (Phi-2): [acon96/Home-3B-v2-GGUF](https://huggingface.co/acon96/Home-3B-v2-GGUF)
- 1B v2 (Phi-1.5): [acon96/Home-1B-v2-GGUF](https://huggingface.co/acon96/Home-1B-v2-GGUF)
- 1B v1 (Phi-1.5): [acon96/Home-1B-v1-GGUF](https://huggingface.co/acon96/Home-1B-v1-GGUF)

**Multilingual Experiments:**
- German, French, & Spanish (3B): [acon96/stablehome-multilingual-experimental](https://huggingface.co/acon96/stablehome-multilingual-experimental)
- Polish (1B): [acon96/tinyhome-polish-experimental](https://huggingface.co/acon96/tinyhome-polish-experimental)

> **Note:** Models v1 (3B) and earlier are only compatible with integration version 0.2.17 and older.

</details>

### Using Other Models

Don't have dedicated hardware? You can use any instruction-tuned model with **in-context learning (ICL)**. The integration provides examples that teach general-purpose models (like Qwen3, Llama 3, Mistral) how to control your smart home. See the [Setup Guide](./docs/Setup.md) for configuration details.

### Training Your Own

The fine-tuning dataset and training scripts are included in this repository:
- **Dataset:** [Home-Assistant-Requests-V2](https://huggingface.co/datasets/acon96/Home-Assistant-Requests-V2) on HuggingFace
- **Source:** [data/](./data) directory
- **Training:** See [train/README.md](./train/README.md)

---

## Documentation

- [Setup Guide](./docs/Setup.md) – Installation and configuration
- [Backend Configuration](./docs/Backend%20Configuration.md) – Detailed backend options
- [Model Prompting](./docs/Model%20Prompting.md) – Customize system prompts
- [AI Tasks](./docs/AI%20Tasks.md) – Using AI in automations

---

## Changelog

### v0.5.0 — 2026-07-28

> First release of the **Home LLM (Router)** fork. This version introduces an embedding-based semantic routing layer built on top of the original home-llm integration.

**New features:**
- **Router Agent** — Auto-registered `ConversationEntity` that acts as a smart dispatcher between multiple LLM backends. Select it in your voice pipeline and it automatically routes each request to the best-suited model.
- **Embedding-based semantic routing** — Configure route categories with example utterances. User queries are compared via cosine similarity to determine intent and dispatched to the appropriate backend.
- **OpenAI-compatible Embedding API** — Connect to any embedding service (OpenAI, or any OpenAI-compatible provider) via configurable base URL, API key, model, and dimensions.
- **`router_configure` service** — Programmatic configuration endpoint for setting embedding service, route rules, and fallback backend.
- **OptionsFlow integration** — Embedding service and fallback backend configurable through the existing HA UI options flow.
- **Fallback chain** — When a matched route's target backend is unavailable, automatically falls through to the next matching rule, then to the configured fallback, then to any available backend.

**Fixes & improvements over upstream v0.4.10:**
- Anthropic backend: Fixed authentication header format for third-party API proxies (added Bearer token prefix, removed dummy-key workaround)
- Changed integration domain to `home_llm_router` to allow coexistence with the original `acon96/home-llm` integration

### Post-release updates (2026-07-28)

- **Dedicated Router ConfigEntry** — Router Agent now has its own config entry in the Integrations page. When adding the integration, choose "Configure Router Agent" to create it, or it auto-creates on first backend setup.
- **Route management UI** — Full CRUD UI for routing rules: add/edit/delete rules with backend selection (dropdown from discovered backends) and multi-line utterance input.
- **Route delete step** — Dedicated step for removing routing rules with backend confirmation.
- **ICL CSV files restored** — Fixed missing in_context_examples.csv during domain migration.
- **UI translations** — Added field labels and descriptions for Router Agent config steps (embedding config, route editing).
- **Routing decision logging** — Every routing decision is now logged with query preview, category, confidence, source, and backend name.
- **Router ConfigEntry auto-creation** — Automatic ConfigEntry creation for existing installations that upgrade from earlier versions.
- **Duplicate router guard** — Prevent accidental creation of multiple Router ConfigEntries.
- **Router entry safety** — Protect Router entries in `async_unload_entry` and `async_migrate_entry` to prevent migration code from operating on incompatible data.
- **Anthropic auth fix** — Removed redundant `default_headers` that could conflict with SDK's built-in `x-api-key` authentication.
- **Anthropic empty messages fix** — When `remember_conversation=False` causes `message_history` to lack user input, inject system prompt as fallback user message to satisfy Anthropic API's `messages` requirement.
- **User input in message history** — Fixed upstream bug where `remember_conversation=False` dropped the current user input from `message_history`, affecting all backends (not just Anthropic).
- **Embedding URL normalization** — Strip trailing `/v1` from embedding base URL to prevent double path in API endpoint.
- **Config flow error handling** — Wrapped router config step with exception handler to show clear error messages instead of 500.
- **SelectOptionDict access fix** — Changed `.value` to `["value"]` for HA's TypedDict-based select options.
- **Services.yaml** — Added to suppress HA warning about missing service definition file.

<details>
<summary>Upstream Version History (acon96/home-llm v0.4.10 and earlier)</summary>

| Version    | Highlights                                                                                                                                                                                                                                                                                                          |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **v0.4.10** | Bug fixes and manual parsing improvements for tool calls and thinking blocks                                                                                                                                                                                                                                       |
| **v0.4.9**  | Relax dependency requirements to avoid conflicting with internal HA lib versions                                                                                                                                                                                                                                   |
| **v0.4.8**  | OpenAI backends rewritten using the official openai Python library for better reliability and compatibility. New "Use server sampling defaults" to let your backend set the sampling parameters. More robust tool call parsing with auto-repair for malformed JSON, ability to disable streaming for all backends. |
| **v0.4.7**  | Bug fixes, update default llama_cpp_python version to support new models, and support python 3.14 for new Home Assistant versions                                                                                                                                                                                  |
| **v0.4.6**  | Anthropic API support, on-disk caching for Llama.cpp, new tool calling dataset                                                                                                                                                                                                                                     |
| **v0.4.5**  | AI Task entities, multiple LLM APIs at once, official Ollama package                                                                                                                                                                                                                                               |
| **v0.4**    | Tool calling rewrite, voice streaming, agentic tool use loop, multiple configs per backend                                                                                                                                                                                                                         |
| **v0.3**    | Home Assistant LLM API support, improved prompting, HuggingFace GGUF auto-detection                                                                                                                                                                                                                                |

</details>

