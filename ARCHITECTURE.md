# Project Architecture

This document describes the modular architecture of the Twitch AI chatbot.

## Module Overview

The project has been refactored from a single monolithic file into a clean, modular architecture:

```
ai-vaarabot-ttv/
├── main.py              # Entry point & orchestration
├── config.py            # Configuration & environment variables
├── auth.py              # Twitch OAuth & token management
├── database.py          # SQLite + ChromaDB operations
├── rate_limit.py        # Cooldown & rate limiting
├── tools.py             # Tool functions (screenshot, search)
├── handlers.py          # Chat event handlers
└── llm/
    ├── __init__.py
    ├── base.py         # Base LLM interface
    ├── gemini.py       # Gemini implementation
    └── ollama.py       # Ollama implementation
```

## Module Responsibilities

### `config.py`

Centralizes all configuration and constants:

- Environment variables loading
- LLM provider settings (Gemini/Ollama)
- API keys and credentials
- Rate limiting settings
- File paths and directories
- System prompts

**Why separate?** Makes it easy to adjust settings without diving into code logic. Future improvements could include a config UI or dynamic reloading.

### `database.py`

Manages all data persistence:

- SQLite connection for structured message storage
- ChromaDB for vector search/RAG
- Message retrieval (recent, similar, by user)
- Context building for LLM prompts

**Why separate?** Database logic is complex and independent. This allows easy swapping to other DBs (PostgreSQL, Redis) or adding features like message analytics.

### `auth.py`

Handles Twitch authentication:

- OAuth flow with browser
- Token storage and loading
- Token validation and refresh

**Why separate?** Authentication is security-critical and self-contained. Separation makes it easier to audit, test, and extend (e.g., multi-account support).

### `rate_limit.py`

Manages anti-spam mechanisms:

- Per-user cooldowns
- Hourly message limits
- Timestamp tracking

**Why separate?** Rate limiting is a distinct concern that could be extended with more sophisticated algorithms or Redis-based distributed rate limiting.

### `tools.py`

Contains bot tool functions:

- Screenshot capture from stream
- Web search integration
- Tool declarations for LLMs

**Why separate?** Tools are independent capabilities. This structure makes it trivial to add new tools (weather, games, database queries) without touching other code.

### `llm/` Module

Abstracts LLM providers:

- **`base.py`**: Abstract interface all LLMs must implement
- **`gemini.py`**: Google Gemini implementation with vision & tools
- **`ollama.py`**: Local Ollama implementation with separate vision model

**Why separate?** Clean abstraction allows:

- Easy addition of new providers (OpenAI, Anthropic, local models)
- Provider-specific optimizations
- Runtime switching between providers
- Independent testing

### `handlers.py`

Manages Twitch chat events:

- `on_ready` - Bot initialization
- `on_message` - Message processing and response logic
- Mode switching commands

**Why separate?** Event handling logic is complex and stateful. Separation enables easier testing and the potential for multiple handler strategies.

### `main.py`

Orchestrates all modules:

- Validates environment
- Initializes all subsystems
- Wires dependencies together
- Manages bot lifecycle

**Why small?** Main should be thin - just setup and coordination. All logic lives in specialized modules.

## Design Principles

### 1. **Separation of Concerns**

Each module has a single, well-defined responsibility. Changes to database logic don't affect rate limiting, authentication, or LLM providers.

### 2. **Dependency Injection**

Components receive their dependencies (database, rate limiter, LLM) rather than creating them. This enables:

- Easy testing with mocks
- Runtime reconfiguration
- Clear dependency graphs

### 3. **Abstraction**

The `BaseLLM` interface means handlers don't care whether they're using Gemini or Ollama. This makes adding new providers trivial.

### 4. **Future-Proof**

The architecture anticipates growth:

- **More tools**: Just add to `tools.py`
- **New LLM providers**: Implement `BaseLLM` interface
- **Plugin system**: Tools and handlers are already modular
- **Web dashboard**: Can import and use these modules
- **Multi-bot**: Database already supports channel scoping
- **Analytics**: Database module is self-contained

## Running the Bot

The bot still runs exactly the same way:

```bash
python main.py
```

All functionality is preserved - the refactoring is purely structural with no behavioral changes.

## Testing Individual Modules

Each module can be imported and tested independently:

```python
# Test database
from database import Database
db = Database()
db.initialize()

# Test LLM
from llm import GeminiLLM
llm = GeminiLLM()
llm.test_connection()

# Test rate limiting
from rate_limit import RateLimiter
limiter = RateLimiter()
```

## Future Enhancements

With this architecture, the following are now much easier to implement:

1. **Plugin System** - Load tools dynamically from a `plugins/` directory
2. **Web Dashboard** - Import modules for monitoring and configuration UI
3. **Multiple Bots** - Database already supports it, just need multiple instances
4. **Command System** - Add `commands.py` for custom chat commands
5. **Event Logging** - Add `logging.py` for structured logs and analytics
6. **Caching Layer** - Add `cache.py` for response caching and optimization
7. **A/B Testing** - Easy to swap LLM providers or prompts per channel
8. **Backup/Export** - Database module makes data portability simple

## Migration Notes

- All original functionality is preserved
- Configuration values moved from code to `config.py`
- Global state eliminated - everything is properly scoped
- The bot still uses the same database files and token storage
