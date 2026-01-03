"""
Simple Twitch Chat Bot with Gemini AI / Ollama
Responds when pinged with @botname
Refactored into modular architecture for maintainability and extensibility.
"""

import asyncio
from twitchAPI.twitch import Twitch
from twitchAPI.type import ChatEvent
from twitchAPI.chat import Chat

# Import all modules
import config
from auth import TwitchAuth
from database import Database
from rate_limit import RateLimiter
from tools import init_screenshots_dir
from llm import GeminiLLM, OllamaLLM
from handlers import ChatHandlers

async def run() -> None:
    """Main bot entry point."""
    print("=" * 50)
    print("  Twitch AI Chat Bot")
    print("=" * 50)

    # Validate environment
    missing = []
    if not config.TWITCH_APP_ID:
        missing.append("TWITCH_APP_ID")
    if not config.TWITCH_APP_SECRET:
        missing.append("TWITCH_APP_SECRET")
    if not config.USE_OLLAMA and not config.GEMINI_API_KEY:
        missing.append("GEMINI_API_KEY")

    if missing:
        print(f"\n✗ Missing environment variables: {', '.join(missing)}")
        print("  Please check your .env file")
        return

    # Initialize LLM provider
    if config.USE_OLLAMA:
        llm_provider = OllamaLLM()
    else:
        llm_provider = GeminiLLM()

    # Test the selected LLM provider
    if not llm_provider.test_connection():
        print(f"\n✗ Cannot start without working {'Ollama' if config.USE_OLLAMA else 'Gemini'} API")
        return

    # Initialize databases and directories
    print("\n📦 Initializing databases...")
    database = Database()
    database.initialize()
    init_screenshots_dir()

    # Initialize rate limiter
    rate_limiter = RateLimiter()

    # Initialize Twitch API
    print("\n🔌 Connecting to Twitch...")
    twitch = await Twitch(config.TWITCH_APP_ID, config.TWITCH_APP_SECRET)

    # Authenticate
    auth = TwitchAuth()
    tokens = await auth.authenticate(twitch)

    if not tokens:
        print("\n✗ Authentication failed. Exiting.")
        await twitch.close()
        return

    # Create chat instance
    chat = await Chat(twitch)

    # Create handlers with dependencies
    handlers = ChatHandlers(database, rate_limiter, llm_provider)

    # Register event handlers
    chat.register_event(ChatEvent.READY, handlers.on_ready)
    chat.register_event(ChatEvent.MESSAGE, handlers.on_message)

    # Start the bot
    chat.start()

    print("\n" + "=" * 50)
    print("  Bot is running! Press Ctrl+C to stop")
    print("=" * 50)

    try:
        # Keep running
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    finally:
        chat.stop()
        await twitch.close()
        database.close()
        print("Bot stopped.")


if __name__ == "__main__":
    asyncio.run(run())

