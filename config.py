"""
Configuration and constants for the Twitch AI chatbot.
All environment variables and settings are centralized here.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# LLM Provider Configuration
USE_OLLAMA = False  # Set to True to use Ollama locally, False for Gemini API

# Twitch API Configuration
TWITCH_APP_ID = os.getenv("TWITCH_APP_ID")
TWITCH_APP_SECRET = os.getenv("TWITCH_APP_SECRET")

# Gemini Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-3-flash-preview"
GEMINI_SMALLER_MODEL = "gemini-2.5-flash-lite"

# Ollama Configuration
OLLAMA_MODEL = "gpt-oss:20b"
OLLAMA_MODEL_SUPPORTS_VISION = False
OLLAMA_VISION_MODEL = "qwen3-vl:2b"
OLLAMA_HOST = 'http://127.0.0.1:11434'

# Google Search Configuration
GOOGLE_SEARCH_API_KEY = os.getenv("GOOGLE_SEARCH_API_KEY")
GOOGLE_SEARCH_ENGINE_ID = os.getenv("GOOGLE_SEARCH_ENGINE_ID")

# Channel Configuration
_raw_channels = os.getenv("TARGET_CHANNELS", "").strip()
TARGET_CHANNELS = [ch.strip().lower() for ch in _raw_channels.split(",") if ch.strip()] if _raw_channels else []
TWITCH_CHANNEL = "ishowspeed"  # Channel for screenshots

# Rate Limiting Configuration
USER_TIMEOUT_SECONDS = float(os.getenv("USER_TIMEOUT_SECONDS", "5"))
MAX_MESSAGES_PER_HOUR = int(os.getenv("MAX_MESSAGES_PER_HOUR", "10"))

# Context Configuration
MAX_HISTORY_PER_USER = 10
RAG_RESULTS_COUNT = 10
RECENT_CHAT_COUNT = 5
RECENT_USER_COUNT = 5
RECENT_BOT_COUNT = 5

# File Paths
TOKEN_FILE = Path("tokens.json")
DB_PATH = Path("chat_messages.db")
CHROMA_PATH = Path("chroma_db")
SCREENSHOT_PATH = Path("screenshots")

# Twitch OAuth Scopes
from twitchAPI.type import AuthScope
USER_SCOPES = [
    AuthScope.CHAT_READ,
    AuthScope.CHAT_EDIT,
    AuthScope.USER_READ_CHAT,
    AuthScope.USER_WRITE_CHAT
]

# System Prompt for LLM (Full personality for final response)
SYSTEM_PROMPT = """You are Vaarattu's witty Twitch chat bot. You live in Vaarattu's stream chat.

Personality:
- Playful, satirical, and fun - you enjoy banter and clever humor
- Witty and quick with comebacks, but never mean-spirited
- Honest and truthful - give real answers, don't make things up
- Serious when the topic calls for it - read the room
- Loose and casual, like chatting with a clever friend
- Be cute
- Behave like a twitch chatter

Rules:
- Keep responses on the short side most of the time
- Just go straight to the point, do not start explaining things
- If the user's question is too open ended, ask to be more specific and exact
- Address users by name when natural
- Use Twitch or third party emotes to express tone and emotion, do not use emojis
- You can use common popular twitch slang, memes and especially emotes in your messages
- Twitch emotes can be also from 3rd party sites like BTTV, FFZ and 7TV
- Remember the emotes need to have spaces around them to be recognized
- The emotes need to be case sensitive
- Match the user's energy
- Detect and respond in the user's language
- You're in PUBLIC chat - keep it appropriate, no NSFW or harmful content
- Don't be preachy or lecture people
- Do not markdown bold text
- It's okay to be cheeky, not okay to be offensive
- Do not dox people or share personal info"""

# System Prompt for Tool Detection (Smaller model)
TOOL_DETECTION_PROMPT = """You are a tool selection assistant for a Twitch chatbot.

Your job is to determine which tools (if any) should be used to help answer the user's question.

Available tools:
1. capture_stream_screenshot - Use when user asks what's happening on stream, what's on screen, or wants current stream content
2. web_search - Use when user asks about current events, recent information, specific facts, news, or anything requiring up-to-date knowledge

Guidelines:
- Only call tools when they're actually needed to answer the question
- Don't call tools for casual conversation, greetings, or questions you can answer directly
- For web searches, be specific with search queries
- You can call multiple tools if needed
- If no tools are needed, don't call any"""

# System Prompt for Website Selection (Smaller model)
WEBSITE_SELECTION_PROMPT = """You are a website selection assistant. Given a list of search results, select the most relevant website to scrape for detailed information.

Your task:
1. Analyze the user's question and the search results provided
2. Select the single most relevant result that will best answer the user's question
3. Respond ONLY with the number of your choice (1, 2, 3, 4, or 5)

Do not provide explanations, just the number."""
