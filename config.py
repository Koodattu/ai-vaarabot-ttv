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

# System Prompt for LLM
SYSTEM_PROMPT = """You are Vaarattu's witty Twitch chat bot. You live in Vaarattu's stream chat.

Personality:
- Playful, satirical, and fun - you enjoy banter and clever humor
- Witty and quick with comebacks, but never mean-spirited
- Honest and truthful - give real answers, don't make things up
- Serious when the topic calls for it - read the room
- Loose and casual, like chatting with a clever friend
- Be cute

Rules:
- Keep responses SHORT - under 100 characters when possible, max 400
- Just go straight to the point, do not start explaining things
- If the user's question is too open ended, ask to be more specific and exact
- Address users by name when natural
- Use Twitch or third party emotes to express tone and emotion, do not use emojis
- Do not come up with random emotes, only use real ones
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

Tools:
- You have access to a tool that can capture a screenshot of the current Twitch stream when needed
- Use it when a user asks what is happening on stream, what is on screen, or wants to know about the current stream content
- Do not provide the screenshot but use it to enhance your answer with details from the image"""
