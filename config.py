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

# Streamlink Configuration
STREAMLINK_OAUTH_TOKEN = os.getenv("STREAMLINK_OAUTH_TOKEN")

# Channel Configuration
_raw_channels = os.getenv("TARGET_CHANNELS", "").strip()
TARGET_CHANNELS = [ch.strip().lower() for ch in _raw_channels.split(",") if ch.strip()] if _raw_channels else []
TWITCH_CHANNEL = "ishowspeed"  # Channel for screenshots

# Rate Limiting Configuration
USER_TIMEOUT_SECONDS = float(os.getenv("USER_TIMEOUT_SECONDS", "5"))
MAX_MESSAGES_PER_HOUR = int(os.getenv("MAX_MESSAGES_PER_HOUR", "10"))

# Ad Detection Configuration
AD_DETECTION_ENABLED = os.getenv("AD_DETECTION_ENABLED", "true").lower() == "true"
AD_DETECTION_CHECK_INTERVAL = float(os.getenv("AD_DETECTION_CHECK_INTERVAL", "5.0"))  # Check every 5 seconds
AD_DETECTION_MAX_WAIT = float(os.getenv("AD_DETECTION_MAX_WAIT", "120.0"))  # Max 30 seconds wait

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
- Keep responses on the short side most of the time, just like most Twitch chatters do, sometimes even just a single emote
- Really prefer short answers unless the question demands more detail
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
TOOL_DETECTION_PROMPT = f"""You are a tool selection assistant for a Twitch chatbot.

Your job is to determine which tools (if any) should be used to help answer the user's question.

Available tools:
1. capture_stream_screenshot - Use when user asks what's happening on stream, what's on screen, or wants current stream content.
   - IMPORTANT: This tool automatically checks if the stream is live first. If offline, it returns an error - you should inform the user the stream is offline.
   - Has optional 'channel' parameter to specify which Twitch channel to capture
   - Default channel is '{TWITCH_CHANNEL}' (use this most of the time unless user asks about a different channel)
2. web_search - Use when user asks about current events, recent information, specific facts, news, or anything requiring up-to-date knowledge

Guidelines:
- Only call tools when they're actually needed to answer the question
- Don't call tools for casual conversation, greetings, or questions you can answer directly
- For web searches, be specific with search queries
- You can call multiple tools if needed
- If no tools are needed, don't call any
- When screenshot tool returns offline status, inform the user naturally"""

# System Prompt for Website Selection (Smaller model)
WEBSITE_SELECTION_PROMPT = """You are a website selection assistant. Given a list of search results, select the most relevant website to scrape for detailed information.

Your task:
1. Analyze the user's question and the search results provided
2. Select the single most relevant result that will best answer the user's question
3. Respond ONLY with the number of your choice (1, 2, 3, 4, or 5)

Do not provide explanations, just the number."""

# System Prompt for Ad Wait Message Generation (Smaller model)
AD_WAIT_PROMPT = """Generate a super short Twitch chat message (max 30 characters) telling the user you're waiting for ads to finish.

Rules:
- Match the language of the user's message
- Use casual Twitch chatter style
- Be brief and friendly
- Can use common Twitch emotes if appropriate
- Just say you're waiting for ads, nothing else"""
