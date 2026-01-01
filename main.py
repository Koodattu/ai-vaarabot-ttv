"""
Simple Twitch Chat Bot with Gemini AI
Responds when pinged with @botname
"""

import os
import json
import asyncio
import sqlite3
import time
from pathlib import Path
from collections import defaultdict
from dotenv import load_dotenv
from twitchAPI.twitch import Twitch
from twitchAPI.oauth import UserAuthenticator
from twitchAPI.type import AuthScope, ChatEvent
from twitchAPI.chat import Chat, EventData, ChatMessage
from google import genai
from google.genai import types
import chromadb
from chromadb.config import Settings
import subprocess
import streamlink

# Load environment variables
load_dotenv()

# LLM Provider Switch: Set to True to use Ollama locally, False for Gemini API
USE_OLLAMA = True

# Configuration
TWITCH_APP_ID = os.getenv("TWITCH_APP_ID")
TWITCH_APP_SECRET = os.getenv("TWITCH_APP_SECRET")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# Channels to join (comma-separated, empty = bot's own channel)
_raw_channels = os.getenv("TARGET_CHANNELS", "").strip()
TARGET_CHANNELS = [ch.strip().lower() for ch in _raw_channels.split(",") if ch.strip()] if _raw_channels else []
TOKEN_FILE = Path("tokens.json")
USER_TIMEOUT_SECONDS = float(os.getenv("USER_TIMEOUT_SECONDS", "5"))  # Cooldown per user
MAX_MESSAGES_PER_HOUR = int(os.getenv("MAX_MESSAGES_PER_HOUR", "10"))  # Max responses per user per hour
MAX_HISTORY_PER_USER = 10  # Keep last 10 exchanges per user
RAG_RESULTS_COUNT = 10  # Number of similar messages to retrieve for RAG
RECENT_CHAT_COUNT = 5  # Recent chat messages to include
RECENT_USER_COUNT = 5  # Recent messages from specific user
RECENT_BOT_COUNT = 5  # Recent bot messages to include

OLLAMA_MODEL = "gpt-oss:20b"#"hf.co/mradermacher/Llama-Poro-2-8B-Instruct-GGUF:Q4_K_M"
OLLAMA_VISION_MODEL = "qwen3-vl:2b"

# Database paths
DB_PATH = Path("chat_messages.db")
CHROMA_PATH = Path("chroma_db")
SCREENSHOT_PATH = Path("screenshots")

# Twitch channel for screenshots
TWITCH_CHANNEL = "paulinpelivideot"

# Required scopes for chat read/write
USER_SCOPES = [AuthScope.CHAT_READ, AuthScope.CHAT_EDIT, AuthScope.USER_READ_CHAT, AuthScope.USER_WRITE_CHAT]

# Gemini model
GEMINI_MODEL = "gemini-3-flash-preview"

# System prompt for brief responses
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
- It's okay to be cheeky, not okay to be offensive"""

# Store last message time per user for cooldown
user_last_message: dict[str, float] = {}
# Store message timestamps per user for hourly rate limit
user_message_timestamps: dict[str, list[float]] = defaultdict(list)

# Bot info (set after login)
bot_username: str = ""
bot_user_id: str = ""

# Database connection and ChromaDB collection
db_conn: sqlite3.Connection | None = None
chroma_collection = None


def init_screenshots_dir() -> None:
    """Initialize screenshots directory."""
    SCREENSHOT_PATH.mkdir(exist_ok=True)
    print(f"✓ Screenshots directory: {SCREENSHOT_PATH}")


def capture_stream_screenshot() -> dict:
    """Capture a screenshot from the Twitch stream using streamlink + ffmpeg.

    Returns a dict with success status, file path, and any error message.
    """
    channel_url = f"https://www.twitch.tv/{TWITCH_CHANNEL}"
    timestamp = int(time.time())
    output_file = SCREENSHOT_PATH / f"screenshot_{timestamp}.jpg"

    try:
        # Get stream URL via streamlink
        streams = streamlink.streams(channel_url)

        if not streams:
            return {"success": False, "error": "Stream is offline", "file_path": None}

        # Get best quality stream URL
        stream_url = streams.get('best') or streams.get('720p') or streams.get('480p')
        if not stream_url:
            return {"success": False, "error": "No suitable stream quality found", "file_path": None}

        stream_url = stream_url.url

        # Use FFmpeg to grab 1 frame
        command = [
            'ffmpeg',
            '-i', stream_url,
            '-ss', '00:00:01',
            '-frames:v', '1',
            '-q:v', '2',
            str(output_file),
            '-y'
        ]

        result = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=30
        )

        if result.returncode == 0 and output_file.exists():
            print(f"✓ Screenshot saved: {output_file}")
            return {"success": True, "file_path": str(output_file), "error": None}
        else:
            return {"success": False, "error": "FFmpeg failed to capture frame", "file_path": None}

    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Screenshot capture timed out", "file_path": None}
    except Exception as e:
        return {"success": False, "error": str(e), "file_path": None}


# Gemini tool declaration for screenshot
SCREENSHOT_TOOL_DECLARATION = {
    "name": "capture_stream_screenshot",
    "description": "Captures a screenshot of the current Twitch livestream. Use this when a user asks to see what's happening on stream, wants to know what's on screen, or asks about the current stream content.",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": []
    }
}


def init_database() -> None:
    """Initialize SQLite database for storing all chat messages."""
    global db_conn
    db_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    db_conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            channel TEXT NOT NULL,
            user_id TEXT NOT NULL,
            user_name TEXT NOT NULL,
            message TEXT NOT NULL,
            is_bot INTEGER DEFAULT 0
        )
    """)
    db_conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON messages(timestamp)")
    db_conn.execute("CREATE INDEX IF NOT EXISTS idx_user_id ON messages(user_id)")
    db_conn.execute("CREATE INDEX IF NOT EXISTS idx_is_bot ON messages(is_bot)")
    db_conn.execute("CREATE INDEX IF NOT EXISTS idx_channel ON messages(channel)")
    db_conn.execute("CREATE INDEX IF NOT EXISTS idx_channel_timestamp ON messages(channel, timestamp)")
    db_conn.commit()
    print(f"✓ SQLite database initialized: {DB_PATH}")


def init_chromadb() -> None:
    """Initialize ChromaDB for vector search."""
    global chroma_collection
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    chroma_collection = client.get_or_create_collection(
        name="chat_messages",
        metadata={"hnsw:space": "cosine"}
    )
    print(f"✓ ChromaDB initialized: {CHROMA_PATH}")


def store_message(channel: str, user_id: str, user_name: str, message: str, is_bot: bool = False) -> int:
    """Store a message in SQLite and ChromaDB. Returns the message ID."""
    global db_conn, chroma_collection

    timestamp = time.time()
    channel = channel.lower()  # Normalize channel name

    # Store in SQLite
    cursor = db_conn.execute(
        "INSERT INTO messages (timestamp, channel, user_id, user_name, message, is_bot) VALUES (?, ?, ?, ?, ?, ?)",
        (timestamp, channel, user_id, user_name, message, 1 if is_bot else 0)
    )
    db_conn.commit()
    msg_id = cursor.lastrowid

    # Store in ChromaDB for vector search
    # Format: "username: message" for better context
    doc_text = f"{user_name}: {message}"
    chroma_collection.add(
        documents=[doc_text],
        metadatas=[{"channel": channel, "user_id": user_id, "user_name": user_name, "is_bot": is_bot, "timestamp": timestamp}],
        ids=[str(msg_id)]
    )

    return msg_id


def get_recent_chat_messages(channel: str, limit: int = 5) -> list[dict]:
    """Get the most recent chat messages from any user in a specific channel."""
    cursor = db_conn.execute(
        "SELECT user_name, message, is_bot FROM messages WHERE channel = ? ORDER BY timestamp DESC LIMIT ?",
        (channel.lower(), limit)
    )
    rows = cursor.fetchall()
    return [{"user": row[0], "message": row[1], "is_bot": bool(row[2])} for row in reversed(rows)]


def get_recent_user_messages(channel: str, user_id: str, limit: int = 5) -> list[dict]:
    """Get the most recent messages from a specific user in a specific channel."""
    cursor = db_conn.execute(
        "SELECT user_name, message FROM messages WHERE channel = ? AND user_id = ? AND is_bot = 0 ORDER BY timestamp DESC LIMIT ?",
        (channel.lower(), user_id, limit)
    )
    rows = cursor.fetchall()
    return [{"user": row[0], "message": row[1]} for row in reversed(rows)]


def get_recent_bot_messages(channel: str, limit: int = 5) -> list[dict]:
    """Get the most recent messages from the bot in a specific channel."""
    cursor = db_conn.execute(
        "SELECT user_name, message FROM messages WHERE channel = ? AND is_bot = 1 ORDER BY timestamp DESC LIMIT ?",
        (channel.lower(), limit)
    )
    rows = cursor.fetchall()
    return [{"user": row[0], "message": row[1]} for row in reversed(rows)]


def search_similar_messages(channel: str, query: str, limit: int = 10) -> list[dict]:
    """Search for similar messages using ChromaDB vector search, filtered by channel."""
    global chroma_collection

    # Check if collection has any documents
    if chroma_collection.count() == 0:
        return []

    results = chroma_collection.query(
        query_texts=[query],
        n_results=min(limit * 3, chroma_collection.count()),  # Fetch more to filter
        where={"channel": channel.lower()}
    )

    messages = []
    if results and results['documents'] and results['documents'][0]:
        for i, doc in enumerate(results['documents'][0]):
            if len(messages) >= limit:
                break
            metadata = results['metadatas'][0][i] if results['metadatas'] else {}
            messages.append({
                "text": doc,
                "is_bot": metadata.get("is_bot", False),
                "distance": results['distances'][0][i] if results['distances'] else 0
            })

    return messages


def build_context_for_response(channel: str, user_id: str, user_name: str, current_message: str) -> str:
    """Build context string for Gemini from various sources, scoped to a specific channel."""
    context_parts = []

    # 1. Recent chat messages (last 5 from anyone in this channel)
    recent_chat = get_recent_chat_messages(channel, RECENT_CHAT_COUNT)
    if recent_chat:
        chat_lines = []
        for msg in recent_chat:
            prefix = "[BOT]" if msg["is_bot"] else ""
            chat_lines.append(f"{prefix}{msg['user']}: {msg['message']}")
        context_parts.append("=== RECENT CHAT ===\n" + "\n".join(chat_lines))

    # 2. Recent messages from this specific user in this channel
    recent_user = get_recent_user_messages(channel, user_id, RECENT_USER_COUNT)
    if recent_user:
        user_lines = [f"{msg['user']}: {msg['message']}" for msg in recent_user]
        context_parts.append(f"=== RECENT FROM {user_name.upper()} ===\n" + "\n".join(user_lines))

    # 3. Recent bot messages in this channel
    recent_bot = get_recent_bot_messages(channel, RECENT_BOT_COUNT)
    if recent_bot:
        bot_lines = [f"{msg['user']}: {msg['message']}" for msg in recent_bot]
        context_parts.append("=== RECENT BOT RESPONSES ===\n" + "\n".join(bot_lines))

    # 4. RAG - similar messages from history in this channel
    similar = search_similar_messages(channel, current_message, RAG_RESULTS_COUNT)
    if similar:
        rag_lines = [msg["text"] for msg in similar]
        context_parts.append("=== RELATED PAST MESSAGES ===\n" + "\n".join(rag_lines))

    return "\n\n".join(context_parts)


def save_tokens(access_token: str, refresh_token: str) -> None:
    """Save tokens to file."""
    with open(TOKEN_FILE, "w") as f:
        json.dump({"access_token": access_token, "refresh_token": refresh_token}, f)


def load_tokens() -> tuple[str, str] | None:
    """Load tokens from file if they exist."""
    if TOKEN_FILE.exists():
        try:
            with open(TOKEN_FILE) as f:
                data = json.load(f)
                return data.get("access_token"), data.get("refresh_token")
        except (json.JSONDecodeError, KeyError):
            return None
    return None


async def validate_and_refresh_token(twitch: Twitch, access_token: str, refresh_token: str) -> tuple[str, str] | None:
    """Validate token and refresh if needed. Returns new tokens or None if failed."""
    try:
        # Try to set the user authentication
        await twitch.set_user_authentication(access_token, USER_SCOPES, refresh_token)

        # Validate by getting user info
        users = twitch.get_users()
        async for u in users:
            print(f"✓ Logged in as: {u.display_name} (ID: {u.id})")
            return access_token, refresh_token
    except Exception as e:
        print(f"Token validation failed: {e}")

        # Try to refresh
        try:
            from twitchAPI.oauth import refresh_access_token
            new_token, new_refresh = await refresh_access_token(refresh_token, TWITCH_APP_ID, TWITCH_APP_SECRET)
            await twitch.set_user_authentication(new_token, USER_SCOPES, new_refresh)

            # Validate again
            users = twitch.get_users()
            async for u in users:
                print(f"✓ Token refreshed. Logged in as: {u.display_name} (ID: {u.id})")
                save_tokens(new_token, new_refresh)
                return new_token, new_refresh
        except Exception as refresh_error:
            print(f"Token refresh failed: {refresh_error}")

    return None


async def do_auth_flow(twitch: Twitch) -> tuple[str, str] | None:
    """Perform OAuth flow with browser and callback server."""
    print("\n🔐 Starting OAuth flow...")
    print("A browser window will open for Twitch login.")

    auth = UserAuthenticator(twitch, USER_SCOPES, force_verify=False)

    try:
        token, refresh_token = await auth.authenticate()
        await twitch.set_user_authentication(token, USER_SCOPES, refresh_token)
        save_tokens(token, refresh_token)
        print("✓ Authentication successful!")
        return token, refresh_token
    except Exception as e:
        print(f"✗ Authentication failed: {e}")
        return None


def test_gemini() -> bool:
    """Test Gemini API with a simple prompt."""
    print("\n🤖 Testing Gemini API...")
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents="Say 'OK' if you can hear me.",
            config=types.GenerateContentConfig(
                max_output_tokens=100,
                thinking_config=types.ThinkingConfig(thinking_level="low")
            )
        )
        print(f"✓ Gemini response: {response.text.strip()}")
        return True
    except Exception as e:
        print(f"✗ Gemini API test failed: {e}")
        return False


def test_ollama() -> bool:
    """Test Ollama API with both main model and vision model."""
    from ollama import Client
    client = Client(host='http://127.0.0.1:11434')

    # Test main model
    print(f"\n🤖 Testing Ollama API (model: {OLLAMA_MODEL})...")
    try:
        response = client.chat(
            model=OLLAMA_MODEL,
            messages=[{'role': 'user', 'content': "Say 'OK' if you can hear me."}]
        )
        print(f"✓ Ollama response: {response.message.content.strip()}")
    except Exception as e:
        print(f"✗ Ollama API test failed: {e}")
        return False

    # Test vision model
    print(f"🤖 Testing Ollama vision model: {OLLAMA_VISION_MODEL}...")
    try:
        response = client.chat(
            model=OLLAMA_VISION_MODEL,
            messages=[{'role': 'user', 'content': "Say 'OK' if you can hear me."}]
        )
        print(f"✓ Vision model response: {response.message.content.strip()}")
    except Exception as e:
        print(f"✗ Ollama vision model test failed: {e}")
        return False

    return True


def describe_image_ollama(image_path: str) -> str:
    """Use vision model to describe an image. Returns text description."""
    from ollama import Client
    client = Client(host='http://127.0.0.1:11434')

    response = client.chat(
        model=OLLAMA_VISION_MODEL,
        messages=[{
            'role': 'user',
            'content': 'Describe this stream screenshot briefly. What game/activity, what is on screen, any visible text.',
            'images': [image_path]
        }]
    )
    return response.message.content.strip()


def should_capture_screenshot(message: str) -> bool:
    """Check if the message is asking about what's on stream."""
    keywords = ['stream', 'screen', 'screenshot', 'what is on', "what's on", 'näytöllä', 'ruudulla', 'striimissä', 'mitä tapahtuu']
    message_lower = message.lower()
    return any(kw in message_lower for kw in keywords)


def get_ollama_response(channel: str, user_id: str, user_name: str, message: str) -> str:
    """Get response from Ollama with RAG context, scoped to a specific channel."""
    try:
        # Build context from database and RAG (channel-scoped)
        context = build_context_for_response(channel, user_id, user_name, message)

        # Check if user is asking about the stream
        image_description = None
        if should_capture_screenshot(message):
            print("[Ollama] Screenshot requested, capturing...")
            result = capture_stream_screenshot()
            if result["success"]:
                print(f"[Ollama] Screenshot captured: {result['file_path']}")
                try:
                    image_description = describe_image_ollama(result["file_path"])
                    print(f"[Ollama] Image description: {image_description[:100]}...")
                except Exception as e:
                    print(f"[Ollama] Vision model failed: {e}")
                    image_description = f"(Failed to analyze screenshot: {e})"
            else:
                print(f"[Ollama] Screenshot failed: {result['error']}")
                image_description = f"(Screenshot capture failed: {result['error']})"

        # Create the prompt with context
        user_prompt = f"{user_name}: {message}"

        if context:
            full_prompt = f"""Here is context from the chat history:

{context}

=== CURRENT MESSAGE ===
{user_prompt}"""
        else:
            full_prompt = user_prompt

        # Add image description if we captured one
        if image_description:
            full_prompt += f"\n\n=== STREAM SCREENSHOT ANALYSIS ===\n{image_description}"

        print(f"[Ollama Prompt]\n{full_prompt}\n")

        from ollama import Client
        client = Client(host='http://127.0.0.1:11434')
        response = client.chat(
            model=OLLAMA_MODEL,
            messages=[
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': full_prompt}
            ],
            options={'temperature': 0.7}
        )

        response_text = response.message.content.strip() if response.message.content else "I couldn't generate a response."

        return response_text

    except Exception as e:
        print(f"Ollama error: {e}")
        return "Sorry, I couldn't process that right now."


def get_gemini_response(channel: str, user_id: str, user_name: str, message: str) -> str:
    """Get response from Gemini with RAG context and tool support, scoped to a specific channel."""
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)

        # Build context from database and RAG (channel-scoped)
        context = build_context_for_response(channel, user_id, user_name, message)

        # Create the prompt with context
        user_prompt = f"{user_name}: {message}"

        if context:
            full_prompt = f"""Here is context from the chat history:

{context}

=== CURRENT MESSAGE ===
{user_prompt}"""
        else:
            full_prompt = user_prompt

        # Tools: Screenshot function (can't mix with google_search)
        tools = types.Tool(function_declarations=[SCREENSHOT_TOOL_DECLARATION])

        print(f"[Gemini Prompt]\n{full_prompt}\n")

        # Build conversation messages
        messages = [types.Content(role="user", parts=[types.Part(text=full_prompt)])]

        # Initial request with manual function calling
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=messages,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                max_output_tokens=400,
                temperature=0.7,
                tools=[tools],
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                thinking_config=types.ThinkingConfig(thinking_level="low")
            )
        )

        # Check for function call and handle it
        response_text = handle_gemini_response(client, messages, response)

        return response_text

    except Exception as e:
        print(f"Gemini error: {e}")
        return "Sorry, I couldn't process that right now."


def handle_gemini_response(client, messages: list, response) -> str:
    """Handle Gemini response, including function calls."""
    # Check if there's a function call in the response
    if (response.candidates and
        response.candidates[0].content and
        response.candidates[0].content.parts):

        for part in response.candidates[0].content.parts:
            if hasattr(part, 'function_call') and part.function_call:
                function_call = part.function_call
                print(f"[Gemini Tool Call] {function_call.name}")

                # Execute the function
                if function_call.name == "capture_stream_screenshot":
                    result = capture_stream_screenshot()

                    # Add assistant's function call to messages
                    messages.append(response.candidates[0].content)

                    if result["success"]:
                        print(f"[Tool Result] Screenshot captured: {result['file_path']}")

                        # Read and add the image to messages
                        with open(result["file_path"], 'rb') as f:
                            image_bytes = f.read()

                        image_part = types.Part.from_bytes(data=image_bytes, mime_type='image/jpeg')
                        messages.append(types.Content(role="user", parts=[image_part]))
                    else:
                        print(f"[Tool Result] Screenshot failed: {result['error']}")
                        messages.append(types.Content(
                            role="user",
                            parts=[types.Part(text=f"(Screenshot capture failed: {result['error']})")]
                        ))

                    # Get final response
                    final_response = client.models.generate_content(
                        model=GEMINI_MODEL,
                        contents=messages,
                        config=types.GenerateContentConfig(
                            system_instruction=SYSTEM_PROMPT,
                            max_output_tokens=400,
                            temperature=0.7,
                            thinking_config=types.ThinkingConfig(thinking_level="low")
                        )
                    )

                    return final_response.text.strip() if final_response.text else "Screenshot captured!"

    # No function call, return regular text
    return response.text.strip() if response.text else "I couldn't generate a response."


def is_user_on_cooldown(user_id: str) -> bool:
    """Check if user is on cooldown."""
    import time
    now = time.time()
    last = user_last_message.get(user_id, 0)
    return (now - last) < USER_TIMEOUT_SECONDS


def is_user_rate_limited(user_id: str) -> bool:
    """Check if user has exceeded hourly message limit."""
    import time
    now = time.time()
    one_hour_ago = now - 3600

    # Clean old timestamps and count recent ones
    timestamps = user_message_timestamps[user_id]
    user_message_timestamps[user_id] = [ts for ts in timestamps if ts > one_hour_ago]

    return len(user_message_timestamps[user_id]) >= MAX_MESSAGES_PER_HOUR


def update_user_cooldown(user_id: str) -> None:
    """Update user's last message time and add to hourly count."""
    import time
    now = time.time()
    user_last_message[user_id] = now
    user_message_timestamps[user_id].append(now)


async def on_ready(ready_event: EventData) -> None:
    """Called when bot is ready."""
    global bot_username, bot_user_id

    print(f"\n✓ Bot is ready!")

    # Get bot info
    twitch = ready_event.chat.twitch
    users = twitch.get_users()
    async for user in users:
        bot_username = user.login.lower()
        bot_user_id = user.id
        print(f"  Bot username: {user.display_name}")
        print(f"  Listening for @{bot_username} mentions")

    # Join target channels (or bot's own channel if not specified)
    channels_to_join = TARGET_CHANNELS if TARGET_CHANNELS else [bot_username]
    await ready_event.chat.join_room(channels_to_join)
    print(f"  Joined channel(s): {', '.join('#' + ch for ch in channels_to_join)}")


async def on_message(msg: ChatMessage) -> None:
    """Handle incoming chat messages."""
    global bot_username, USE_OLLAMA

    message_text = msg.text.strip()
    user_id = msg.user.id
    user_name = msg.user.display_name
    channel = msg.room.name

    # Ignore our own messages
    if msg.user.name.lower() == bot_username:
        return

    # Handle !vaarabot mode commands
    message_lower = message_text.lower()
    if message_lower == "!vaarabot mode local":
        if USE_OLLAMA:
            await msg.reply("Already using local mode (Ollama)")
            return
        print(f"[Command] {user_name} requested switch to local mode")
        if test_ollama():
            USE_OLLAMA = True
            await msg.reply("Switched to local mode (Ollama) SeemsGood")
            print("[Mode] Switched to Ollama")
        else:
            await msg.reply("Failed to switch to local mode - Ollama test failed NotLikeThis")
        return

    if message_lower == "!vaarabot mode cloud":
        if not USE_OLLAMA:
            await msg.reply("Already using cloud mode (Gemini)")
            return
        print(f"[Command] {user_name} requested switch to cloud mode")
        if test_gemini():
            USE_OLLAMA = False
            await msg.reply("Switched to cloud mode (Gemini) SeemsGood")
            print("[Mode] Switched to Gemini")
        else:
            await msg.reply("Failed to switch to cloud mode - Gemini test failed NotLikeThis")
        return

    # Check if bot is mentioned
    mention_patterns = [f"@{bot_username}"]
    is_mentioned = any(pattern in message_text.lower() for pattern in mention_patterns)

    # Also check if it's a reply to the bot (if reply info available)
    is_reply_to_bot = False
    if hasattr(msg, 'reply') and msg.reply:
        if hasattr(msg.reply, 'parent_user_login'):
            is_reply_to_bot = msg.reply.parent_user_login.lower() == bot_username

    if not is_mentioned and not is_reply_to_bot:
        # Store message but don't respond
        store_message(channel, user_id, user_name, message_text, is_bot=False)
        return

    # Check cooldown
    if is_user_on_cooldown(user_id):
        print(f"[Cooldown] {user_name} is on cooldown")
        store_message(channel, user_id, user_name, message_text, is_bot=False)
        return

    # Check hourly rate limit
    if MAX_MESSAGES_PER_HOUR > 0 and is_user_rate_limited(user_id):
        print(f"[Rate Limited] {user_name} has exceeded {MAX_MESSAGES_PER_HOUR} messages/hour")
        store_message(channel, user_id, user_name, message_text, is_bot=False)
        return

    # Remove the mention from the message for cleaner processing
    clean_message = message_text
    for pattern in mention_patterns:
        clean_message = clean_message.lower().replace(pattern, "").strip()

    if not clean_message:
        clean_message = "Hello!"

    print(f"\n[{channel}] {user_name}: {message_text}")

    # Get response from LLM (channel-scoped context) - BEFORE storing current message
    if USE_OLLAMA:
        response = get_ollama_response(channel, user_id, user_name, clean_message)
    else:
        response = get_gemini_response(channel, user_id, user_name, clean_message)

    # Now store the user's message (after context was built without it)
    store_message(channel, user_id, user_name, message_text, is_bot=False)

    # Truncate if too long for Twitch (500 char limit)
    if len(response) > 480:
        response = response[:477] + "..."

    print(f"[Bot] -> {response}")

    # Send reply
    try:
        await msg.reply(response)
        # Store bot response in database
        store_message(channel, bot_user_id, bot_username, response, is_bot=True)
        update_user_cooldown(user_id)
    except Exception as e:
        print(f"Error sending reply: {e}")
        # Try sending as regular message
        try:
            await msg.chat.send_message(msg.room.name, f"@{user_name} {response}")
            store_message(channel, bot_user_id, bot_username, f"@{user_name} {response}", is_bot=True)
            update_user_cooldown(user_id)
        except Exception as e2:
            print(f"Error sending message: {e2}")


async def run() -> None:
    """Main bot entry point."""
    print("=" * 50)
    print("  Twitch AI Chat Bot")
    print("=" * 50)

    # Validate environment
    missing = []
    if not TWITCH_APP_ID:
        missing.append("TWITCH_APP_ID")
    if not TWITCH_APP_SECRET:
        missing.append("TWITCH_APP_SECRET")
    if not USE_OLLAMA and not GEMINI_API_KEY:
        missing.append("GEMINI_API_KEY")

    if missing:
        print(f"\n✗ Missing environment variables: {', '.join(missing)}")
        print("  Please check your .env file")
        return

    # Test the selected LLM provider
    if USE_OLLAMA:
        if not test_ollama():
            print("\n✗ Cannot start without working Ollama API")
            return
    else:
        if not test_gemini():
            print("\n✗ Cannot start without working Gemini API")
            return

    # Initialize databases and directories
    print("\n📦 Initializing databases...")
    init_database()
    init_chromadb()
    init_screenshots_dir()

    # Initialize Twitch API
    print("\n🔌 Connecting to Twitch...")
    twitch = await Twitch(TWITCH_APP_ID, TWITCH_APP_SECRET)

    # Try to load existing tokens
    tokens = load_tokens()

    if tokens and tokens[0] and tokens[1]:
        print("  Found saved tokens, validating...")
        valid_tokens = await validate_and_refresh_token(twitch, tokens[0], tokens[1])

        if not valid_tokens:
            # Need new auth
            valid_tokens = await do_auth_flow(twitch)
            if not valid_tokens:
                print("\n✗ Authentication failed. Exiting.")
                await twitch.close()
                return
    else:
        # No tokens, do auth flow
        valid_tokens = await do_auth_flow(twitch)
        if not valid_tokens:
            print("\n✗ Authentication failed. Exiting.")
            await twitch.close()
            return

    # Create chat instance
    chat = await Chat(twitch)

    # Register event handlers
    chat.register_event(ChatEvent.READY, on_ready)
    chat.register_event(ChatEvent.MESSAGE, on_message)

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
        print("Bot stopped.")


if __name__ == "__main__":
    asyncio.run(run())
