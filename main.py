"""
Simple Twitch Chat Bot with Gemini AI
Responds when pinged with @botname
"""

import os
import json
import asyncio
import webbrowser
from pathlib import Path
from collections import defaultdict
from dotenv import load_dotenv
from twitchAPI.twitch import Twitch
from twitchAPI.oauth import UserAuthenticator
from twitchAPI.type import AuthScope, ChatEvent
from twitchAPI.chat import Chat, EventData, ChatMessage
from google import genai
from google.genai import types

# Load environment variables
load_dotenv()

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

# Required scopes for chat read/write
USER_SCOPES = [AuthScope.CHAT_READ, AuthScope.CHAT_EDIT, AuthScope.USER_READ_CHAT, AuthScope.USER_WRITE_CHAT]

# Gemini model
GEMINI_MODEL = "gemini-2.5-flash-lite"

# System prompt for brief responses
SYSTEM_PROMPT = """You are Vaarattu's witty Twitch chat bot. You live in Vaarattu's stream chat.

Personality:
- Playful, satirical, and fun - you enjoy banter and clever humor
- Witty and quick with comebacks, but never mean-spirited
- Honest and truthful - give real answers, don't make things up
- Serious when the topic calls for it - read the room
- Loose and casual, like chatting with a clever friend

Rules:
- Keep responses SHORT - under 100 characters when possible, max 400
- Just go straight to the point, do not start explaining things
- If the user's question is too open ended, ask to be more specific and exact
- Users message you as "Username: message" - address them by name when natural
- Use emojis when fitting to add personality, but don't over use them
- Match the user's energy and language
- Detect and respond in the user's language (default: English)
- You're in PUBLIC chat - keep it appropriate, no NSFW or harmful content
- Don't be preachy or lecture people
- Do not markdown bold text
- It's okay to be cheeky, not okay to be offensive"""

# Store conversation history per user
user_histories: dict[str, list[dict]] = defaultdict(list)
# Store last message time per user for cooldown
user_last_message: dict[str, float] = {}
# Store message timestamps per user for hourly rate limit
user_message_timestamps: dict[str, list[float]] = defaultdict(list)

# Bot info (set after login)
bot_username: str = ""
bot_user_id: str = ""


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
                max_output_tokens=10
            )
        )
        print(f"✓ Gemini response: {response.text.strip()}")
        return True
    except Exception as e:
        print(f"✗ Gemini API test failed: {e}")
        return False


def get_gemini_response(user_id: str, user_name: str, message: str) -> str:
    """Get response from Gemini with conversation history."""
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)

        # Build conversation history for context
        history = user_histories[user_id]

        # Create contents with history
        contents = []
        for entry in history:
            contents.append(types.Content(
                role=entry["role"],
                parts=[types.Part(text=entry["text"])]
            ))

        # Add current message
        contents.append(types.Content(
            role="user",
            parts=[types.Part(text=f"{user_name}: {message}")]
        ))

        # Grounding tools for up-to-date information
        google_search_tool = types.Tool(google_search=types.GoogleSearch())
        google_maps_tool = types.Tool(google_maps=types.GoogleMaps())

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                max_output_tokens=150,
                temperature=0.7,
                tools=[google_search_tool, google_maps_tool]
            )
        )

        response_text = response.text.strip() if response.text else "I couldn't generate a response."

        # Update history
        user_histories[user_id].append({"role": "user", "text": f"{user_name}: {message}"})
        user_histories[user_id].append({"role": "model", "text": response_text})

        # Keep only last MAX_HISTORY_PER_USER exchanges (pairs)
        if len(user_histories[user_id]) > MAX_HISTORY_PER_USER * 2:
            user_histories[user_id] = user_histories[user_id][-(MAX_HISTORY_PER_USER * 2):]

        return response_text

    except Exception as e:
        print(f"Gemini error: {e}")
        return "Sorry, I couldn't process that right now."


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
    global bot_username

    # Ignore our own messages
    if msg.user.name.lower() == bot_username:
        return

    message_text = msg.text.strip()
    user_id = msg.user.id
    user_name = msg.user.display_name

    # Check if bot is mentioned
    mention_patterns = [f"@{bot_username}"]
    is_mentioned = any(pattern in message_text.lower() for pattern in mention_patterns)

    # Also check if it's a reply to the bot (if reply info available)
    is_reply_to_bot = False
    if hasattr(msg, 'reply') and msg.reply:
        if hasattr(msg.reply, 'parent_user_login'):
            is_reply_to_bot = msg.reply.parent_user_login.lower() == bot_username

    if not is_mentioned and not is_reply_to_bot:
        # Store message in history anyway for context
        return

    # Check cooldown
    if is_user_on_cooldown(user_id):
        print(f"[Cooldown] {user_name} is on cooldown")
        return

    # Check hourly rate limit
    if MAX_MESSAGES_PER_HOUR > 0 and is_user_rate_limited(user_id):
        print(f"[Rate Limited] {user_name} has exceeded {MAX_MESSAGES_PER_HOUR} messages/hour")
        return

    # Remove the mention from the message for cleaner processing
    clean_message = message_text
    for pattern in mention_patterns:
        clean_message = clean_message.lower().replace(pattern, "").strip()

    if not clean_message:
        clean_message = "Hello!"

    print(f"\n[Chat] {user_name}: {message_text}")

    # Get response from Gemini
    response = get_gemini_response(user_id, user_name, clean_message)

    # Truncate if too long for Twitch (500 char limit)
    if len(response) > 480:
        response = response[:477] + "..."

    print(f"[Bot] -> {response}")

    # Send reply
    try:
        await msg.reply(response)
        update_user_cooldown(user_id)
    except Exception as e:
        print(f"Error sending reply: {e}")
        # Try sending as regular message
        try:
            await msg.chat.send_message(msg.room.name, f"@{user_name} {response}")
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
    if not GEMINI_API_KEY:
        missing.append("GEMINI_API_KEY")

    if missing:
        print(f"\n✗ Missing environment variables: {', '.join(missing)}")
        print("  Please check your .env file")
        return

    # Test Gemini first
    if not test_gemini():
        print("\n✗ Cannot start without working Gemini API")
        return

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
