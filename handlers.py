"""
Chat event handlers for Twitch bot.
Handles ready events and incoming messages.
"""

from twitchAPI.type import ChatEvent
from twitchAPI.chat import EventData, ChatMessage

from config import TARGET_CHANNELS, MAX_MESSAGES_PER_HOUR


class ChatHandlers:
    """Manages Twitch chat event handlers."""

    def __init__(self, database, rate_limiter, llm_provider):
        self.database = database
        self.rate_limiter = rate_limiter
        self.llm_provider = llm_provider

        # Bot info (set after login)
        self.bot_username: str = ""
        self.bot_user_id: str = ""

    async def on_ready(self, ready_event: EventData) -> None:
        """Called when bot is ready."""
        print(f"\n✓ Bot is ready!")

        # Get bot info
        twitch = ready_event.chat.twitch
        users = twitch.get_users()
        async for user in users:
            self.bot_username = user.login.lower()
            self.bot_user_id = user.id
            print(f"  Bot username: {user.display_name}")
            print(f"  Listening for @{self.bot_username} mentions")

        # Join target channels (or bot's own channel if not specified)
        channels_to_join = TARGET_CHANNELS if TARGET_CHANNELS else [self.bot_username]
        await ready_event.chat.join_room(channels_to_join)
        print(f"  Joined channel(s): {', '.join('#' + ch for ch in channels_to_join)}")

    async def on_message(self, msg: ChatMessage) -> None:
        """Handle incoming chat messages."""
        message_text = msg.text.strip()
        user_id = msg.user.id
        user_name = msg.user.display_name
        channel = msg.room.name

        # Ignore our own messages
        if msg.user.name.lower() == self.bot_username:
            return

        # Handle mode switching commands
        if await self._handle_mode_commands(msg, message_text, user_name):
            return

        # Check if bot is mentioned
        mention_patterns = [f"@{self.bot_username}"]
        is_mentioned = any(pattern in message_text.lower() for pattern in mention_patterns)

        # Also check if it's a reply to the bot
        is_reply_to_bot = False
        if hasattr(msg, 'reply') and msg.reply:
            if hasattr(msg.reply, 'parent_user_login'):
                is_reply_to_bot = msg.reply.parent_user_login.lower() == self.bot_username

        if not is_mentioned and not is_reply_to_bot:
            # Store message but don't respond
            self.database.store_message(channel, user_id, user_name, message_text, is_bot=False)
            return

        # Check cooldown
        if self.rate_limiter.is_user_on_cooldown(user_id):
            print(f"[Cooldown] {user_name} is on cooldown")
            self.database.store_message(channel, user_id, user_name, message_text, is_bot=False)
            return

        # Check hourly rate limit
        if MAX_MESSAGES_PER_HOUR > 0 and self.rate_limiter.is_user_rate_limited(user_id):
            print(f"[Rate Limited] {user_name} has exceeded {MAX_MESSAGES_PER_HOUR} messages/hour")
            self.database.store_message(channel, user_id, user_name, message_text, is_bot=False)
            return

        # Remove the mention from the message
        clean_message = message_text
        for pattern in mention_patterns:
            clean_message = clean_message.lower().replace(pattern, "").strip()

        if not clean_message:
            clean_message = "Hello!"

        print(f"\n[{channel}] {user_name}: {message_text}")

        # Create callback for sending intermediate messages (e.g., ad notifications)
        async def send_chat_message(text: str):
            """Send a message to the chat."""
            try:
                await msg.chat.send_message(msg.room.name, text)
            except Exception as e:
                print(f"[Callback] Error sending message: {e}")

        # Get response from LLM (channel-scoped context) - BEFORE storing current message
        response = await self.llm_provider.get_response(channel, user_id, user_name, clean_message, self.database, msg_callback=send_chat_message)

        # Now store the user's message (after context was built without it)
        self.database.store_message(channel, user_id, user_name, message_text, is_bot=False)

        # Truncate if too long for Twitch (500 char limit)
        if len(response) > 480:
            response = response[:477] + "..."

        print(f"[Bot] -> {response}")

        # Send reply
        try:
            await msg.reply(response)
            # Store bot response in database
            self.database.store_message(channel, self.bot_user_id, self.bot_username, response, is_bot=True)
            self.rate_limiter.update_user_cooldown(user_id)
        except Exception as e:
            print(f"Error sending reply: {e}")
            # Try sending as regular message
            try:
                await msg.chat.send_message(msg.room.name, f"@{user_name} {response}")
                self.database.store_message(channel, self.bot_user_id, self.bot_username, f"@{user_name} {response}", is_bot=True)
                self.rate_limiter.update_user_cooldown(user_id)
            except Exception as e2:
                print(f"Error sending message: {e2}")

    async def _handle_mode_commands(self, msg: ChatMessage, message_text: str, user_name: str) -> bool:
        """Handle !vaarabot mode commands. Returns True if a command was handled."""
        message_lower = message_text.lower()

        if message_lower == "!vaarabot mode local":
            from llm import OllamaLLM
            if isinstance(self.llm_provider, OllamaLLM):
                await msg.reply("Already using local mode (Ollama)")
                return True

            print(f"[Command] {user_name} requested switch to local mode")
            new_llm = OllamaLLM()
            if new_llm.test_connection():
                self.llm_provider = new_llm
                await msg.reply("Switched to local mode (Ollama) SeemsGood")
                print("[Mode] Switched to Ollama")
            else:
                await msg.reply("Failed to switch to local mode - Ollama test failed NotLikeThis")
            return True

        if message_lower == "!vaarabot mode cloud":
            from llm import GeminiLLM
            if isinstance(self.llm_provider, GeminiLLM):
                await msg.reply("Already using cloud mode (Gemini)")
                return True

            print(f"[Command] {user_name} requested switch to cloud mode")
            new_llm = GeminiLLM()
            if new_llm.test_connection():
                self.llm_provider = new_llm
                await msg.reply("Switched to cloud mode (Gemini) SeemsGood")
                print("[Mode] Switched to Gemini")
            else:
                await msg.reply("Failed to switch to cloud mode - Gemini test failed NotLikeThis")
            return True

        return False
