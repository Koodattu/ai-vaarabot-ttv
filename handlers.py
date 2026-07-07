"""
Chat event handlers for Twitch bot.
Handles ready events and incoming messages.
"""

from twitchAPI.type import ChatEvent
from twitchAPI.chat import EventData, ChatMessage

from config import TARGET_CHANNELS, MAX_MESSAGES_PER_HOUR


class ChatHandlers:
    """Manages Twitch chat event handlers."""

    def __init__(self, database, rate_limiter, llm_provider, input_queue, transcriber=None):
        self.database = database
        self.rate_limiter = rate_limiter
        self.llm_provider = llm_provider
        self.input_queue = input_queue
        self.transcriber = transcriber

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
            self.input_queue.set_bot_identity(self.bot_username, self.bot_user_id)
            self.input_queue.set_chat(ready_event.chat)
            self.input_queue.start()
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
        is_vip = msg.user.vip
        is_mod = msg.user.mod
        badges = msg.user.badges

        print(badges)

        # Check for subscriber badge and duration
        # badges is a dict like {'subscriber': '3066', 'broadcaster': '1', 'premium': '1'}
        # Subscriber badge value format: TXXX where T=tier (1-3) and XXX=months
        sub_info = None
        if badges and isinstance(badges, dict) and "subscriber" in badges:
            try:
                badge_value = int(badges["subscriber"])
                if badge_value >= 1000:
                    # Extract tier (first digit) and months (remaining digits)
                    tier = badge_value // 1000
                    months = badge_value % 1000
                    sub_info = (tier, months)
                else:
                    # Regular badge value (just months, assume tier 1)
                    sub_info = (1, badge_value)
            except (ValueError, TypeError):
                pass

        # Format username with badge for storage (priority: MOD > VIP > SUB)
        formatted_user_name = user_name
        if is_mod:
            formatted_user_name = f"[MOD] {user_name}"
        elif is_vip:
            formatted_user_name = f"[VIP] {user_name}"
        elif sub_info:
            tier, months = sub_info
            formatted_user_name = f"[T{tier}-{months}M] {user_name}"

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
            self.database.store_message(channel, user_id, formatted_user_name, message_text, is_bot=False)
            return

        # Check cooldown
        if self.rate_limiter.is_user_on_cooldown(user_id):
            print(f"[Cooldown] {user_name} is on cooldown")
            self.database.store_message(channel, user_id, formatted_user_name, message_text, is_bot=False)
            return

        # Check hourly rate limit
        if MAX_MESSAGES_PER_HOUR > 0 and self.rate_limiter.is_user_rate_limited(user_id):
            minutes_left = self.rate_limiter.get_time_until_rate_limit_reset(user_id)
            print(f"[Rate Limited] {user_name} has exceeded {MAX_MESSAGES_PER_HOUR} messages/hour")
            self.database.store_message(channel, user_id, formatted_user_name, message_text, is_bot=False)
            # Send rate limit notification
            try:
                rate_limit_msg = f"You've reached the message limit ({MAX_MESSAGES_PER_HOUR}/hour). Please wait {minutes_left} minute{'s' if minutes_left != 1 else ''} before messaging again."
                await msg.reply(rate_limit_msg)
            except Exception as e:
                print(f"Error sending rate limit message: {e}")
            return

        # Remove the mention from the message
        clean_message = message_text
        for pattern in mention_patterns:
            clean_message = clean_message.lower().replace(pattern, "").strip()

        if not clean_message:
            clean_message = "Hello!"

        print(f"\n[{channel}] {formatted_user_name}: {message_text}")

        queued = await self.input_queue.enqueue_chat(
            msg=msg,
            channel=channel,
            user_id=user_id,
            user_name=formatted_user_name,
            message=clean_message,
            raw_message=message_text
        )
        if not queued:
            self.database.store_message(channel, user_id, formatted_user_name, message_text, is_bot=False)
            try:
                await msg.reply("I'm a little backed up rn NotLikeThis")
            except Exception as e:
                print(f"Error sending queue full message: {e}")
            return

        self.rate_limiter.update_user_cooldown(user_id)

    async def _handle_mode_commands(self, msg: ChatMessage, message_text: str, user_name: str) -> bool:
        """Handle !vaarabot mode commands. Returns True if a command was handled."""
        message_lower = message_text.lower()

        if message_lower == "!vaarabot transcribe" or message_lower.startswith("!vaarabot transcribe "):
            await self._handle_transcription_command(msg, message_lower)
            return True

        if message_lower == "!vaarabot mode local":
            from llm import OllamaLLM
            if isinstance(self.llm_provider, OllamaLLM):
                await msg.reply("Already using local mode (Ollama)")
                return True

            print(f"[Command] {user_name} requested switch to local mode")
            new_llm = OllamaLLM()
            if new_llm.test_connection():
                self.llm_provider = new_llm
                self.input_queue.set_llm_provider(new_llm)
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
                self.input_queue.set_llm_provider(new_llm)
                await msg.reply("Switched to cloud mode (Gemini) SeemsGood")
                print("[Mode] Switched to Gemini")
            else:
                await msg.reply("Failed to switch to cloud mode - Gemini test failed NotLikeThis")
            return True

        return False

    async def _handle_transcription_command(self, msg: ChatMessage, message_lower: str) -> None:
        """Handle stream audio transcription controls."""
        if not self.transcriber:
            await msg.reply("Stream audio transcription is not available")
            return

        parts = message_lower.split()
        action = parts[2] if len(parts) >= 3 else "status"

        if action == "status":
            await msg.reply(self.transcriber.status())
            return

        if action in ("on", "start"):
            if not self._can_manage_transcription(msg):
                await msg.reply("Mods only for transcription controls")
                return

            success, reply = await self.transcriber.start()
            if not success:
                reply = f"{reply} NotLikeThis"
            await msg.reply(reply)
            return

        if action in ("off", "stop"):
            if not self._can_manage_transcription(msg):
                await msg.reply("Mods only for transcription controls")
                return

            success, reply = await self.transcriber.stop()
            await msg.reply(reply)
            return

        await msg.reply("Use: !vaarabot transcribe on/off/status")

    def _can_manage_transcription(self, msg: ChatMessage) -> bool:
        """Allow broadcaster and moderators to start or stop transcription."""
        badges = msg.user.badges if isinstance(msg.user.badges, dict) else {}
        room_name = getattr(msg.room, "name", "").lower()
        user_name = getattr(msg.user, "name", "").lower()
        is_broadcaster = bool(badges.get("broadcaster")) or (room_name and user_name == room_name)
        return bool(msg.user.mod or is_broadcaster)
