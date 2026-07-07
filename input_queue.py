"""
Unified bot input queue.

Chat mentions and finalized streamer speech both enter this queue. The worker
processes one input at a time, which keeps LLM/tool calls serialized and avoids
the bot talking over itself when chat and stream audio are active together.
"""

import asyncio
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import config


_BRACKETED_CHAT_TOKEN_RE = re.compile(r"\[([A-Za-z][A-Za-z0-9_]{1,31})\]")
_PROTECTED_BRACKET_LABELS = {"BOT", "MOD", "VIP", "STREAMER", "USER", "SOURCE", "ERROR"}


def clean_bracketed_chat_tokens(text: str) -> str:
    """Convert model-written [PogChamp] style artifacts into raw Twitch tokens."""
    def replace_token(match: re.Match) -> str:
        token = match.group(1)
        if token.upper() in _PROTECTED_BRACKET_LABELS:
            return match.group(0)
        return token

    return _BRACKETED_CHAT_TOKEN_RE.sub(replace_token, text)


@dataclass(order=True)
class QueuedInput:
    priority: int
    sequence: int
    created_at: float = field(compare=False)
    source: str = field(compare=False)
    channel: str = field(compare=False)
    user_id: str = field(compare=False)
    user_name: str = field(compare=False)
    message: str = field(compare=False)
    raw_message: str = field(compare=False)
    reply_message: Optional[Any] = field(default=None, compare=False)
    store_user_message: bool = field(default=True, compare=False)
    update_rate_limit: bool = field(default=True, compare=False)


class BotInputQueue:
    """Serializes LLM responses for chat and streamer speech."""

    CHAT_PRIORITY = 0
    STREAMER_PRIORITY = 1

    def __init__(self, database, rate_limiter, llm_provider):
        self.database = database
        self.rate_limiter = rate_limiter
        self.llm_provider = llm_provider

        self._queue = asyncio.PriorityQueue(maxsize=config.INPUT_QUEUE_MAX_SIZE)
        self._task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        self._sequence = 0
        self._chat = None
        self._bot_username = ""
        self._bot_user_id = ""
        self._last_response_time = 0.0
        self._last_streamer_enqueue_time = 0.0
        self._pending_streamer_jobs = 0
        self._idle_event = asyncio.Event()
        self._idle_event.set()

    def set_bot_identity(self, username: str, user_id: str) -> None:
        self._bot_username = username
        self._bot_user_id = user_id

    def set_chat(self, chat) -> None:
        self._chat = chat

    def start(self) -> None:
        if self._task and not self._task.done():
            return

        self._stop_event = asyncio.Event()
        self._task = asyncio.create_task(self._run(), name="bot-input-queue")
        print("[Input Queue] Started")

    async def stop(self) -> None:
        if not self._task:
            return

        self._stop_event.set()
        await self._queue.put(self._make_stop_item())

        try:
            await asyncio.wait_for(self._task, timeout=10)
        except asyncio.TimeoutError:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        print("[Input Queue] Stopped")

    def set_llm_provider(self, llm_provider) -> None:
        self.llm_provider = llm_provider

    @property
    def is_busy(self) -> bool:
        return not self._idle_event.is_set()

    async def wait_until_idle(self) -> None:
        await self._idle_event.wait()

    async def enqueue_chat(self, msg, channel: str, user_id: str, user_name: str, message: str, raw_message: str) -> bool:
        item = self._make_item(
            priority=self.CHAT_PRIORITY,
            source="chat",
            channel=channel,
            user_id=user_id,
            user_name=user_name,
            message=message,
            raw_message=raw_message,
            reply_message=msg,
            store_user_message=True,
            update_rate_limit=False
        )
        return await self._enqueue(item)

    async def enqueue_streamer_speech(self, channel: str, text: str) -> bool:
        if not config.STREAMER_SPEECH_RESPONSES_ENABLED:
            return False

        now = time.monotonic()
        if now - self._last_streamer_enqueue_time < config.STREAMER_SPEECH_RESPONSE_COOLDOWN_SECONDS:
            print("[Input Queue] Dropping streamer speech during cooldown")
            return False

        if self._pending_streamer_jobs >= config.STREAMER_SPEECH_MAX_PENDING:
            print("[Input Queue] Dropping streamer speech because one is already pending")
            return False

        text = text.strip()
        if len(text.split()) < config.STREAMER_SPEECH_MIN_WORDS:
            return False

        item = self._make_item(
            priority=self.STREAMER_PRIORITY,
            source="streamer",
            channel=channel,
            user_id=f"streamer:{channel.lower()}",
            user_name=channel,
            message=text,
            raw_message=text,
            reply_message=None,
            store_user_message=False,
            update_rate_limit=False
        )

        queued = await self._enqueue(item)
        if queued:
            self._last_streamer_enqueue_time = now
            self._pending_streamer_jobs += 1
        return queued

    async def _enqueue(self, item: QueuedInput) -> bool:
        if self._queue.full():
            print(f"[Input Queue] Dropping {item.source} input because queue is full")
            return False

        await self._queue.put(item)
        print(f"[Input Queue] Queued {item.source} input for #{item.channel}: {item.raw_message[:100]}")
        return True

    def _make_item(
        self,
        priority: int,
        source: str,
        channel: str,
        user_id: str,
        user_name: str,
        message: str,
        raw_message: str,
        reply_message,
        store_user_message: bool,
        update_rate_limit: bool
    ) -> QueuedInput:
        self._sequence += 1
        return QueuedInput(
            priority=priority,
            sequence=self._sequence,
            created_at=time.monotonic(),
            source=source,
            channel=channel,
            user_id=user_id,
            user_name=user_name,
            message=message,
            raw_message=raw_message,
            reply_message=reply_message,
            store_user_message=store_user_message,
            update_rate_limit=update_rate_limit
        )

    def _make_stop_item(self) -> QueuedInput:
        return QueuedInput(
            priority=-1,
            sequence=-1,
            created_at=time.monotonic(),
            source="stop",
            channel="",
            user_id="",
            user_name="",
            message="",
            raw_message=""
        )

    async def _run(self) -> None:
        while not self._stop_event.is_set():
            item = await self._queue.get()
            try:
                if item.source == "stop":
                    return

                await self._process_item(item)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                print(f"[Input Queue] Error processing {item.source} input: {exc}")
            finally:
                if item.source == "streamer" and self._pending_streamer_jobs > 0:
                    self._pending_streamer_jobs -= 1
                self._queue.task_done()

    async def _process_item(self, item: QueuedInput) -> None:
        age = time.monotonic() - item.created_at
        if age > config.INPUT_QUEUE_MAX_AGE_SECONDS:
            print(f"[Input Queue] Dropping stale {item.source} input after {age:.1f}s")
            return

        wait_seconds = config.INPUT_QUEUE_MIN_RESPONSE_INTERVAL - (time.monotonic() - self._last_response_time)
        if wait_seconds > 0:
            await asyncio.sleep(wait_seconds)

        print(f"[Input Queue] Processing {item.source} input for #{item.channel}: {item.raw_message[:100]}")
        self._idle_event.clear()

        try:
            game_name = await self._get_game_name(item.channel)
            response = await self.llm_provider.get_response(
                item.channel,
                item.user_id,
                item.user_name,
                item.message,
                self.database,
                game_name=game_name,
                msg_callback=lambda text: self._send_intermediate_message(item, text),
                allow_tools=item.source == "chat"
            )
        finally:
            self._idle_event.set()

        if len(response) > 480:
            response = response[:477] + "..."
        response = clean_bracketed_chat_tokens(response)

        sent_text = await self._send_response(item, response)
        if not sent_text:
            return

        if item.store_user_message:
            self.database.store_message(item.channel, item.user_id, item.user_name, item.raw_message, is_bot=False)

        self.database.store_message(item.channel, self._bot_user_id, self._bot_username, sent_text, is_bot=True)

        if item.update_rate_limit:
            self.rate_limiter.update_user_cooldown(item.user_id)

        self._last_response_time = time.monotonic()

    async def _get_game_name(self, channel: str) -> Optional[str]:
        try:
            from twitch_api import check_stream_status
            stream_status = await check_stream_status(channel)
            if stream_status["is_live"] and stream_status.get("stream_info"):
                game_name = stream_status["stream_info"].get("game_name")
                if game_name:
                    print(f"[Game] Current game: {game_name}")
                    return game_name
        except Exception as exc:
            print(f"[Game] Failed to get game name: {exc}")
        return None

    async def _send_intermediate_message(self, item: QueuedInput, text: str) -> None:
        if item.reply_message:
            try:
                await item.reply_message.reply(text)
                return
            except Exception as exc:
                print(f"[Input Queue] Error sending intermediate reply: {exc}")

        if self._chat:
            try:
                await self._chat.send_message(item.channel, text)
            except Exception as exc:
                print(f"[Input Queue] Error sending intermediate message: {exc}")

    async def _send_response(self, item: QueuedInput, response: str) -> Optional[str]:
        print(f"[Bot] -> {response}")

        if item.reply_message:
            try:
                await item.reply_message.reply(response)
                return response
            except Exception as exc:
                print(f"Error sending reply: {exc}")

            try:
                mention_name = getattr(item.reply_message.user, "display_name", item.user_name)
                text = f"@{mention_name} {response}"
                await item.reply_message.chat.send_message(item.channel, text)
                return text
            except Exception as exc:
                print(f"Error sending message: {exc}")
                return None

        if self._chat:
            try:
                await self._chat.send_message(item.channel, response)
                return response
            except Exception as exc:
                print(f"Error sending queued streamer response: {exc}")

        return None
