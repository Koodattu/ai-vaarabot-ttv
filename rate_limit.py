"""
Rate limiting and cooldown management for chat responses.
Prevents spam and enforces per-user message limits.
"""

import time
from collections import defaultdict

from config import USER_TIMEOUT_SECONDS, MAX_MESSAGES_PER_HOUR


class RateLimiter:
    """Manages user cooldowns and hourly rate limits."""

    def __init__(self):
        # Store last message time per user for cooldown
        self.user_last_message: dict[str, float] = {}
        # Store message timestamps per user for hourly rate limit
        self.user_message_timestamps: dict[str, list[float]] = defaultdict(list)

    def is_user_on_cooldown(self, user_id: str) -> bool:
        """Check if user is on cooldown."""
        now = time.time()
        last = self.user_last_message.get(user_id, 0)
        return (now - last) < USER_TIMEOUT_SECONDS

    def is_user_rate_limited(self, user_id: str) -> bool:
        """Check if user has exceeded hourly message limit."""
        if MAX_MESSAGES_PER_HOUR <= 0:
            return False

        now = time.time()
        one_hour_ago = now - 3600

        # Clean old timestamps and count recent ones
        timestamps = self.user_message_timestamps[user_id]
        self.user_message_timestamps[user_id] = [ts for ts in timestamps if ts > one_hour_ago]

        return len(self.user_message_timestamps[user_id]) >= MAX_MESSAGES_PER_HOUR

    def update_user_cooldown(self, user_id: str) -> None:
        """Update user's last message time and add to hourly count."""
        now = time.time()
        self.user_last_message[user_id] = now
        self.user_message_timestamps[user_id].append(now)
