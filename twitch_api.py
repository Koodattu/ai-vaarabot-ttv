"""
Twitch API client for checking stream status and other Twitch operations.
Uses the TwitchAPI library for authentication and API calls.
"""

from twitchAPI.twitch import Twitch
from twitchAPI.type import AuthScope
from twitchAPI.oauth import UserAuthenticator
from typing import Optional, Dict
import asyncio

from config import TWITCH_APP_ID, TWITCH_APP_SECRET


class TwitchAPIClient:
    """Client for interacting with Twitch API."""

    def __init__(self):
        """Initialize the Twitch API client."""
        self.twitch: Optional[Twitch] = None
        self._initialized = False

    async def initialize(self) -> bool:
        """Initialize and authenticate with Twitch API.

        Returns:
            True if initialization successful, False otherwise
        """
        if self._initialized and self.twitch:
            return True

        try:
            # Initialize Twitch API client with app credentials
            self.twitch = await Twitch(TWITCH_APP_ID, TWITCH_APP_SECRET)
            self._initialized = True
            print("✓ Twitch API client initialized")
            return True
        except Exception as e:
            print(f"✗ Failed to initialize Twitch API: {e}")
            return False

    async def is_stream_live(self, channel_name: str) -> Dict[str, any]:
        """Check if a Twitch stream is currently live.

        Args:
            channel_name: The Twitch channel name (username) to check

        Returns:
            Dict with:
                - is_live (bool): True if stream is live, False otherwise
                - error (str): Error message if check failed, None otherwise
                - stream_info (dict): Stream information if live (title, viewer_count, game_name, etc.)
        """
        # Ensure client is initialized
        if not self._initialized:
            init_success = await self.initialize()
            if not init_success:
                return {
                    "is_live": False,
                    "error": "Failed to initialize Twitch API client",
                    "stream_info": None
                }

        try:
            # Get user information first to get the user ID
            # Note: get_users returns an async generator, so we need to iterate over it
            users = []
            async for user in self.twitch.get_users(logins=[channel_name.lower()]):
                users.append(user)

            if not users or len(users) == 0:
                return {
                    "is_live": False,
                    "error": f"Channel '{channel_name}' not found",
                    "stream_info": None
                }

            user = users[0]
            user_id = user.id

            # Check if stream is live
            # Note: get_streams also returns an async generator
            streams = []
            async for stream in self.twitch.get_streams(user_id=[user_id]):
                streams.append(stream)

            if not streams or len(streams) == 0:
                # Stream is offline
                return {
                    "is_live": False,
                    "error": None,
                    "stream_info": None
                }

            # Stream is live, extract info
            stream = streams[0]
            stream_info = {
                "title": stream.title,
                "game_name": stream.game_name,
                "viewer_count": stream.viewer_count,
                "started_at": stream.started_at.isoformat() if stream.started_at else None,
                "language": stream.language,
                "thumbnail_url": stream.thumbnail_url
            }

            print(f"✓ Stream '{channel_name}' is LIVE: {stream.viewer_count} viewers watching {stream.game_name}")

            return {
                "is_live": True,
                "error": None,
                "stream_info": stream_info
            }

        except Exception as e:
            print(f"✗ Error checking stream status for '{channel_name}': {e}")
            return {
                "is_live": False,
                "error": f"API error: {str(e)}",
                "stream_info": None
            }

    async def close(self):
        """Close the Twitch API client and cleanup resources."""
        if self.twitch:
            await self.twitch.close()
            self._initialized = False
            print("✓ Twitch API client closed")


# Singleton instance for reuse
_twitch_client: Optional[TwitchAPIClient] = None


def get_twitch_client() -> TwitchAPIClient:
    """Get or create the singleton Twitch API client instance.

    Returns:
        TwitchAPIClient instance
    """
    global _twitch_client
    if _twitch_client is None:
        _twitch_client = TwitchAPIClient()
    return _twitch_client


async def check_stream_status(channel_name: str) -> Dict[str, any]:
    """Convenience function to check if a stream is live.

    Args:
        channel_name: The Twitch channel name to check

    Returns:
        Dict with is_live, error, and stream_info fields
    """
    client = get_twitch_client()
    return await client.is_stream_live(channel_name)
