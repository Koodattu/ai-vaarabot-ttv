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
        self._external_twitch: Optional[Twitch] = None  # For using authenticated instance from main bot

    def set_authenticated_client(self, twitch: Twitch) -> None:
        """Set an external authenticated Twitch client for user-level operations.

        Args:
            twitch: Already authenticated Twitch instance from main bot
        """
        self._external_twitch = twitch
        print("✓ TwitchAPIClient now using authenticated Twitch instance for user operations")

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

    async def ban_user(self, broadcaster_id: str, moderator_id: str, user_login: str, duration: int = 1, reason: str = "Timeout by AI bot") -> Dict[str, any]:
        """Ban or timeout a user in a channel.

        Args:
            broadcaster_id: The ID of the broadcaster whose chat the user is being banned from
            moderator_id: The ID of the moderator issuing the ban (must match auth token)
            user_login: The login name of the user to ban
            duration: Duration in seconds (1-1209600). Defaults to 1 second.
            reason: Reason for the ban (max 500 chars)

        Returns:
            Dict with:
                - success (bool): True if ban successful
                - error (str): Error message if failed, None otherwise
                - user_name (str): Display name of banned user
                - ends_at (str): When the timeout ends (ISO format)
        """
        # Use external authenticated client if available, otherwise use internal client
        client = self._external_twitch if self._external_twitch else self.twitch

        if not client:
            # Ensure client is initialized
            if not self._initialized:
                init_success = await self.initialize()
                if not init_success:
                    return {
                        "success": False,
                        "error": "Failed to initialize Twitch API client",
                        "user_name": None,
                        "ends_at": None
                    }
                client = self.twitch

        try:
            # First, get the user ID from the login name
            users = []
            async for user in client.get_users(logins=[user_login.lower()]):
                users.append(user)

            if not users:
                return {
                    "success": False,
                    "error": f"User '{user_login}' not found",
                    "user_name": None,
                    "ends_at": None
                }

            user = users[0]
            user_id = user.id
            user_name = user.display_name

            # Validate duration (1 second to 2 weeks)
            duration = max(1, min(1209600, duration))

            # Ban the user with the specified duration
            result = await client.ban_user(
                broadcaster_id=broadcaster_id,
                moderator_id=moderator_id,
                user_id=user_id,
                duration=duration,
                reason=reason
            )

            print(f"✓ Banned user '{user_name}' for {duration} second(s)")

            return {
                "success": True,
                "error": None,
                "user_name": user_name,
                "ends_at": result.end_time.isoformat() if result.end_time else None
            }

        except Exception as e:
            print(f"✗ Error banning user '{user_login}': {e}")
            return {
                "success": False,
                "error": f"Ban failed: {str(e)}",
                "user_name": None,
                "ends_at": None
            }

    async def get_user_info(self, user_login: str) -> Dict[str, any]:
        """Get information about a Twitch user by their username.

        Args:
            user_login: The login name of the user to look up

        Returns:
            Dict with:
                - success (bool): True if user found
                - error (str): Error message if failed, None otherwise
                - user_data (dict): User information including:
                    - id: User ID
                    - login: Login name
                    - display_name: Display name
                    - description: Bio/description
                    - profile_image_url: Profile picture URL
                    - created_at: Account creation date (ISO format)
                    - broadcaster_type: Type (partner, affiliate, or empty)
                    - view_count: Total channel views
        """
        # Use external authenticated client if available, otherwise use internal client
        client = self._external_twitch if self._external_twitch else self.twitch

        if not client:
            # Ensure client is initialized
            if not self._initialized:
                init_success = await self.initialize()
                if not init_success:
                    return {
                        "success": False,
                        "error": "Failed to initialize Twitch API client",
                        "user_data": None
                    }
                client = self.twitch

        try:
            # Get user by login name
            users = []
            async for user in client.get_users(logins=[user_login.lower()]):
                users.append(user)

            if not users:
                return {
                    "success": False,
                    "error": f"User '{user_login}' not found",
                    "user_data": None
                }

            user = users[0]

            # Build user data dict
            user_data = {
                "id": user.id,
                "login": user.login,
                "display_name": user.display_name,
                "description": user.description or "No bio",
                "profile_image_url": user.profile_image_url,
                "created_at": user.created_at.isoformat() if user.created_at else None,
                "broadcaster_type": user.broadcaster_type or "normal user",
                "view_count": user.view_count
            }

            print(f"✓ Fetched info for user: {user.display_name}")

            return {
                "success": True,
                "error": None,
                "user_data": user_data
            }

        except Exception as e:
            print(f"✗ Error fetching user info for '{user_login}': {e}")
            return {
                "success": False,
                "error": f"Failed to fetch user info: {str(e)}",
                "user_data": None
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
