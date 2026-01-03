"""
Twitch OAuth authentication and token management.
Handles login flow, token storage, and token refresh.
"""

import json
from pathlib import Path
from twitchAPI.twitch import Twitch
from twitchAPI.oauth import UserAuthenticator, refresh_access_token

from config import TOKEN_FILE, USER_SCOPES, TWITCH_APP_ID, TWITCH_APP_SECRET


class TwitchAuth:
    """Manages Twitch OAuth authentication and token lifecycle."""

    def __init__(self):
        self.token_file = TOKEN_FILE

    def save_tokens(self, access_token: str, refresh_token: str) -> None:
        """Save tokens to file."""
        with open(self.token_file, "w") as f:
            json.dump({"access_token": access_token, "refresh_token": refresh_token}, f)

    def load_tokens(self) -> tuple[str, str] | None:
        """Load tokens from file if they exist."""
        if self.token_file.exists():
            try:
                with open(self.token_file) as f:
                    data = json.load(f)
                    return data.get("access_token"), data.get("refresh_token")
            except (json.JSONDecodeError, KeyError):
                return None
        return None

    async def validate_and_refresh_token(self, twitch: Twitch, access_token: str, refresh_token: str) -> tuple[str, str] | None:
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
                new_token, new_refresh = await refresh_access_token(refresh_token, TWITCH_APP_ID, TWITCH_APP_SECRET)
                await twitch.set_user_authentication(new_token, USER_SCOPES, new_refresh)

                # Validate again
                users = twitch.get_users()
                async for u in users:
                    print(f"✓ Token refreshed. Logged in as: {u.display_name} (ID: {u.id})")
                    self.save_tokens(new_token, new_refresh)
                    return new_token, new_refresh
            except Exception as refresh_error:
                print(f"Token refresh failed: {refresh_error}")

        return None

    async def do_auth_flow(self, twitch: Twitch) -> tuple[str, str] | None:
        """Perform OAuth flow with browser and callback server."""
        print("\n🔐 Starting OAuth flow...")
        print("A browser window will open for Twitch login.")

        auth = UserAuthenticator(twitch, USER_SCOPES, force_verify=False)

        try:
            token, refresh_token = await auth.authenticate()
            await twitch.set_user_authentication(token, USER_SCOPES, refresh_token)
            self.save_tokens(token, refresh_token)
            print("✓ Authentication successful!")
            return token, refresh_token
        except Exception as e:
            print(f"✗ Authentication failed: {e}")
            return None

    async def authenticate(self, twitch: Twitch) -> tuple[str, str] | None:
        """Main authentication flow. Try saved tokens first, then do OAuth if needed."""
        # Try to load existing tokens
        tokens = self.load_tokens()

        if tokens and tokens[0] and tokens[1]:
            print("  Found saved tokens, validating...")
            valid_tokens = await self.validate_and_refresh_token(twitch, tokens[0], tokens[1])

            if valid_tokens:
                return valid_tokens

        # Need new auth
        return await self.do_auth_flow(twitch)
