"""
Base class and utilities for LLM implementations.
"""

from abc import ABC, abstractmethod


class BaseLLM(ABC):
    """Abstract base class for LLM implementations."""

    @abstractmethod
    async def get_response(self, channel: str, user_id: str, user_name: str, message: str, database, game_name: str = None, msg_callback=None) -> str:
        """Get response from the LLM with context from database.

        Args:
            channel: Twitch channel name
            user_id: User's Twitch ID
            user_name: User's display name
            message: User's message
            database: Database instance for context retrieval
            game_name: Current game being played on Twitch (if available)
            msg_callback: Optional async callback to send intermediate messages (e.g., for ad notifications)

        Returns:
            str: The LLM's response
        """
        pass

    @abstractmethod
    def test_connection(self) -> bool:
        """Test if the LLM API is accessible and working.

        Returns:
            bool: True if test successful, False otherwise
        """
        pass

    @abstractmethod
    def get_simple_response(self, prompt: str) -> str:
        """Get a simple response without context or tools.

        Args:
            prompt: Simple prompt string

        Returns:
            str: The LLM's response
        """
        pass
