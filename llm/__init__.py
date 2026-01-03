"""
LLM module initialization.
Provides unified interface for both Gemini and Ollama.
"""

from .gemini import GeminiLLM
from .ollama import OllamaLLM

__all__ = ['GeminiLLM', 'OllamaLLM']
