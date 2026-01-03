"""
Ollama LLM implementation with tool support.
"""

from ollama import Client

from config import OLLAMA_MODEL, OLLAMA_HOST, OLLAMA_MODEL_SUPPORTS_VISION, OLLAMA_VISION_MODEL, SYSTEM_PROMPT
from tools import capture_stream_screenshot, OLLAMA_SCREENSHOT_TOOL
from .base import BaseLLM


class OllamaLLM(BaseLLM):
    """Ollama local LLM implementation with RAG context and tool support."""

    def __init__(self):
        self.client = Client(host=OLLAMA_HOST)

    def test_connection(self) -> bool:
        """Test Ollama API with both main model and vision model."""
        # Test main model
        print(f"\n🤖 Testing Ollama API (model: {OLLAMA_MODEL})...")
        try:
            response = self.client.chat(
                model=OLLAMA_MODEL,
                messages=[{'role': 'user', 'content': "Say 'OK' if you can hear me."}]
            )
            print(f"✓ Ollama response: {response.message.content.strip()}")
        except Exception as e:
            print(f"✗ Ollama API test failed: {e}")
            return False

        # Test vision model
        print(f"🤖 Testing Ollama vision model: {OLLAMA_VISION_MODEL}...")
        try:
            response = self.client.chat(
                model=OLLAMA_VISION_MODEL,
                messages=[{'role': 'user', 'content': "Say 'OK' if you can hear me."}]
            )
            print(f"✓ Vision model response: {response.message.content.strip()}")
        except Exception as e:
            print(f"✗ Ollama vision model test failed: {e}")
            return False

        return True

    def _describe_image(self, image_path: str) -> str:
        """Use vision model to describe an image. Returns text description."""
        response = self.client.chat(
            model=OLLAMA_VISION_MODEL,
            messages=[{
                'role': 'user',
                'content': 'Describe this stream screenshot briefly. What game/activity, what is on screen, any visible text.',
                'images': [image_path]
            }]
        )
        return response.message.content.strip()

    def get_response(self, channel: str, user_id: str, user_name: str, message: str, database) -> str:
        """Get response from Ollama with RAG context and tool support."""
        try:
            # Build context from database (channel-scoped)
            context = database.build_context(channel, user_id, user_name, message)

            # Create the prompt with context
            user_prompt = f"{user_name}: {message}"

            if context:
                full_prompt = f"""Here is context from the chat history:

{context}

=== CURRENT MESSAGE ===
{user_prompt}"""
            else:
                full_prompt = user_prompt

            print(f"[Ollama Prompt]\n{full_prompt}\n")

            # First call with tools
            response = self.client.chat(
                model=OLLAMA_MODEL,
                messages=[
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': full_prompt}
                ],
                tools=[OLLAMA_SCREENSHOT_TOOL],
                options={'temperature': 0.7}
            )

            # Check if model wants to call a tool
            if response.message.tool_calls:
                for tool_call in response.message.tool_calls:
                    tool_name = tool_call.function.name
                    print(f"[Ollama Tool Call] {tool_name}")

                    # Handle screenshot tool
                    if tool_name == "capture_stream_screenshot":
                        # Extract channel parameter if provided
                        channel = tool_call.function.arguments.get("channel") if hasattr(tool_call.function, 'arguments') and tool_call.function.arguments else None

                        if channel:
                            print(f"[Ollama] Capturing screenshot from channel: {channel}...")
                            result = capture_stream_screenshot(channel=channel)
                        else:
                            print("[Ollama] Capturing screenshot from default channel...")
                            result = capture_stream_screenshot()

                        if not result["success"]:
                            print(f"[Ollama] Screenshot failed: {result['error']}")
                            error_prompt = full_prompt + f"\n\n(Screenshot capture failed: {result['error']})"
                            second_response = self.client.chat(
                                model=OLLAMA_MODEL,
                                messages=[
                                    {'role': 'system', 'content': SYSTEM_PROMPT},
                                    {'role': 'user', 'content': error_prompt}
                                ],
                                options={'temperature': 0.7}
                            )
                            return second_response.message.content.strip() if second_response.message.content else "Failed to capture screenshot."

                        print(f"[Ollama] Screenshot captured: {result['file_path']}")

                        # Diverging path based on vision support
                        if OLLAMA_MODEL_SUPPORTS_VISION:
                            # Model supports images - send image directly
                            print("[Ollama] Main model supports vision, sending image directly...")
                            second_response = self.client.chat(
                                model=OLLAMA_MODEL,
                                messages=[
                                    {'role': 'system', 'content': SYSTEM_PROMPT},
                                    {'role': 'user', 'content': full_prompt, 'images': [result["file_path"]]}
                                ],
                                options={'temperature': 0.7}
                            )
                        else:
                            # Model doesn't support images - use vision model to describe
                            print("[Ollama] Main model doesn't support vision, using separate vision model...")
                            try:
                                image_description = self._describe_image(result["file_path"])
                                print(f"[Ollama] Image description: {image_description[:100]}...")
                            except Exception as e:
                                print(f"[Ollama] Vision model failed: {e}")
                                image_description = f"(Failed to analyze screenshot: {e})"

                            second_prompt = full_prompt + f"\n\n=== STREAM SCREENSHOT ANALYSIS ===\n{image_description}"
                            second_response = self.client.chat(
                                model=OLLAMA_MODEL,
                                messages=[
                                    {'role': 'system', 'content': SYSTEM_PROMPT},
                                    {'role': 'user', 'content': second_prompt}
                                ],
                                options={'temperature': 0.7}
                            )

                        return second_response.message.content.strip() if second_response.message.content else "Screenshot captured!"

            response_text = response.message.content.strip() if response.message.content else "I couldn't generate a response."
            return response_text

        except Exception as e:
            print(f"Ollama error: {e}")
            return "Sorry, I couldn't process that right now."
