"""
Gemini LLM implementation with tool support.
"""

from google import genai
from google.genai import types

from config import GEMINI_API_KEY, GEMINI_MODEL, SYSTEM_PROMPT
from tools import capture_stream_screenshot, perform_web_search, GEMINI_SCREENSHOT_TOOL, GEMINI_WEB_SEARCH_TOOL
from .base import BaseLLM


class GeminiLLM(BaseLLM):
    """Gemini AI implementation with RAG context and tool support."""

    def __init__(self):
        self.client = genai.Client(api_key=GEMINI_API_KEY)

    def test_connection(self) -> bool:
        """Test Gemini API with a simple prompt."""
        print("\n🤖 Testing Gemini API...")
        try:
            response = self.client.models.generate_content(
                model=GEMINI_MODEL,
                contents="Say 'OK' if you can hear me.",
                config=types.GenerateContentConfig(
                    max_output_tokens=100,
                    thinking_config=types.ThinkingConfig(thinking_level="low")
                )
            )
            print(f"✓ Gemini response: {response.text.strip()}")
            return True
        except Exception as e:
            print(f"✗ Gemini API test failed: {e}")
            return False

    def get_response(self, channel: str, user_id: str, user_name: str, message: str, database) -> str:
        """Get response from Gemini with RAG context and tool support."""
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

            # Define custom tools
            custom_tools = types.Tool(function_declarations=[
                GEMINI_SCREENSHOT_TOOL,
                GEMINI_WEB_SEARCH_TOOL
            ])

            print(f"[Gemini Prompt]\n{full_prompt}\n")

            # Build conversation messages
            messages = [types.Content(role="user", parts=[types.Part(text=full_prompt)])]

            # Initial request with all tools
            response = self.client.models.generate_content(
                model=GEMINI_MODEL,
                contents=messages,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    max_output_tokens=400,
                    temperature=0.7,
                    tools=[custom_tools],
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                    thinking_config=types.ThinkingConfig(thinking_level="low")
                )
            )

            # Handle response including function calls
            response_text = self._handle_response(messages, response)
            return response_text

        except Exception as e:
            print(f"Gemini error: {e}")
            return "Sorry, I couldn't process that right now."

    def _handle_response(self, messages: list, response) -> str:
        """Handle Gemini response, including function calls."""
        # Check if there's a function call in the response
        if (response.candidates and
            response.candidates[0].content and
            response.candidates[0].content.parts):

            for part in response.candidates[0].content.parts:
                if hasattr(part, 'function_call') and part.function_call:
                    function_call = part.function_call
                    print(f"[Gemini Tool Call] {function_call.name}")

                    # Add assistant's function call to messages
                    messages.append(response.candidates[0].content)

                    # Execute the function based on name
                    if function_call.name == "capture_stream_screenshot":
                        result = capture_stream_screenshot()

                        if result["success"]:
                            print(f"[Tool Result] Screenshot captured: {result['file_path']}")

                            # Read and add the image to messages
                            with open(result["file_path"], 'rb') as f:
                                image_bytes = f.read()

                            image_part = types.Part.from_bytes(data=image_bytes, mime_type='image/jpeg')
                            messages.append(types.Content(role="user", parts=[image_part]))
                        else:
                            print(f"[Tool Result] Screenshot failed: {result['error']}")
                            messages.append(types.Content(
                                role="user",
                                parts=[types.Part(text=f"(Screenshot capture failed: {result['error']})")]
                            ))

                    elif function_call.name == "web_search":
                        # Extract parameters
                        query = function_call.args.get("query", "")

                        print(f"[Gemini] Searching web for: {query}")
                        result = perform_web_search(query)

                        if result["success"]:
                            # Format search results as text
                            if result["results"]:
                                results_text = f"Search results for '{query}':\n\n"
                                for i, item in enumerate(result["results"], 1):
                                    results_text += f"{i}. {item['title']}\n"
                                    results_text += f"   {item['snippet']}\n"
                                print(f"[Tool Result] Found {len(result['results'])} results")
                            else:
                                results_text = f"No search results found for '{query}'."
                                print(f"[Tool Result] No results found")

                            # Add search results to messages
                            messages.append(types.Content(
                                role="user",
                                parts=[types.Part(text=results_text)]
                            ))
                        else:
                            print(f"[Tool Result] Search failed: {result['error']}")
                            messages.append(types.Content(
                                role="user",
                                parts=[types.Part(text=f"(Web search failed: {result['error']})")]
                            ))

                    # Get final response after tool execution
                    final_response = self.client.models.generate_content(
                        model=GEMINI_MODEL,
                        contents=messages,
                        config=types.GenerateContentConfig(
                            system_instruction=SYSTEM_PROMPT,
                            max_output_tokens=400,
                            temperature=0.7,
                            thinking_config=types.ThinkingConfig(thinking_level="low")
                        )
                    )

                    return final_response.text.strip() if final_response.text else "Done!"

        # No function call, return regular text
        return response.text.strip() if response.text else "I couldn't generate a response."
