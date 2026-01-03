"""
Gemini LLM implementation with tool support.
"""

from google import genai
from google.genai import types

from config import GEMINI_API_KEY, GEMINI_MODEL, GEMINI_SMALLER_MODEL, SYSTEM_PROMPT, TOOL_DETECTION_PROMPT
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
        """Get response from Gemini with RAG context and two-model tool detection flow."""
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

            print(f"[Gemini Prompt]\n{full_prompt}\n")

            # STEP 1: Use smaller model to detect which tools to use
            tool_results = self._detect_and_execute_tools(full_prompt)

            # STEP 2: Use larger model to generate final response
            final_response = self._generate_final_response(full_prompt, tool_results)

            return final_response

        except Exception as e:
            print(f"Gemini error: {e}")
            return "Sorry, I couldn't process that right now."

    def _detect_and_execute_tools(self, user_prompt: str) -> dict:
        """Use smaller model to detect tools and execute them.

        Returns dict with:
        - screenshot_data: bytes or None
        - search_results: str or None
        """
        result = {
            "screenshot_data": None,
            "search_results": None
        }

        # Define custom tools
        custom_tools = types.Tool(function_declarations=[
            GEMINI_SCREENSHOT_TOOL,
            GEMINI_WEB_SEARCH_TOOL
        ])

        # Build messages for tool detection
        messages = [types.Content(role="user", parts=[types.Part(text=user_prompt)])]

        print(f"[Gemini] Using {GEMINI_SMALLER_MODEL} for tool detection...")

        # Request with smaller model for tool detection
        response = self.client.models.generate_content(
            model=GEMINI_SMALLER_MODEL,
            contents=messages,
            config=types.GenerateContentConfig(
                system_instruction=TOOL_DETECTION_PROMPT,
                max_output_tokens=200,
                temperature=0.3,  # Lower temperature for more deterministic tool selection
                tools=[custom_tools],
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                thinking_config=types.ThinkingConfig(thinking_level="low")
            )
        )

        # Check if there are function calls in the response
        if (response.candidates and
            response.candidates[0].content and
            response.candidates[0].content.parts):

            for part in response.candidates[0].content.parts:
                if hasattr(part, 'function_call') and part.function_call:
                    function_call = part.function_call
                    print(f"[Gemini Tool Call] {function_call.name}")

                    # Execute the function based on name
                    if function_call.name == "capture_stream_screenshot":
                        screenshot_result = capture_stream_screenshot()

                        if screenshot_result["success"]:
                            print(f"[Tool Result] Screenshot captured: {screenshot_result['file_path']}")
                            # Read the image
                            with open(screenshot_result["file_path"], 'rb') as f:
                                result["screenshot_data"] = f.read()
                        else:
                            print(f"[Tool Result] Screenshot failed: {screenshot_result['error']}")

                    elif function_call.name == "web_search":
                        # Extract parameters
                        query = function_call.args.get("query", "")

                        print(f"[Gemini] Searching web for: {query}")
                        search_result = perform_web_search(query)

                        if search_result["success"]:
                            # Format search results as text
                            if search_result["results"]:
                                results_text = f"Search results for '{query}':\n\n"
                                for i, item in enumerate(search_result["results"], 1):
                                    results_text += f"{i}. {item['title']}\n"
                                    results_text += f"   {item['snippet']}\n"
                                result["search_results"] = results_text
                                print(f"[Tool Result] Found {len(search_result['results'])} results")
                            else:
                                result["search_results"] = f"No search results found for '{query}'."
                                print(f"[Tool Result] No results found")
                        else:
                            print(f"[Tool Result] Search failed: {search_result['error']}")

        return result

    def _generate_final_response(self, user_prompt: str, tool_results: dict) -> str:
        """Use larger model to generate final response with tool results.

        Args:
            user_prompt: The original user prompt with context
            tool_results: Dict containing screenshot_data and search_results
        """
        print(f"[Gemini] Using {GEMINI_MODEL} for final response generation...")

        # Build messages for final response
        parts = [types.Part(text=user_prompt)]

        # Add screenshot if available
        if tool_results["screenshot_data"]:
            print("[Gemini] Adding screenshot to final response context")
            image_part = types.Part.from_bytes(
                data=tool_results["screenshot_data"],
                mime_type='image/jpeg'
            )
            parts.append(image_part)

        # Add search results if available
        if tool_results["search_results"]:
            print("[Gemini] Adding search results to final response context")
            parts.append(types.Part(text=f"\n\n{tool_results['search_results']}"))

        messages = [types.Content(role="user", parts=parts)]

        # Generate final response with larger model
        response = self.client.models.generate_content(
            model=GEMINI_MODEL,
            contents=messages,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                max_output_tokens=400,
                temperature=0.7,
                thinking_config=types.ThinkingConfig(thinking_level="low")
            )
        )

        return response.text.strip() if response.text else "I couldn't generate a response."
