"""
Gemini LLM implementation with tool support.
"""

from google import genai
from google.genai import types

from config import GEMINI_API_KEY, GEMINI_MODEL, GEMINI_SMALLER_MODEL, SYSTEM_PROMPT, TOOL_DETECTION_PROMPT, WEBSITE_SELECTION_PROMPT, CONTENT_EXTRACTION_PROMPT
from tools import capture_stream_screenshot, perform_web_search, scrape_website, GEMINI_SCREENSHOT_TOOL, GEMINI_WEB_SEARCH_TOOL
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

    async def get_response(self, channel: str, user_id: str, user_name: str, message: str, database, game_name: str = None, msg_callback=None) -> str:
        """Get response from Gemini with RAG context and two-model tool detection flow."""
        try:
            # Build context from database (channel-scoped)
            context = database.build_context(channel, user_id, user_name, message, game_name=game_name)

            # Create the prompt with context
            user_prompt = f"{user_name}: {message}"

            # Build context message separately from current message
            context_message = ""
            if context:
                context_message = f"""Here is context from the chat history:

{context}"""

            print(f"[Gemini Prompt]\n{context_message}\n\n=== CURRENT MESSAGE ===\n{user_prompt}\n")

            # STEP 1: Use smaller model to detect which tools to use
            # For tool detection, combine context and message
            full_prompt = f"{context_message}\n\n=== CURRENT MESSAGE ===\n{user_prompt}" if context_message else user_prompt
            tool_results = await self._detect_and_execute_tools(full_prompt, user_prompt, channel, message, msg_callback)

            # STEP 2: Use larger model to generate final response with restructured messages
            final_response = self._generate_final_response(context_message, user_prompt, tool_results)

            return final_response

        except Exception as e:
            print(f"Gemini error: {e}")
            return "Sorry, I couldn't process that right now."

    async def _detect_and_execute_tools(self, full_prompt: str, user_prompt: str, channel: str, user_message: str, msg_callback=None) -> dict:
        """Use smaller model to detect tools and execute them.

        Args:
            full_prompt: The complete prompt with chat history context
            user_prompt: Just the user's message (for website selection context)
            channel: Twitch channel name for screenshot capture
            user_message: The user's original message for ad notification context
            msg_callback: Optional async callback to send intermediate messages

        Returns dict with:
        - screenshot_data: bytes or None
        - search_results: str or None
        - screenshot_error: str or None (error message if screenshot failed)
        """
        result = {
            "screenshot_data": None,
            "search_results": None,
            "screenshot_error": None
        }

        # Define custom tools
        custom_tools = types.Tool(function_declarations=[
            GEMINI_SCREENSHOT_TOOL,
            GEMINI_WEB_SEARCH_TOOL
        ])

        # Build messages for tool detection
        messages = [types.Content(role="user", parts=[types.Part(text=full_prompt)])]

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
                #thinking_config=types.ThinkingConfig(thinking_level="low")
            )
        )

        # First pass: collect all tool calls to check if both screenshot and web_search are requested
        tool_calls = []
        if (response.candidates and
            response.candidates[0].content and
            response.candidates[0].content.parts):

            for part in response.candidates[0].content.parts:
                if hasattr(part, 'function_call') and part.function_call:
                    tool_calls.append(part.function_call)

        # Check if BOTH screenshot and web_search are requested
        has_screenshot = any(tc.name == "capture_stream_screenshot" for tc in tool_calls)
        has_web_search = any(tc.name == "web_search" for tc in tool_calls)
        both_tools_requested = has_screenshot and has_web_search

        if both_tools_requested:
            print("[Gemini] Both screenshot and web_search requested - using enriched query flow")

            # Step 1: Capture screenshot first
            screenshot_call = next(tc for tc in tool_calls if tc.name == "capture_stream_screenshot")
            target_channel = screenshot_call.args.get("channel") if screenshot_call.args else channel

            print(f"[Gemini] Capturing screenshot from channel: {target_channel}")
            from tools import capture_screenshot_with_ad_detection
            from config import AD_DETECTION_ENABLED, AD_WAIT_PROMPT

            if AD_DETECTION_ENABLED:
                screenshot_result = await capture_screenshot_with_ad_detection(
                    channel=target_channel,
                    llm_provider=self,
                    simple_prompt=AD_WAIT_PROMPT,
                    user_message=user_message,
                    msg_callback=msg_callback
                )
            else:
                screenshot_result = await capture_stream_screenshot(channel=target_channel)

            if screenshot_result["success"]:
                print(f"[Tool Result] Screenshot captured: {screenshot_result['file_path']}")
                # Read the screenshot
                with open(screenshot_result["file_path"], 'rb') as f:
                    screenshot_data = f.read()
                result["screenshot_data"] = screenshot_data
                # Store stream info (including game name)
                result["stream_info"] = screenshot_result.get("stream_info")

                # Step 2: Generate enriched query using screenshot
                enriched_query = self._generate_enriched_query(user_message, screenshot_data)

                if enriched_query:
                    # Step 3: Perform web search with enriched query (discard original query)
                    print(f"[Gemini] Searching web with enriched query: {enriched_query}")

                    # Send notification to chat if enabled
                    from config import WEB_SEARCH_NOTIFICATION
                    if WEB_SEARCH_NOTIFICATION and msg_callback:
                        try:
                            await msg_callback(f"searching web for: {enriched_query}")
                        except Exception as e:
                            print(f"[Web Search] Failed to send notification: {e}")

                    search_result = perform_web_search(enriched_query, num_results=10)

                    if search_result["success"] and search_result["results"]:
                        results_list = []
                        for i, item in enumerate(search_result["results"], 1):
                            results_list.append({
                                "number": i,
                                "title": item["title"],
                                "snippet": item["snippet"],
                                "link": item["link"]
                            })

                        print(f"[Tool Result] Found {len(results_list)} results")

                        # Select, scrape, and extract from multiple websites
                        extracted_content = self._select_scrape_and_extract(user_message, results_list)

                        if extracted_content:
                            result["search_results"] = f"""Web search for '{enriched_query}':\n\n{extracted_content}"""
                            print(f"[Tool Result] Successfully extracted and compacted content")
                        else:
                            # Fallback to search snippets
                            results_text = f"Web search results for '{enriched_query}':\n\n"
                            for item in results_list:
                                results_text += f"{item['number']}. {item['title']}\n"
                                results_text += f"   {item['snippet']}\n\n"
                            result["search_results"] = results_text
                            print(f"[Tool Result] Extraction failed, using search snippets")
                    else:
                        if search_result["success"]:
                            result["search_results"] = f"No search results found for '{enriched_query}'."
                            print(f"[Tool Result] No results found")
                        else:
                            print(f"[Tool Result] Search failed: {search_result['error']}")
                else:
                    print("[Gemini] Failed to generate enriched query, skipping web search")
            else:
                print(f"[Tool Result] Screenshot failed: {screenshot_result['error']}")
                result["screenshot_error"] = screenshot_result["error"]
                if screenshot_result.get("is_live") is False:
                    result["screenshot_error"] = f"The stream is currently offline. {screenshot_result['error']}"

            # Return early - we've handled both tools
            return result

        # Normal flow: execute tools independently
        for part in response.candidates[0].content.parts:
            if not hasattr(part, 'function_call') or not part.function_call:
                continue

            function_call = part.function_call
            print(f"[Gemini Tool Call] {function_call.name}")

            # Execute the function based on name
            if function_call.name == "capture_stream_screenshot":
                # Extract channel parameter if provided
                target_channel = function_call.args.get("channel") if function_call.args else channel

                print(f"[Gemini] Capturing screenshot from channel: {target_channel}")

                # Import ad detection wrapper
                from tools import capture_screenshot_with_ad_detection
                from config import AD_DETECTION_ENABLED

                # Use ad detection if enabled
                if AD_DETECTION_ENABLED:
                    from config import AD_WAIT_PROMPT
                    screenshot_result = await capture_screenshot_with_ad_detection(
                        channel=target_channel,
                        llm_provider=self,
                        simple_prompt=AD_WAIT_PROMPT,
                        user_message=user_message,
                        msg_callback=msg_callback
                    )
                else:
                    screenshot_result = await capture_stream_screenshot(channel=target_channel)

                if screenshot_result["success"]:
                    print(f"[Tool Result] Screenshot captured: {screenshot_result['file_path']}")
                    if screenshot_result.get("ad_wait_info"):
                        ad_info = screenshot_result["ad_wait_info"]
                        if ad_info["had_ads"]:
                            print(f"[Tool Result] Ad detection: waited {ad_info['wait_time']:.1f}s, timed_out={ad_info['timed_out']}")
                    # Read the image
                    with open(screenshot_result["file_path"], 'rb') as f:
                        result["screenshot_data"] = f.read()
                    # Store stream info (including game name)
                    result["stream_info"] = screenshot_result.get("stream_info")
                else:
                    print(f"[Tool Result] Screenshot failed: {screenshot_result['error']}")
                    # Store error info to pass to LLM
                    result["screenshot_error"] = screenshot_result["error"]
                    # Also check if stream is offline
                    if screenshot_result.get("is_live") is False:
                        result["screenshot_error"] = f"The stream is currently offline. {screenshot_result['error']}"

            elif function_call.name == "web_search":
                # Extract parameters
                query = function_call.args.get("query", "")

                print(f"[Gemini] Searching web for: {query}")

                # Send notification to chat if enabled
                from config import WEB_SEARCH_NOTIFICATION
                if WEB_SEARCH_NOTIFICATION and msg_callback:
                    try:
                        await msg_callback(f"searching web for: {query}")
                    except Exception as e:
                        print(f"[Web Search] Failed to send notification: {e}")

                search_result = perform_web_search(query, num_results=10)

                if search_result["success"] and search_result["results"]:
                    results_list = []
                    for i, item in enumerate(search_result["results"], 1):
                        results_list.append({
                            "number": i,
                            "title": item["title"],
                            "snippet": item["snippet"],
                            "link": item["link"]
                        })

                    print(f"[Tool Result] Found {len(results_list)} results")

                    # Select, scrape, and extract from multiple websites
                    extracted_content = self._select_scrape_and_extract(user_message, results_list)

                    if extracted_content:
                        result["search_results"] = f"""Web search for '{query}':\n\n{extracted_content}"""
                        print(f"[Tool Result] Successfully extracted and compacted content")
                    else:
                        # Fallback to search snippets
                        results_text = f"Web search results for '{query}':\n\n"
                        for item in results_list:
                            results_text += f"{item['number']}. {item['title']}\n"
                            results_text += f"   {item['snippet']}\n\n"
                        result["search_results"] = results_text
                        print(f"[Tool Result] Extraction failed, using search snippets")
                else:
                    if search_result["success"]:
                        result["search_results"] = f"No search results found for '{query}'."
                        print(f"[Tool Result] No results found")
                    else:
                        print(f"[Tool Result] Search failed: {search_result['error']}")

        return result

    def _generate_enriched_query(self, user_message: str, screenshot_data: bytes) -> str | None:
        """Use smaller model with screenshot to generate an enriched search query.

        Args:
            user_message: The user's original question
            screenshot_data: The screenshot image data

        Returns:
            Enriched search query string, or None if generation failed
        """
        print(f"[Gemini] Using {GEMINI_SMALLER_MODEL} to generate enriched query with screenshot...")

        # Create prompt for query enrichment
        enrichment_prompt = f"""Based on the user's question and the screenshot of the Twitch stream, generate a single, specific search keyword or short phrase that would find the most relevant information to answer their question.

User question: {user_message}

Provide ONLY the search query, nothing else. Be specific and concise (1-3 words maximum)."""

        try:
            # Build message with screenshot
            parts = [
                types.Part(text=enrichment_prompt),
                types.Part.from_bytes(data=screenshot_data, mime_type='image/jpeg')
            ]
            messages = [types.Content(role="user", parts=parts)]

            # Request enriched query
            response = self.client.models.generate_content(
                model=GEMINI_SMALLER_MODEL,
                contents=messages,
                config=types.GenerateContentConfig(
                    max_output_tokens=50,
                    temperature=0.3,  # Low temperature for focused query
                )
            )

            enriched_query = response.text.strip()
            print(f"[Gemini] Enriched query: '{enriched_query}'")
            return enriched_query

        except Exception as e:
            print(f"[Gemini] Error generating enriched query: {e}")
            return None

    def _select_scrape_and_extract(self, user_message: str, results_list: list) -> str | None:
        """Use smaller model to select multiple websites, scrape them, and extract relevant content.

        Args:
            user_message: The user's original question
            results_list: List of result dicts (with number, title, snippet, link)

        Returns:
            Compacted extracted content from all selected websites, or None if failed
        """
        print(f"[Gemini] Using {GEMINI_SMALLER_MODEL} for multi-website selection...")

        # Format results for selection
        results_text = ""
        for item in results_list:
            results_text += f"{item['number']}. {item['title']}\n"
            results_text += f"   {item['snippet']}\n\n"

        # Create selection prompt
        selection_prompt = f"""User question: {user_message}

{results_text}

Which results (1-{len(results_list)}) are most relevant to answer the user's question? Select 1-5 most useful ones. Reply with ONLY a JSON array of numbers."""

        # Request website selection with JSON response
        try:
            response = self.client.models.generate_content(
                model=GEMINI_SMALLER_MODEL,
                contents=selection_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=WEBSITE_SELECTION_PROMPT,
                    max_output_tokens=50,
                    temperature=0.1,
                    response_mime_type="application/json"
                )
            )

            # Parse JSON response
            import json
            response_text = response.text.strip()
            selected_indices = json.loads(response_text)

            if not isinstance(selected_indices, list):
                print(f"[Gemini] Response is not a list: {response_text}")
                selected_indices = [1]  # Default to first result

            # Convert to 0-based and validate
            valid_indices = []
            for num in selected_indices:
                if isinstance(num, int) and 1 <= num <= len(results_list):
                    valid_indices.append(num - 1)  # Convert to 0-based

            if not valid_indices:
                print(f"[Gemini] No valid selections, defaulting to first result")
                valid_indices = [0]

            print(f"[Gemini] Selected {len(valid_indices)} websites: {[i+1 for i in valid_indices]}")

            # Scrape and extract from all selected websites
            extracted_contents = []
            for idx in valid_indices:
                selected_url = results_list[idx]["link"]
                selected_title = results_list[idx]["title"]
                print(f"[Gemini] Scraping result #{idx + 1}: {selected_url}")

                scrape_result = scrape_website(selected_url)

                if scrape_result["success"]:
                    print(f"[Gemini] Scraped {len(scrape_result['text'])} chars, extracting relevant content...")

                    # Extract relevant content using smaller model
                    extracted = self._extract_relevant_content(user_message, scrape_result['text'], selected_url, selected_title)

                    if extracted and extracted != "No relevant information found.":
                        extracted_contents.append(extracted)
                        print(f"[Gemini] Extracted content from {selected_url}")
                    else:
                        print(f"[Gemini] No relevant content found in {selected_url}")
                else:
                    print(f"[Gemini] Failed to scrape {selected_url}: {scrape_result['error']}")

            if not extracted_contents:
                print(f"[Gemini] No content extracted from any website")
                return None

            # Combine all extracted content
            combined = "\n\n---\n\n".join(extracted_contents)
            print(f"[Gemini] Combined {len(extracted_contents)} extracts into {len(combined)} characters")
            return combined

        except json.JSONDecodeError as e:
            print(f"[Gemini] JSON parse error: {e}, response: {response_text}")
            return None
        except Exception as e:
            print(f"[Gemini] Error in website selection/scraping: {e}")
            return None

    def _extract_relevant_content(self, user_question: str, scraped_text: str, url: str, title: str) -> str | None:
        """Use smaller model to extract only relevant information from scraped content.

        Args:
            user_question: The user's original question
            scraped_text: The full scraped webpage text
            url: The URL of the webpage
            title: The title of the webpage

        Returns:
            Compacted relevant content (2-4 sentences), or None if extraction failed
        """
        print(f"[Gemini] Using {GEMINI_SMALLER_MODEL} for content extraction...")

        # Create extraction prompt
        extraction_prompt = f"""User question: {user_question}

Webpage: {title}
URL: {url}

Webpage content:
{scraped_text}

Extract only the most relevant information that answers the user's question. Be concise (2-4 sentences max)."""

        try:
            response = self.client.models.generate_content(
                model=GEMINI_SMALLER_MODEL,
                contents=extraction_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=CONTENT_EXTRACTION_PROMPT,
                    max_output_tokens=200,
                    temperature=0.3
                )
            )

            extracted = response.text.strip() if response.text else None

            if extracted:
                # Add source attribution
                extracted = f"[Source: {title}]\n{extracted}"

            return extracted

        except Exception as e:
            print(f"[Gemini] Error extracting content: {e}")
            return None

    def _generate_final_response(self, context_message: str, user_prompt: str, tool_results: dict) -> str:
        """Use larger model to generate final response with tool results.

        Args:
            context_message: The context from chat history (empty string if no context)
            user_prompt: The current user message
            tool_results: Dict containing screenshot_data, search_results, and screenshot_error
        """
        print(f"[Gemini] Using {GEMINI_MODEL} for final response generation...")

        # Build messages for final response
        # Message structure: system prompt, then single user message with context and current message

        # Start with system instruction with current date/time
        from datetime import datetime
        current_datetime = datetime.now().strftime("%B %d, %Y at %I:%M %p")
        system_instruction = SYSTEM_PROMPT.format(current_datetime=current_datetime)

        # Build single user message combining context and current prompt
        message_parts = []

        # Combine context and current message into one text part
        if context_message:
            combined_text = f"{context_message}\n\n=== CURRENT MESSAGE ===\n{user_prompt}"
        else:
            combined_text = f"=== CURRENT MESSAGE ===\n{user_prompt}"

        message_parts.append(types.Part(text=combined_text))

        # Add screenshot if available
        if tool_results["screenshot_data"]:
            print("[Gemini] Adding screenshot to final response context")
            image_part = types.Part.from_bytes(
                data=tool_results["screenshot_data"],
                mime_type='image/jpeg'
            )
            message_parts.append(image_part)

        # Add screenshot error if screenshot was attempted but failed
        if tool_results.get("screenshot_error"):
            print(f"[Gemini] Adding screenshot error to context: {tool_results['screenshot_error']}")
            message_parts.append(types.Part(text=f"\n\n[SCREENSHOT TOOL ERROR]: {tool_results['screenshot_error']}"))

        # Add search results if available
        if tool_results["search_results"]:
            print("[Gemini] Adding search results to final response context")
            message_parts.append(types.Part(text=f"\n\n{tool_results['search_results']}"))

        # Create single user message with all parts
        user_messages = [types.Content(role="user", parts=message_parts)]

        # Generate final response with larger model
        response = self.client.models.generate_content(
            model=GEMINI_MODEL,
            contents=user_messages,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                max_output_tokens=800,
                temperature=0.7,
                thinking_config=types.ThinkingConfig(thinking_level="low")
            )
        )

        return response.text.strip() if response.text else "I couldn't generate a response."

    def get_simple_response(self, prompt: str) -> str:
        """Get a simple response without context or tools."""
        try:
            response = self.client.models.generate_content(
                model=GEMINI_SMALLER_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=100,
                    temperature=0.8
                )
            )
            return response.text.strip() if response.text else ""
        except Exception as e:
            print(f"[Gemini Simple Response] Error: {e}")
            return ""
