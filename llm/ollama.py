"""
Ollama LLM implementation with tool support.
"""

import json
from ollama import Client

from config import OLLAMA_MODEL, OLLAMA_HOST, SYSTEM_PROMPT, TOOL_DETECTION_PROMPT, WEBSITE_SELECTION_PROMPT, CONTENT_EXTRACTION_PROMPT, ENABLED_TOOLS
from tools import capture_stream_screenshot, perform_web_search, scrape_website, ban_user_from_chat, fetch_user_data, OLLAMA_SCREENSHOT_TOOL, OLLAMA_WEB_SEARCH_TOOL, OLLAMA_BAN_TOOL, OLLAMA_USER_INFO_TOOL
from .base import BaseLLM


def _get_enabled_ollama_tools():
    """Get list of enabled Ollama tools based on ENABLED_TOOLS config."""
    tool_map = {
        "screenshot": OLLAMA_SCREENSHOT_TOOL,
        "web_search": OLLAMA_WEB_SEARCH_TOOL,
        "ban_user": OLLAMA_BAN_TOOL,
        "user_info": OLLAMA_USER_INFO_TOOL
    }
    enabled = [tool_map[tool_name] for tool_name in ENABLED_TOOLS if tool_name in tool_map]
    if enabled:
        print(f"[Ollama] Enabled tools: {', '.join(ENABLED_TOOLS)}")
    return enabled


class OllamaLLM(BaseLLM):
    """Ollama local LLM implementation with RAG context and tool support."""

    def __init__(self):
        self.client = Client(host=OLLAMA_HOST)

    def test_connection(self) -> bool:
        """Test Ollama API with main model."""
        print(f"\n🤖 Testing Ollama API (model: {OLLAMA_MODEL})...")
        try:
            response = self.client.chat(
                model=OLLAMA_MODEL,
                messages=[{'role': 'user', 'content': "Say 'OK' if you can hear me."}]
            )
            print(f"✓ Ollama response: {response.message.content.strip()}")
            return True
        except Exception as e:
            print(f"✗ Ollama API test failed: {e}")
            return False

    async def get_response(self, channel: str, user_id: str, user_name: str, message: str, database, game_name: str = None, msg_callback=None) -> str:
        """Get response from Ollama with RAG context and tool support."""
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

            print(f"[Ollama Prompt]\n{context_message}\n\n=== CURRENT MESSAGE ===\n{user_prompt}\n")

            # STEP 1: Use model to detect which tools to use
            # For tool detection, combine context and message
            full_prompt = f"{context_message}\n\n=== CURRENT MESSAGE ===\n{user_prompt}" if context_message else user_prompt
            tool_results = await self._detect_and_execute_tools(full_prompt, user_prompt, channel, message, msg_callback)

            # STEP 2: Use model to generate final response with restructured messages
            final_response = self._generate_final_response(context_message, user_prompt, tool_results)

            return final_response

        except Exception as e:
            print(f"Ollama error: {e}")
            return "Sorry, I couldn't process that right now."

    async def _detect_and_execute_tools(self, full_prompt: str, user_prompt: str, channel: str, user_message: str, msg_callback=None) -> dict:
        """Use model to detect tools and execute them.

        Args:
            full_prompt: The complete prompt with chat history context
            user_prompt: Just the users message (for website selection context)
            channel: Twitch channel name for screenshot capture
            user_message: The users original message for ad notification context
            msg_callback: Optional async callback to send intermediate messages

        Returns dict with:
        - screenshot_data: file path or None
        - search_results: str or None
        - screenshot_error: str or None (error message if screenshot failed)
        """
        result = {
            "screenshot_data": None,
            "search_results": None,
            "screenshot_error": None
        }

        # Get enabled tools based on configuration
        enabled_tools = _get_enabled_ollama_tools()
        if not enabled_tools:
            print("[Ollama] No tools enabled, skipping tool detection")
            return result

        print(f"[Ollama] Using {OLLAMA_MODEL} for tool detection...")

        # Request with model for tool detection
        response = self.client.chat(
            model=OLLAMA_MODEL,
            messages=[
                {'role': 'system', 'content': TOOL_DETECTION_PROMPT},
                {'role': 'user', 'content': full_prompt}
            ],
            tools=enabled_tools,
            options={'temperature': 0.3}  # Lower temperature for more deterministic tool selection
        )

        # First pass: collect all tool calls to check if both screenshot and web_search are requested
        tool_calls = []
        if response.message.tool_calls:
            for tool_call in response.message.tool_calls:
                tool_calls.append({
                    'name': tool_call.function.name,
                    'args': tool_call.function.arguments if hasattr(tool_call.function, 'arguments') else {}
                })

        # Check if BOTH screenshot and web_search are requested
        has_screenshot = any(tc['name'] == "capture_stream_screenshot" for tc in tool_calls)
        has_web_search = any(tc['name'] == "web_search" for tc in tool_calls)
        both_tools_requested = has_screenshot and has_web_search

        if both_tools_requested:
            print("[Ollama] Both screenshot and web_search requested - using enriched query flow")

            # Step 1: Capture screenshot first
            screenshot_call = next(tc for tc in tool_calls if tc['name'] == "capture_stream_screenshot")
            target_channel = screenshot_call['args'].get("channel", channel)

            print(f"[Ollama] Capturing screenshot from channel: {target_channel}")
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

            # Log ad detection info if present
            if screenshot_result.get("ad_wait_info"):
                ad_info = screenshot_result["ad_wait_info"]
                if ad_info["had_ads"]:
                    print(f"[Tool Result] Ad detection: waited {ad_info['wait_time']:.1f}s, timed_out={ad_info['timed_out']}")

            # Step 2: If screenshot succeeded, use it to enrich the search query
            if screenshot_result["success"]:
                print(f"[Ollama] Screenshot captured: {screenshot_result['file_path']}")
                result["screenshot_data"] = screenshot_result["file_path"]

                # Generate enriched search query using screenshot
                enriched_query = self._generate_enriched_query(user_message, screenshot_result["file_path"])

                if enriched_query:
                    print(f"[Ollama] Enriched search query: {enriched_query}")
                    # Perform web search with enriched query
                    search_result = await perform_web_search(enriched_query)
                    if search_result["success"]:
                        # Use multi-website selection and extraction
                        compacted_content = self._select_scrape_and_extract(user_message, search_result["results"])
                        if compacted_content:
                            result["search_results"] = compacted_content
                    else:
                        print(f"[Ollama] Web search failed: {search_result.get('error', 'Unknown error')}")
                else:
                    print("[Ollama] Failed to generate enriched query, proceeding without web search")
            else:
                # Screenshot failed
                print(f"[Ollama] Screenshot failed: {screenshot_result['error']}")
                error_message = screenshot_result['error']
                if screenshot_result.get("is_live") is False:
                    error_message = f"The stream is currently offline. {error_message}"
                result["screenshot_error"] = error_message

        # Normal flow: execute tools independently
        else:
            for tool_call in tool_calls:
                tool_name = tool_call['name']
                tool_args = tool_call['args']
                print(f"[Ollama Tool Call] {tool_name}")

                # Handle screenshot tool
                if tool_name == "capture_stream_screenshot":
                    target_channel = tool_args.get("channel", channel)
                    print(f"[Ollama] Capturing screenshot from channel: {target_channel}...")

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

                    # Log ad detection info if present
                    if screenshot_result.get("ad_wait_info"):
                        ad_info = screenshot_result["ad_wait_info"]
                        if ad_info["had_ads"]:
                            print(f"[Tool Result] Ad detection: waited {ad_info['wait_time']:.1f}s, timed_out={ad_info['timed_out']}")

                    if screenshot_result["success"]:
                        print(f"[Ollama] Screenshot captured: {screenshot_result['file_path']}")
                        result["screenshot_data"] = screenshot_result["file_path"]
                    else:
                        print(f"[Ollama] Screenshot failed: {screenshot_result['error']}")
                        error_message = screenshot_result['error']
                        if screenshot_result.get("is_live") is False:
                            error_message = f"The stream is currently offline. {error_message}"
                        result["screenshot_error"] = error_message

                # Handle web search tool
                elif tool_name == "web_search":
                    query = tool_args.get("query", "")
                    print(f"[Ollama] Performing web search: {query}")

                    from config import WEB_SEARCH_NOTIFICATION
                    if WEB_SEARCH_NOTIFICATION and msg_callback:
                        try:
                            await msg_callback("🔍 Searching the web...")
                        except Exception as e:
                            print(f"[Ollama] Failed to send search notification: {e}")

                    search_result = await perform_web_search(query)

                    if search_result["success"]:
                        print(f"[Ollama] Search returned {len(search_result['results'])} results")

                        # Use multi-website selection and extraction
                        compacted_content = self._select_scrape_and_extract(user_message, search_result["results"])

                        if compacted_content:
                            result["search_results"] = compacted_content
                        else:
                            result["search_results"] = "No relevant information found from web search."
                    else:
                        print(f"[Ollama] Web search failed: {search_result.get('error', 'Unknown error')}")
                        result["search_results"] = f"Web search failed: {search_result.get('error', 'Unknown error')}"

                # Handle ban user tool
                elif tool_name == "ban_user":
                    user_login = tool_args.get("user_login", "")

                    if not user_login:
                        print(f"[Ollama] Ban tool called without user_login")
                        result["ban_error"] = "No username provided"
                        continue

                    print(f"[Ollama] Attempting to ban user: {user_login}")

                    # Get broadcaster and moderator IDs
                    from twitch_api import get_twitch_client
                    twitch_client = get_twitch_client()

                    # Get broadcaster ID from channel name
                    broadcaster_users = []
                    async for user in twitch_client.twitch.get_users(logins=[channel.lower()]):
                        broadcaster_users.append(user)

                    if not broadcaster_users:
                        print(f"[Tool Result] Could not find broadcaster '{channel}'")
                        result["ban_error"] = f"Could not find channel '{channel}'"
                        continue

                    broadcaster_id = broadcaster_users[0].id

                    # Get moderator ID (bot's own user ID)
                    moderator_users = []
                    async for user in twitch_client.twitch.get_users():
                        moderator_users.append(user)

                    if not moderator_users:
                        print(f"[Tool Result] Could not get bot user ID")
                        result["ban_error"] = "Could not authenticate bot"
                        continue

                    moderator_id = moderator_users[0].id

                    # Execute the ban
                    ban_result = await ban_user_from_chat(
                        user_login=user_login,
                        broadcaster_id=broadcaster_id,
                        moderator_id=moderator_id,
                        duration=1
                    )

                    if ban_result["success"]:
                        print(f"[Tool Result] Successfully banned {ban_result['user_name']} for 1 second")
                        result["ban_result"] = f"Successfully timed out {ban_result['user_name']} for 1 second"
                    else:
                        print(f"[Tool Result] Ban failed: {ban_result['error']}")
                        result["ban_error"] = ban_result["error"]

                # Handle user info tool
                elif tool_name == "get_user_info":
                    user_login = tool_args.get("user_login", "")

                    if not user_login:
                        print(f"[Ollama] User info tool called without user_login")
                        result["user_info_error"] = "No username provided"
                        continue

                    print(f"[Ollama] Fetching user info for: {user_login}")

                    # Execute the user info fetch
                    user_info_result = await fetch_user_data(user_login=user_login)

                    if user_info_result["success"]:
                        user_data = user_info_result["user_data"]
                        print(f"[Tool Result] Successfully fetched info for {user_data['display_name']}")

                        # Format user data as readable text
                        user_info_text = f"""User Info for {user_data['display_name']}:
- Username: {user_data['login']}
- Display Name: {user_data['display_name']}
- User ID: {user_data['id']}
- Bio: {user_data['description']}
- Account Created: {user_data['created_at']}
- Broadcaster Type: {user_data['broadcaster_type']}
- Total Views: {user_data['view_count']:,}
- Profile Image: {user_data['profile_image_url']}"""

                        result["user_info_data"] = user_info_text
                    else:
                        print(f"[Tool Result] User info fetch failed: {user_info_result['error']}")
                        result["user_info_error"] = user_info_result["error"]

        return result

    def _generate_enriched_query(self, user_message: str, screenshot_path: str) -> str | None:
        """Use model with screenshot to generate an enriched search query.

        Args:
            user_message: The user's original question
            screenshot_path: Path to the screenshot image

        Returns:
            Enriched search query string, or None if generation failed
        """
        print(f"[Ollama] Using {OLLAMA_MODEL} to generate enriched query with screenshot...")

        # Create prompt for query enrichment
        enrichment_prompt = f"""Based on the user's question and the screenshot of the Twitch stream, generate a single, specific search keyword or short phrase that would find the most relevant information to answer their question.

User question: {user_message}

Provide ONLY the search query, nothing else. Be specific and concise (1-3 words maximum)."""

        try:
            response = self.client.chat(
                model=OLLAMA_MODEL,
                messages=[{
                    'role': 'user',
                    'content': enrichment_prompt,
                    'images': [screenshot_path]
                }],
                options={'temperature': 0.5}
            )

            enriched_query = response.message.content.strip()
            print(f"[Ollama] Generated enriched query: {enriched_query}")
            return enriched_query

        except Exception as e:
            print(f"[Ollama] Failed to generate enriched query: {e}")
            return None

    def _select_scrape_and_extract(self, user_message: str, results_list: list) -> str | None:
        """Use model to select multiple websites, scrape them, and extract relevant content.

        Args:
            user_message: The user's original question
            results_list: List of result dicts (with number, title, snippet, link)

        Returns:
            Compacted extracted content from all selected websites, or None if failed
        """
        print(f"[Ollama] Using {OLLAMA_MODEL} for multi-website selection...")

        # Format results for selection
        results_text = ""
        for item in results_list:
            results_text += f"{item['number']}. {item['title']}\n   {item['snippet']}\n   {item['link']}\n\n"

        # Create selection prompt
        selection_prompt = f"""User question: {user_message}

{results_text}

Which results (1-{len(results_list)}) are most relevant to answer the user's question? Select 1-5 most useful ones. Reply with ONLY a JSON array of numbers."""

        # Request website selection with JSON response
        try:
            response = self.client.chat(
                model=OLLAMA_MODEL,
                messages=[
                    {'role': 'system', 'content': WEBSITE_SELECTION_PROMPT},
                    {'role': 'user', 'content': selection_prompt}
                ],
                format='json',
                options={'temperature': 0.3}
            )

            # Parse JSON response
            selection_json = response.message.content.strip()
            print(f"[Ollama] Selection response: {selection_json}")

            selected_indices = json.loads(selection_json)

            if not isinstance(selected_indices, list) or not selected_indices:
                print("[Ollama] Invalid selection format, defaulting to first result")
                selected_indices = [1]

            # Validate and cap selections
            selected_indices = [idx for idx in selected_indices if 1 <= idx <= len(results_list)][:5]
            print(f"[Ollama] Selected {len(selected_indices)} websites to scrape")

            # Scrape and extract from each selected website
            all_extracted_content = []

            for idx in selected_indices:
                result_item = results_list[idx - 1]  # Convert to 0-indexed
                url = result_item['link']
                title = result_item['title']

                print(f"[Ollama] Scraping website {idx}: {title}")
                scrape_result = scrape_website(url)

                if scrape_result["success"]:
                    scraped_text = scrape_result["content"]
                    print(f"[Ollama] Scraped {len(scraped_text)} characters from {title}")

                    # Extract relevant content
                    extracted = self._extract_relevant_content(user_message, scraped_text, url, title)
                    if extracted and extracted != "No relevant information found.":
                        all_extracted_content.append(f"From {title}:\n{extracted}")
                else:
                    print(f"[Ollama] Failed to scrape {url}: {scrape_result.get('error', 'Unknown error')}")

            if all_extracted_content:
                # Combine all extracted content
                combined_content = "\n\n".join(all_extracted_content)
                print(f"[Ollama] Combined extracted content: {len(combined_content)} characters")
                return combined_content
            else:
                print("[Ollama] No content could be extracted from any website")
                return None

        except json.JSONDecodeError as e:
            print(f"[Ollama] Failed to parse selection JSON: {e}")
            return None
        except Exception as e:
            print(f"[Ollama] Website selection/scraping failed: {e}")
            return None

    def _extract_relevant_content(self, user_question: str, scraped_text: str, url: str, title: str) -> str | None:
        """Use model to extract only relevant information from scraped content.

        Args:
            user_question: The user's original question
            scraped_text: The full scraped webpage text
            url: The URL of the webpage
            title: The title of the webpage

        Returns:
            Compacted relevant content (2-4 sentences), or None if extraction failed
        """
        print(f"[Ollama] Using {OLLAMA_MODEL} for content extraction...")

        # Create extraction prompt
        extraction_prompt = f"""User question: {user_question}

Webpage: {title}
URL: {url}

Webpage content:
{scraped_text}

Extract only the most relevant information that answers the user's question. Be concise (2-4 sentences max)."""

        try:
            response = self.client.chat(
                model=OLLAMA_MODEL,
                messages=[
                    {'role': 'system', 'content': CONTENT_EXTRACTION_PROMPT},
                    {'role': 'user', 'content': extraction_prompt}
                ],
                options={'temperature': 0.3}
            )

            extracted_content = response.message.content.strip()
            print(f"[Ollama] Extracted: {extracted_content[:100]}...")
            return extracted_content

        except Exception as e:
            print(f"[Ollama] Content extraction failed: {e}")
            return None

    def _generate_final_response(self, context_message: str, user_prompt: str, tool_results: dict) -> str:
        """Use model to generate final response with tool results.

        Args:
            context_message: The context from chat history (empty string if no context)
            user_prompt: The current user message
            tool_results: Dict containing screenshot_data, search_results, and screenshot_error
        """
        print(f"[Ollama] Using {OLLAMA_MODEL} for final response generation...")

        # Build messages for final response
        # Format system prompt with current date/time
        from datetime import datetime
        current_datetime = datetime.now().strftime("%B %d, %Y at %I:%M %p")
        system_instruction = SYSTEM_PROMPT.format(current_datetime=current_datetime)

        # Build user message combining context and current prompt
        if context_message:
            combined_text = f"{context_message}\n\n=== CURRENT MESSAGE ===\n{user_prompt}"
        else:
            combined_text = user_prompt

        # For Ollama, we need to build the message differently
        # If we have a screenshot, we need to include it in the images field
        user_message = {'role': 'user', 'content': combined_text}

        # Add screenshot if available
        if tool_results["screenshot_data"]:
            print("[Ollama] Including screenshot in final response")
            user_message['images'] = [tool_results["screenshot_data"]]

        # Add screenshot error if screenshot was attempted but failed
        if tool_results.get("screenshot_error"):
            combined_text += f"\n\n(Screenshot capture failed: {tool_results['screenshot_error']})"
            user_message['content'] = combined_text

        # Add search results if available
        if tool_results["search_results"]:
            print("[Ollama] Including web search results in final response")
            combined_text += f"\n\n=== WEB SEARCH RESULTS ===\n{tool_results['search_results']}"
            user_message['content'] = combined_text

        # Add ban result if available
        if tool_results.get("ban_result"):
            print(f"[Ollama] Including ban result in final response")
            combined_text += f"\n\n[BAN TOOL RESULT]: {tool_results['ban_result']}"
            user_message['content'] = combined_text

        # Add ban error if ban was attempted but failed
        if tool_results.get("ban_error"):
            print(f"[Ollama] Including ban error in final response")
            combined_text += f"\n\n[BAN TOOL ERROR]: {tool_results['ban_error']}"
            user_message['content'] = combined_text

        # Add user info data if available
        if tool_results.get("user_info_data"):
            print(f"[Ollama] Including user info data in final response")
            combined_text += f"\n\n[USER INFO DATA]:\n{tool_results['user_info_data']}"
            user_message['content'] = combined_text

        # Add user info error if fetch was attempted but failed
        if tool_results.get("user_info_error"):
            print(f"[Ollama] Including user info error in final response")
            combined_text += f"\n\n[USER INFO ERROR]: {tool_results['user_info_error']}"
            user_message['content'] = combined_text

        # Generate final response
        response = self.client.chat(
            model=OLLAMA_MODEL,
            messages=[
                {'role': 'system', 'content': system_instruction},
                user_message
            ],
            options={'temperature': 0.7}
        )

        return response.message.content.strip() if response.message.content else "I couldn't generate a response."

    def get_simple_response(self, prompt: str) -> str:
        """Get a simple response without context or tools."""
        try:
            response = self.client.chat(
                model=OLLAMA_MODEL,
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': 0.8}
            )
            return response.message.content.strip() if response.message.content else ""
        except Exception as e:
            print(f"[Ollama Simple Response] Error: {e}")
            return ""