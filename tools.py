"""
Tool functions for the chatbot.
Includes screenshot capture, web search, and tool declarations for LLMs.
"""

import time
import subprocess
import requests
import asyncio
from pathlib import Path
from streamlink.session import Streamlink
from streamlink.options import Options
from bs4 import BeautifulSoup
from ddgs import DDGS
import cloudscraper
from PIL import Image
import pytesseract

from config import SCREENSHOT_PATH, TWITCH_CHANNEL, GOOGLE_SEARCH_API_KEY, GOOGLE_SEARCH_ENGINE_ID, STREAMLINK_OAUTH_TOKEN, AD_DETECTION_CHECK_INTERVAL, AD_DETECTION_MAX_WAIT
from twitch_api import check_stream_status


def init_screenshots_dir() -> None:
    """Initialize screenshots directory."""
    SCREENSHOT_PATH.mkdir(exist_ok=True)
    print(f"✓ Screenshots directory: {SCREENSHOT_PATH}")


def detect_ad_text_in_screenshot(screenshot_path: str) -> bool:
    """Detect if 'Preparing your stream' text is present in the bottom left 25% of the screenshot.

    Args:
        screenshot_path: Path to the screenshot image file

    Returns:
        True if ad text is detected, False otherwise
    """
    try:
        # Open the image
        img = Image.open(screenshot_path)
        width, height = img.size

        # Crop to bottom left 25% of the image
        # Bottom left quarter: left 50% width, bottom 50% height
        left = 0
        top = height // 2
        right = width // 2
        bottom = height

        cropped_img = img.crop((left, top, right, bottom))

        # Save cropped image for debugging
        cropped_img.save("debug_cropped.jpg")

        # Perform OCR on the cropped region
        text = pytesseract.image_to_string(cropped_img, lang='eng').lower()

        # Normalize whitespace: replace newlines and multiple spaces with single space
        text_normalized = ' '.join(text.split())

        # Also clean up common OCR errors and symbols
        text_cleaned = text_normalized.replace('—', ' ').replace('_', ' ').replace('|', ' ')
        text_cleaned = ' '.join(text_cleaned.split())  # Re-normalize after replacements

        # Print extracted text for debugging
        print(f"[Ad Detection] Extracted text: '{text_cleaned[:100]}'")

        # Strategy 1: Exact phrase matching
        ad_phrases = [
            "preparing your stream",
            "preparing the stream",
            "stream cleaning",
            "ads are running",
            "we're just doing a bit",
            "bit of stream cleaning",
            "preparing stream"
        ]

        if any(phrase in text_cleaned for phrase in ad_phrases):
            print(f"[Ad Detection] ✓ Exact phrase match found")
            return True

        # Strategy 2: Key word matching (fallback for bad OCR)
        # Check if we have the key words that indicate ad screen
        words_in_text = text_cleaned.split()

        has_preparing = any('prepar' in word for word in words_in_text)
        has_stream = any('stream' in word or 'steram' in word for word in words_in_text)
        has_your = any('your' in word or 'you' == word for word in words_in_text)

        # If we have "preparing" + "stream", it's likely an ad
        if has_preparing and has_stream:
            print(f"[Ad Detection] ✓ Key word match (preparing + stream)")
            return True

        # Additional check: "your" + "stream" (common variant)
        if has_your and has_stream:
            print(f"[Ad Detection] ✓ Key word match (your + stream)")
            return True

        print(f"[Ad Detection] ✗ No ad indicators found")
        return False

    except Exception as e:
        print(f"[Ad Detection] Error detecting ad text: {e}")
        return False


async def wait_for_ad_completion(channel: str, initial_screenshot_path: str, msg_callback=None) -> dict:
    """Poll the stream with a continuous ffmpeg process until ads are done or timeout is reached.

    This spawns a continuous ffmpeg stream that outputs screenshots at regular intervals.
    The continuous streaming helps "burn through" the ads.

    Args:
        channel: Twitch channel name
        initial_screenshot_path: Path to the initial screenshot with ads
        msg_callback: Optional async callback to send user message about ads

    Returns:
        Dict with success status and final screenshot path or error
    """
    print(f"[Ad Detection] Starting continuous stream ad polling for channel: {channel}")

    channel_url = f"https://www.twitch.tv/{channel}"
    ffmpeg_process = None

    try:
        # Get stream URL via streamlink with ad filtering
        session = Streamlink()
        plugin_name, plugin_class, resolved_url = session.resolve_url(channel_url)

        # Create options with OAuth token if available
        options = Options()
        options.set("disable-ads", True)
        options.set("low-latency", True)
        if STREAMLINK_OAUTH_TOKEN:
            options.set("api-header", [("Authorization", f"OAuth {STREAMLINK_OAUTH_TOKEN}")])

        plugin = plugin_class(session, resolved_url, options)
        streams = plugin.streams()

        if not streams:
            print(f"[Ad Detection] Could not get stream data")
            return {
                "success": False,
                "error": "Could not get stream data for ad polling",
                "file_path": None,
                "timed_out": True
            }

        # Get lower quality stream for ad detection (save bandwidth)
        stream_url = streams.get('360p') or streams.get('480p') or streams.get('worst') or streams.get('best')
        if not stream_url:
            print(f"[Ad Detection] No suitable stream quality found")
            return {
                "success": False,
                "error": "No suitable stream quality found",
                "file_path": None,
                "timed_out": True
            }

        stream_url = stream_url.url

        # Create pattern for output files
        timestamp = int(time.time())
        output_pattern = SCREENSHOT_PATH / f"ad_check_{timestamp}_%03d.jpg"

        # Calculate FPS for screenshot output based on check interval
        # We want one frame every AD_DETECTION_CHECK_INTERVAL seconds
        fps_value = f"1/{int(AD_DETECTION_CHECK_INTERVAL)}"  # e.g., "1/5" for 1 frame every 5 seconds

        # Spawn continuous ffmpeg process that streams and outputs frames
        # -vf fps=1/N outputs exactly 1 frame every N seconds
        # -q:v 5 lower quality for faster processing during ad detection
        # -start_number 1 starts numbering from 001
        command = [
            'ffmpeg',
            '-i', stream_url,
            '-vf', f'fps={fps_value}',  # Output 1 frame every N seconds
            '-q:v', '5',  # Lower quality for ad detection
            '-start_number', '1',
            str(output_pattern),
            '-y'
        ]

        print(f"[Ad Detection] Starting continuous ffmpeg stream (1 frame every {AD_DETECTION_CHECK_INTERVAL}s)")

        # Start ffmpeg process in background
        ffmpeg_process = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL
        )

        start_time = time.time()
        check_count = 0
        last_screenshot_path = None

        # Monitor for new screenshots
        while True:
            elapsed = time.time() - start_time

            # Check if we've exceeded the maximum wait time
            if elapsed > AD_DETECTION_MAX_WAIT:
                print(f"[Ad Detection] Timeout after {AD_DETECTION_MAX_WAIT}s")
                return {
                    "success": False,
                    "error": "Ad detection timeout - proceeding anyway",
                    "file_path": last_screenshot_path,
                    "timed_out": True
                }

            # Check if ffmpeg process died
            if ffmpeg_process.poll() is not None:
                print(f"[Ad Detection] FFmpeg process ended unexpectedly")
                return {
                    "success": False,
                    "error": "Stream process ended unexpectedly",
                    "file_path": last_screenshot_path,
                    "timed_out": True
                }

            # Wait for next interval
            await asyncio.sleep(AD_DETECTION_CHECK_INTERVAL)
            check_count += 1

            # Look for the latest screenshot file
            # FFmpeg outputs files with incrementing numbers
            expected_file = SCREENSHOT_PATH / f"ad_check_{timestamp}_{check_count:03d}.jpg"

            # Give ffmpeg a moment to finish writing the file
            await asyncio.sleep(0.5)

            if not expected_file.exists():
                print(f"[Ad Detection] Waiting for screenshot file: {expected_file.name}")
                # File might not be ready yet, continue waiting
                continue

            print(f"[Ad Detection] Check #{check_count} at {elapsed:.1f}s - analyzing {expected_file.name}")
            last_screenshot_path = str(expected_file)

            # Check if ads are still present
            has_ads = detect_ad_text_in_screenshot(last_screenshot_path)

            if not has_ads:
                print(f"[Ad Detection] Ads cleared after {elapsed:.1f}s and {check_count} checks")
                return {
                    "success": True,
                    "file_path": last_screenshot_path,
                    "error": None,
                    "timed_out": False,
                    "wait_time": elapsed
                }

            print(f"[Ad Detection] Still showing ads...")

    except Exception as e:
        print(f"[Ad Detection] Error during continuous stream polling: {e}")
        return {
            "success": False,
            "error": f"Error during ad polling: {str(e)}",
            "file_path": None,
            "timed_out": True
        }

    finally:
        # Clean up: terminate ffmpeg process if still running
        if ffmpeg_process and ffmpeg_process.poll() is None:
            print(f"[Ad Detection] Terminating ffmpeg process")
            ffmpeg_process.terminate()
            try:
                ffmpeg_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print(f"[Ad Detection] Force killing ffmpeg process")
                ffmpeg_process.kill()
                ffmpeg_process.wait()

        # Clean up intermediate ad check screenshots (keep only the last one if needed)
        try:
            for file in SCREENSHOT_PATH.glob(f"ad_check_{timestamp}_*.jpg"):
                if last_screenshot_path and file != Path(last_screenshot_path):
                    file.unlink()
                    print(f"[Ad Detection] Cleaned up intermediate file: {file.name}")
        except Exception as e:
            print(f"[Ad Detection] Error cleaning up files: {e}")


async def capture_stream_screenshot(channel: str = TWITCH_CHANNEL, skip_live_check: bool = False) -> dict:
    """Capture a screenshot from the Twitch stream using streamlink + ffmpeg.

    First checks if the stream is live before attempting to capture.
    If stream is offline, returns immediately with stream status info.

    Args:
        channel: Twitch channel name to capture from. Defaults to TWITCH_CHANNEL from config.
        skip_live_check: If True, skips the Twitch API check. Use during ad polling when we already know stream is live.

    Returns a dict with success status, file path, stream status, and any error message.
    Possible return structures:
    - Stream offline: {"success": False, "error": "Stream is offline", "is_live": False, "file_path": None}
    - Stream live with screenshot: {"success": True, "file_path": "path/to/screenshot.jpg", "is_live": True, "error": None}
    - Error: {"success": False, "error": "error message", "is_live": None, "file_path": None}
    """
    stream_status = None

    # Only check if stream is live if not skipping (first check or explicit check)
    if not skip_live_check:
        print(f"[Stream Check] Checking if '{channel}' is live...")

        # Call async stream check
        stream_status = await check_stream_status(channel)

        if stream_status["error"]:
            # Error checking stream status
            print(f"✗ Error checking stream status: {stream_status['error']}")
            return {
                "success": False,
                "error": f"Could not check stream status: {stream_status['error']}",
                "is_live": None,
                "file_path": None
            }

        if not stream_status["is_live"]:
            # Stream is offline
            print(f"✗ Stream '{channel}' is OFFLINE")
            return {
                "success": False,
                "error": f"Stream '{channel}' is currently offline",
                "is_live": False,
                "file_path": None,
                "stream_info": None
            }

        # Stream is live, proceed with screenshot capture
        print(f"✓ Stream '{channel}' is LIVE, capturing screenshot...")
    else:
        # Still fetch stream info even if skipping live check for game name
        print(f"[Stream Check] Fetching stream info for '{channel}'...")
        stream_status = await check_stream_status(channel)
        print(f"✓ Stream '{channel}' is LIVE, capturing screenshot...")

    channel_url = f"https://www.twitch.tv/{channel}"
    timestamp = int(time.time())
    output_file = SCREENSHOT_PATH / f"screenshot_{timestamp}.jpg"

    try:
        # Get stream URL via streamlink with ad filtering
        session = Streamlink()
        plugin_name, plugin_class, resolved_url = session.resolve_url(channel_url)

        # Create options with OAuth token if available
        options = Options()
        options.set("disable-ads", True)
        options.set("low-latency", True)
        if STREAMLINK_OAUTH_TOKEN:
            options.set("api-header", [("Authorization", f"OAuth {STREAMLINK_OAUTH_TOKEN}")])

        plugin = plugin_class(session, resolved_url, options)
        streams = plugin.streams()

        if not streams:
            return {
                "success": False,
                "error": "Could not get stream data (streamlink returned no streams)",
                "is_live": True,
                "file_path": None
            }

        # Get best quality stream URL
        stream_url = streams.get('best') or streams.get('720p') or streams.get('480p')
        if not stream_url:
            return {
                "success": False,
                "error": "No suitable stream quality found",
                "is_live": True,
                "file_path": None
            }

        stream_url = stream_url.url

        # Use FFmpeg to grab 1 frame
        command = [
            'ffmpeg',
            '-i', stream_url,
            '-ss', '00:00:01',
            '-frames:v', '1',
            '-q:v', '2',
            str(output_file),
            '-y'
        ]

        result = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=30
        )

        # Check if file was actually created (ffmpeg can return non-zero but still succeed)
        if output_file.exists() and output_file.stat().st_size > 0:
            print(f"✓ Screenshot saved: {output_file}")
            return {
                "success": True,
                "file_path": str(output_file),
                "is_live": True,
                "stream_info": stream_status.get("stream_info"),
                "error": None
            }
        else:
            # Print ffmpeg stderr output for debugging
            stderr_output = result.stderr.decode('utf-8', errors='ignore') if result.stderr else 'No error output'
            print(f"✗ FFmpeg failed to capture frame (return code: {result.returncode})")
            print(f"[FFmpeg Error Output]:\n{stderr_output[-1000:]}")  # Last 1000 chars

            return {
                "success": False,
                "error": f"FFmpeg failed to capture frame (exit code {result.returncode})",
                "is_live": True,
                "file_path": None
            }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": "Screenshot capture timed out",
            "is_live": True,
            "file_path": None
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "is_live": None,
            "file_path": None
        }


def perform_duckduckgo_search(query: str, num_results: int = 5) -> dict:
    """Perform a DuckDuckGo Search using DDGS library (for Ollama).

    Searches with both Finnish and English regions to get comprehensive results.

    Returns a dict with success status, results list, and any error message.
    Each result contains: title, link, snippet
    """
    try:
        print(f"[DuckDuckGo Search] Query: {query}, Max results: {num_results}")

        # Initialize DDGS
        ddgs = DDGS()

        # Perform searches in both Finnish and English regions
        # We'll get results from both and combine them
        all_results = []
        seen_urls = set()

        # Search in Finnish region first
        try:
            fi_results = ddgs.text(
                query=query,
                region="fi-fi",  # Finland - Finnish
                safesearch="moderate",
                max_results=num_results,
                backend="auto"
            )
            for result in fi_results:
                url = result.get('href', '')
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    all_results.append({
                        'title': result.get('title', 'No title'),
                        'link': url,
                        'snippet': result.get('body', 'No description')
                    })
        except Exception as e:
            print(f"[DuckDuckGo Search] Finnish region search failed: {e}")

        # Search in English region if we need more results
        if len(all_results) < num_results:
            try:
                en_results = ddgs.text(
                    query=query,
                    region="us-en",  # United States - English
                    safesearch="moderate",
                    max_results=num_results,
                    backend="auto"
                )
                for result in en_results:
                    if len(all_results) >= num_results:
                        break
                    url = result.get('href', '')
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        all_results.append({
                            'title': result.get('title', 'No title'),
                            'link': url,
                            'snippet': result.get('body', 'No description')
                        })
            except Exception as e:
                print(f"[DuckDuckGo Search] English region search failed: {e}")

        # Limit to requested number of results
        all_results = all_results[:num_results]

        if not all_results:
            return {
                "success": False,
                "error": "No search results found",
                "results": []
            }

        print(f"[DuckDuckGo Search] Found {len(all_results)} results")
        return {
            "success": True,
            "error": None,
            "results": all_results
        }

    except Exception as e:
        print(f"[DuckDuckGo Search] Error: {e}")
        return {
            "success": False,
            "error": f"DuckDuckGo search failed: {str(e)}",
            "results": []
        }


def perform_web_search(query: str, num_results: int = 5, language: str = "lang_en|lang_fi") -> dict:
    """Perform a Google Custom Search and return results (for Gemini).

    Returns a dict with success status, results list, and any error message.
    Each result contains: title, link, snippet
    """
    if not GOOGLE_SEARCH_API_KEY or not GOOGLE_SEARCH_ENGINE_ID:
        return {
            "success": False,
            "error": "Google Search API credentials not configured",
            "results": []
        }

    # Validate num_results
    num_results = max(1, min(10, num_results))  # Clamp between 1-10

    try:
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": GOOGLE_SEARCH_API_KEY,
            "cx": GOOGLE_SEARCH_ENGINE_ID,
            "q": query,
            "num": num_results,
            "lr": language  # Restrict to Finnish or English
        }

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        # Extract search results
        results = []
        if "items" in data:
            for item in data["items"]:
                results.append({
                    "title": item.get("title", ""),
                    "link": item.get("link", ""),
                    "snippet": item.get("snippet", "")
                })

        print(f"✓ Web search completed: {len(results)} results for '{query}'")
        return {
            "success": True,
            "results": results,
            "error": None
        }

    except requests.exceptions.Timeout:
        return {"success": False, "error": "Search request timed out", "results": []}
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": f"Search failed: {str(e)}", "results": []}
    except Exception as e:
        return {"success": False, "error": f"Unexpected error: {str(e)}", "results": []}


def scrape_website(url: str) -> dict:
    """Scrape text content from a website using BeautifulSoup with cloudscraper fallback.

    Returns a dict with success status, text content, and any error message.
    """
    def extract_text_from_html(html_content: str) -> str:
        """Extract and clean text from HTML content."""
        soup = BeautifulSoup(html_content, 'html.parser')

        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()

        # Get text
        text = soup.get_text()

        # Break into lines and remove leading/trailing space
        lines = (line.strip() for line in text.splitlines())
        # Break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # Remove blank lines
        text = '\n'.join(chunk for chunk in chunks if chunk)

        return text

    # First attempt: Regular requests with BeautifulSoup
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        text = extract_text_from_html(response.content)

        # Check if we got enough content
        if len(text) < 20:
            print(f"⚠ BeautifulSoup got only {len(text)} characters, trying cloudscraper...")
            raise ValueError("Insufficient content, trying cloudscraper")

        # Limit text length to avoid overwhelming the model (first 5000 chars)
        if len(text) > 5000:
            text = text[:5000] + "\n\n[Content truncated...]"

        print(f"✓ Website scraped (requests): {len(text)} characters from {url}")
        print(f"--- Start of scraped content preview ---\n{text[:500]}\n--- End of scraped content preview ---")
        return {
            "success": True,
            "text": text,
            "url": url,
            "error": None
        }

    except (requests.exceptions.RequestException, ValueError) as e:
        print(f"⚠ First attempt failed: {str(e)}, trying cloudscraper...")

        # Second attempt: Use cloudscraper for sites with protection
        try:
            scraper = cloudscraper.create_scraper()
            response = scraper.get(url, timeout=15)
            response.raise_for_status()

            text = extract_text_from_html(response.content)

            # Limit text length to avoid overwhelming the model (first 5000 chars)
            if len(text) > 5000:
                text = text[:5000] + "\n\n[Content truncated...]"

            print(f"✓ Website scraped (cloudscraper): {len(text)} characters from {url}")
            print(f"--- Start of scraped content preview ---\n{text[:500]}\n--- End of scraped content preview ---")
            return {
                "success": True,
                "text": text,
                "url": url,
                "error": None
            }

        except requests.exceptions.Timeout:
            return {"success": False, "text": None, "url": url, "error": "Request timed out (cloudscraper)"}
        except requests.exceptions.RequestException as e:
            return {"success": False, "text": None, "url": url, "error": f"Failed to fetch with cloudscraper: {str(e)}"}
        except Exception as e:
            return {"success": False, "text": None, "url": url, "error": f"Cloudscraper error: {str(e)}"}

    except Exception as e:
        return {"success": False, "text": None, "url": url, "error": f"Unexpected error: {str(e)}"}


async def fetch_user_data(user_login: str) -> dict:
    """Fetch Twitch user information by username.

    Args:
        user_login: The Twitch username to look up

    Returns:
        Dict with success status, user_data, and any error message.
        user_data includes: id, login, display_name, description, profile_image_url,
                          created_at, broadcaster_type, view_count
    """
    from twitch_api import get_twitch_client

    client = get_twitch_client()
    result = await client.get_user_info(user_login=user_login)

    if result["success"]:
        print(f"✓ Fetched data for {result['user_data']['display_name']}")
    else:
        print(f"✗ Failed to fetch user data for {user_login}: {result['error']}")

    return result


async def ban_user_from_chat(user_login: str, broadcaster_id: str, moderator_id: str, duration: int = 1) -> dict:
    """Ban a user from chat for a specified duration (default 1 second).

    Args:
        user_login: The Twitch username to ban
        broadcaster_id: The ID of the broadcaster/channel
        moderator_id: The ID of the moderator (bot user)
        duration: Duration in seconds (1-1209600). Defaults to 1 second.

    Returns:
        Dict with success status, user_name, ends_at, and any error message.
    """
    from twitch_api import get_twitch_client

    client = get_twitch_client()
    result = await client.ban_user(
        broadcaster_id=broadcaster_id,
        moderator_id=moderator_id,
        user_login=user_login,
        duration=duration,
        reason="Timeout by AI bot"
    )

    if result["success"]:
        print(f"✓ Banned {result['user_name']} for {duration} second(s)")
    else:
        print(f"✗ Failed to ban {user_login}: {result['error']}")

    return result


async def capture_screenshot_with_ad_detection(channel: str, llm_provider, simple_prompt: str, user_message: str, msg_callback=None) -> dict:
    """Capture screenshot with automatic ad detection and waiting.

    This is an async wrapper that:
    1. Captures initial screenshot
    2. Checks for ad text in bottom left 25%
    3. If ads detected, optionally notifies user and polls until ads clear
    4. Returns final screenshot or falls back after timeout

    Args:
        channel: Twitch channel name
        llm_provider: LLM provider to generate ad message
        simple_prompt: The simple system prompt to use for generating ad message
        user_message: The user's original message for context
        msg_callback: Optional async callback to send messages (async function)

    Returns:
        Dict with success, file_path, and optional ad_wait_info
    """
    # Capture initial screenshot
    initial_result = await capture_stream_screenshot(channel)

    if not initial_result["success"]:
        return initial_result

    # Check for ads
    has_ads = detect_ad_text_in_screenshot(initial_result["file_path"])

    if not has_ads:
        # No ads, return immediately
        print("[Ad Detection] No ads detected, proceeding normally")
        return {
            **initial_result,
            "ad_wait_info": None
        }

    # Ads detected!
    print("[Ad Detection] Pre-roll ads detected")

    # Send user-facing message if callback provided (only once, when first detected)
    if msg_callback:
        try:
            # Generate ad message using LLM with user's context
            ad_generation_prompt = f"{simple_prompt}\n\nUser's message: \"{user_message}\""

            ad_message = llm_provider.get_simple_response(ad_generation_prompt)

            # Fallback if LLM fails
            if not ad_message or len(ad_message) > 60:
                ad_message = "One sec, ads running..."

            await msg_callback(ad_message)
            print(f"[Ad Detection] Sent user message: {ad_message}")
        except Exception as e:
            print(f"[Ad Detection] Failed to send ad message: {e}")

    # Poll for ad completion
    wait_result = await wait_for_ad_completion(channel, initial_result["file_path"], msg_callback)

    if wait_result["timed_out"]:
        # Timeout - return with error but indicate we can proceed
        return {
            "success": True,  # Still success, we'll use what we have
            "file_path": initial_result["file_path"],
            "error": None,
            "ad_wait_info": {
                "had_ads": True,
                "timed_out": True,
                "wait_time": AD_DETECTION_MAX_WAIT
            }
        }

    # Ads cleared successfully
    return {
        "success": True,
        "file_path": wait_result["file_path"],
        "error": None,
        "ad_wait_info": {
            "had_ads": True,
            "timed_out": False,
            "wait_time": wait_result.get("wait_time", 0)
        }
    }


# Tool declarations for Gemini
GEMINI_SCREENSHOT_TOOL = {
    "name": "capture_stream_screenshot",
    "description": f"Captures a screenshot of a Twitch livestream. Use this when a user asks to see what's happening on stream, wants to know what's on screen, or asks about the current stream content. The default channel is '{TWITCH_CHANNEL}' (most of the time use this default unless the user specifically asks about a different channel).",
    "parameters": {
        "type": "object",
        "properties": {
            "channel": {
                "type": "string",
                "description": f"The Twitch channel name to capture screenshot from. Defaults to '{TWITCH_CHANNEL}' if not specified."
            }
        },
        "required": []
    }
}

GEMINI_WEB_SEARCH_TOOL = {
    "name": "web_search",
    "description": "Search the web ONLY when absolutely necessary. Use ONLY if: (1) user explicitly asks to search/google something, (2) question is about breaking news from the last 24-48 hours, (3) user needs a live price/score/value, or (4) you genuinely cannot answer from your training data. DO NOT use for general knowledge, games, movies, tech, history, science, or anything you already know. When in doubt, just answer without searching.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query. Only provide this if a search is truly necessary."
            },
        },
        "required": ["query"]
    }
}

GEMINI_BAN_TOOL = {
    "name": "ban_user",
    "description": "Temporarily ban (timeout) a user from chat for 1 second as a playful joke. Use this ONLY when the user explicitly asks you to ban someone or timeout someone as a joke/meme (e.g., 'ban John', 'timeout that guy'). This is a harmless 1-second timeout used for fun. Do NOT use this for actual moderation.",
    "parameters": {
        "type": "object",
        "properties": {
            "user_login": {
                "type": "string",
                "description": "The Twitch username (login name) of the user to ban. This should be the exact username without @ symbol."
            }
        },
        "required": ["user_login"]
    }
}

GEMINI_USER_INFO_TOOL = {
    "name": "get_user_info",
    "description": "Fetch detailed information about a Twitch user by their username. Use this when someone asks about a user's account details, profile, creation date, follower stats, or bio. Returns user ID, display name, bio/description, profile image, account creation date, broadcaster type (partner/affiliate), and view count.",
    "parameters": {
        "type": "object",
        "properties": {
            "user_login": {
                "type": "string",
                "description": "The Twitch username (login name) to look up. This should be the exact username without @ symbol."
            }
        },
        "required": ["user_login"]
    }
}

# Tool declarations for Ollama
OLLAMA_SCREENSHOT_TOOL = {
    "type": "function",
    "function": {
        "name": "capture_stream_screenshot",
        "description": f"Captures a screenshot of a Twitch livestream. IMPORTANT: This tool first checks if the stream is live. If the stream is offline, it will return an error message indicating the stream is offline - you should inform the user the stream is not currently live. Use this when a user asks to see what's happening on stream, wants to know what's on screen, or asks about the current stream content. The default channel is '{TWITCH_CHANNEL}' (most of the time use this default unless the user specifically asks about a different channel).",
        "parameters": {
            "type": "object",
            "properties": {
                "channel": {
                    "type": "string",
                    "description": f"The Twitch channel name to capture screenshot from. Defaults to '{TWITCH_CHANNEL}' if not specified."
                }
            },
            "required": []
        }
    }
}

OLLAMA_WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web using DuckDuckGo ONLY when absolutely necessary. Use ONLY if: (1) user explicitly asks to search/google something, (2) question is about breaking news from the last 24-48 hours, (3) user needs a live price/score/value, or (4) you genuinely cannot answer from your training data. DO NOT use for general knowledge, games, movies, tech, history, science, or anything you already know. When in doubt, just answer without searching.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query. Only provide this if a search is truly necessary."
                }
            },
            "required": ["query"]
        }
    }
}

OLLAMA_BAN_TOOL = {
    "type": "function",
    "function": {
        "name": "ban_user",
        "description": "Temporarily ban (timeout) a user from chat for 1 second as a playful joke. Use this ONLY when the user explicitly asks you to ban someone or timeout someone as a joke/meme (e.g., 'ban John', 'timeout that guy'). This is a harmless 1-second timeout used for fun. Do NOT use this for actual moderation.",
        "parameters": {
            "type": "object",
            "properties": {
                "user_login": {
                    "type": "string",
                    "description": "The Twitch username (login name) of the user to ban. This should be the exact username without @ symbol."
                }
            },
            "required": ["user_login"]
        }
    }
}

OLLAMA_USER_INFO_TOOL = {
    "type": "function",
    "function": {
        "name": "get_user_info",
        "description": "Fetch detailed information about a Twitch user by their username. Use this when someone asks about a user's account details, profile, creation date, follower stats, or bio. Returns user ID, display name, bio/description, profile image, account creation date, broadcaster type (partner/affiliate), and view count.",
        "parameters": {
            "type": "object",
            "properties": {
                "user_login": {
                    "type": "string",
                    "description": "The Twitch username (login name) to look up. This should be the exact username without @ symbol."
                }
            },
            "required": ["user_login"]
        }
    }
}
