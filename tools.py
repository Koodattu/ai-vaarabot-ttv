"""
Tool functions for the chatbot.
Includes screenshot capture, web search, and tool declarations for LLMs.
"""

import time
import subprocess
import requests
from pathlib import Path
from streamlink.session import Streamlink
from bs4 import BeautifulSoup
import cloudscraper

from config import SCREENSHOT_PATH, TWITCH_CHANNEL, GOOGLE_SEARCH_API_KEY, GOOGLE_SEARCH_ENGINE_ID


def init_screenshots_dir() -> None:
    """Initialize screenshots directory."""
    SCREENSHOT_PATH.mkdir(exist_ok=True)
    print(f"✓ Screenshots directory: {SCREENSHOT_PATH}")


def capture_stream_screenshot() -> dict:
    """Capture a screenshot from the Twitch stream using streamlink + ffmpeg.

    Returns a dict with success status, file path, and any error message.
    """
    channel_url = f"https://www.twitch.tv/{TWITCH_CHANNEL}"
    timestamp = int(time.time())
    output_file = SCREENSHOT_PATH / f"screenshot_{timestamp}.jpg"

    try:
        # Get stream URL via streamlink with ad filtering
        session = Streamlink()
        plugin_name, plugin_class, resolved_url = session.resolve_url(channel_url)
        plugin = plugin_class(session, resolved_url, options={"disable-ads": True, "low-latency": True})
        streams = plugin.streams()

        if not streams:
            return {"success": False, "error": "Stream is offline", "file_path": None}

        # Get best quality stream URL
        stream_url = streams.get('best') or streams.get('720p') or streams.get('480p')
        if not stream_url:
            return {"success": False, "error": "No suitable stream quality found", "file_path": None}

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

        if result.returncode == 0 and output_file.exists():
            print(f"✓ Screenshot saved: {output_file}")
            return {"success": True, "file_path": str(output_file), "error": None}
        else:
            return {"success": False, "error": "FFmpeg failed to capture frame", "file_path": None}

    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Screenshot capture timed out", "file_path": None}
    except Exception as e:
        return {"success": False, "error": str(e), "file_path": None}


def perform_web_search(query: str, num_results: int = 5, language: str = "lang_en|lang_fi") -> dict:
    """Perform a Google Custom Search and return results.

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


# Tool declarations for Gemini
GEMINI_SCREENSHOT_TOOL = {
    "name": "capture_stream_screenshot",
    "description": "Captures a screenshot of the current Twitch livestream. Use this when a user asks to see what's happening on stream, wants to know what's on screen, or asks about the current stream content.",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": []
    }
}

GEMINI_WEB_SEARCH_TOOL = {
    "name": "web_search",
    "description": "Search the web for current information, facts, news, or any topic not in your knowledge base. Use this when users ask about recent events, current information, specific facts, or anything you don't know. Provide a single search keyword or term.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to look up on the web. Be specific and clear. Prefer a single word, keyword or term."
            },
        },
        "required": ["query"]
    }
}

# Tool declarations for Ollama
OLLAMA_SCREENSHOT_TOOL = {
    "type": "function",
    "function": {
        "name": "capture_stream_screenshot",
        "description": "Captures a screenshot of the current Twitch livestream. Use this when a user asks to see what's happening on stream, wants to know what's on screen, or asks about the current stream content.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
}
