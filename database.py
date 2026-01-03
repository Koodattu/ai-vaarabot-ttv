"""
Database operations for chat message storage and retrieval.
Handles both SQLite (for structured storage) and ChromaDB (for vector search/RAG).
"""

import sqlite3
import time
from pathlib import Path
import chromadb
from chromadb.config import Settings

from config import DB_PATH, CHROMA_PATH, RAG_RESULTS_COUNT, RECENT_CHAT_COUNT, RECENT_USER_COUNT, RECENT_BOT_COUNT


class Database:
    """Manages SQLite and ChromaDB connections for message storage and retrieval."""

    def __init__(self):
        self.db_conn: sqlite3.Connection | None = None
        self.chroma_collection = None

    def initialize(self) -> None:
        """Initialize both SQLite and ChromaDB."""
        self._init_sqlite()
        self._init_chromadb()

    def _init_sqlite(self) -> None:
        """Initialize SQLite database for storing all chat messages."""
        self.db_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self.db_conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                channel TEXT NOT NULL,
                user_id TEXT NOT NULL,
                user_name TEXT NOT NULL,
                message TEXT NOT NULL,
                is_bot INTEGER DEFAULT 0
            )
        """)
        self.db_conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON messages(timestamp)")
        self.db_conn.execute("CREATE INDEX IF NOT EXISTS idx_user_id ON messages(user_id)")
        self.db_conn.execute("CREATE INDEX IF NOT EXISTS idx_is_bot ON messages(is_bot)")
        self.db_conn.execute("CREATE INDEX IF NOT EXISTS idx_channel ON messages(channel)")
        self.db_conn.execute("CREATE INDEX IF NOT EXISTS idx_channel_timestamp ON messages(channel, timestamp)")
        self.db_conn.commit()
        print(f"✓ SQLite database initialized: {DB_PATH}")

    def _init_chromadb(self) -> None:
        """Initialize ChromaDB for vector search."""
        client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        self.chroma_collection = client.get_or_create_collection(
            name="chat_messages",
            metadata={"hnsw:space": "cosine"}
        )
        print(f"✓ ChromaDB initialized: {CHROMA_PATH}")

    def store_message(self, channel: str, user_id: str, user_name: str, message: str, is_bot: bool = False) -> int:
        """Store a message in SQLite and ChromaDB. Returns the message ID."""
        timestamp = time.time()
        channel = channel.lower()  # Normalize channel name

        # Store in SQLite
        cursor = self.db_conn.execute(
            "INSERT INTO messages (timestamp, channel, user_id, user_name, message, is_bot) VALUES (?, ?, ?, ?, ?, ?)",
            (timestamp, channel, user_id, user_name, message, 1 if is_bot else 0)
        )
        self.db_conn.commit()
        msg_id = cursor.lastrowid

        # Store in ChromaDB for vector search
        doc_text = f"{user_name}: {message}"
        self.chroma_collection.add(
            documents=[doc_text],
            metadatas=[{
                "channel": channel,
                "user_id": user_id,
                "user_name": user_name,
                "is_bot": is_bot,
                "timestamp": timestamp
            }],
            ids=[str(msg_id)]
        )

        return msg_id

    def get_recent_chat_messages(self, channel: str, limit: int = RECENT_CHAT_COUNT) -> list[dict]:
        """Get the most recent chat messages from any user in a specific channel."""
        cursor = self.db_conn.execute(
            "SELECT user_name, message, is_bot FROM messages WHERE channel = ? ORDER BY timestamp DESC LIMIT ?",
            (channel.lower(), limit)
        )
        rows = cursor.fetchall()
        return [{"user": row[0], "message": row[1], "is_bot": bool(row[2])} for row in reversed(rows)]

    def get_recent_user_messages(self, channel: str, user_id: str, limit: int = RECENT_USER_COUNT) -> list[dict]:
        """Get the most recent messages from a specific user in a specific channel."""
        cursor = self.db_conn.execute(
            "SELECT user_name, message FROM messages WHERE channel = ? AND user_id = ? AND is_bot = 0 ORDER BY timestamp DESC LIMIT ?",
            (channel.lower(), user_id, limit)
        )
        rows = cursor.fetchall()
        return [{"user": row[0], "message": row[1]} for row in reversed(rows)]

    def get_recent_bot_messages(self, channel: str, limit: int = RECENT_BOT_COUNT) -> list[dict]:
        """Get the most recent messages from the bot in a specific channel."""
        cursor = self.db_conn.execute(
            "SELECT user_name, message FROM messages WHERE channel = ? AND is_bot = 1 ORDER BY timestamp DESC LIMIT ?",
            (channel.lower(), limit)
        )
        rows = cursor.fetchall()
        return [{"user": row[0], "message": row[1]} for row in reversed(rows)]

    def search_similar_messages(self, channel: str, query: str, limit: int = RAG_RESULTS_COUNT) -> list[dict]:
        """Search for similar messages using ChromaDB vector search, filtered by channel."""
        # Check if collection has any documents
        if self.chroma_collection.count() == 0:
            return []

        results = self.chroma_collection.query(
            query_texts=[query],
            n_results=min(limit * 3, self.chroma_collection.count()),  # Fetch more to filter
            where={"channel": channel.lower()}
        )

        messages = []
        if results and results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                if len(messages) >= limit:
                    break
                metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                messages.append({
                    "text": doc,
                    "is_bot": metadata.get("is_bot", False),
                    "distance": results['distances'][0][i] if results['distances'] else 0
                })

        return messages

    def build_context(self, channel: str, user_id: str, user_name: str, current_message: str) -> str:
        """Build context string from various sources, scoped to a specific channel."""
        context_parts = []

        # 1. Recent chat messages
        recent_chat = self.get_recent_chat_messages(channel)
        if recent_chat:
            chat_lines = []
            for msg in recent_chat:
                prefix = "[BOT]" if msg["is_bot"] else ""
                chat_lines.append(f"{prefix}{msg['user']}: {msg['message']}")
            context_parts.append("=== RECENT CHAT ===\n" + "\n".join(chat_lines))

        # 2. Recent messages from this specific user
        recent_user = self.get_recent_user_messages(channel, user_id)
        if recent_user:
            user_lines = [f"{msg['user']}: {msg['message']}" for msg in recent_user]
            context_parts.append(f"=== RECENT FROM {user_name.upper()} ===\n" + "\n".join(user_lines))

        # 3. Recent bot messages
        recent_bot = self.get_recent_bot_messages(channel)
        if recent_bot:
            bot_lines = [f"{msg['user']}: {msg['message']}" for msg in recent_bot]
            context_parts.append("=== RECENT BOT RESPONSES ===\n" + "\n".join(bot_lines))

        # 4. RAG - similar messages from history
        similar = self.search_similar_messages(channel, current_message)
        if similar:
            rag_lines = [msg["text"] for msg in similar]
            context_parts.append("=== RELATED PAST MESSAGES ===\n" + "\n".join(rag_lines))

        return "\n\n".join(context_parts)

    def close(self) -> None:
        """Close database connections."""
        if self.db_conn:
            self.db_conn.close()
