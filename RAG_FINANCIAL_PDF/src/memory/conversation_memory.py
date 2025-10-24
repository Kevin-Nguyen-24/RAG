"""Conversation memory management for chatbot context."""
import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Optional, Any
from pathlib import Path
from loguru import logger
from src.config import config

class ConversationMemory:
    """Manages conversation history and context for the chatbot."""
    
    def __init__(self, db_path: Optional[Path] = None):
        """Initialize conversation memory with SQLite database."""
        self.db_path = db_path or config.memory.db_path
        self.max_history = config.memory.max_conversation_history
        self.short_term_window = config.memory.short_term_window
        self._init_database()
        logger.info(f"Initialized conversation memory at {self.db_path}")
    
    def _init_database(self):
        """Initialize the SQLite database schema."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Create conversations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                user_name TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_activity DATETIME DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT
            )
        """)
        
        # Create index for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_session_timestamp 
            ON conversations(session_id, timestamp)
        """)
        
        conn.commit()
        conn.close()
        logger.debug("Database schema initialized")
    
    def create_session(self, session_id: str, user_name: Optional[str] = None) -> str:
        """
        Create a new conversation session.
        
        Args:
            session_id: Unique identifier for the session
            user_name: Optional user name
            
        Returns:
            The session ID
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT OR IGNORE INTO sessions (session_id, user_name)
                VALUES (?, ?)
            """, (session_id, user_name))
            conn.commit()
            logger.info(f"Created session: {session_id}")
        except Exception as e:
            logger.error(f"Error creating session: {e}")
        finally:
            conn.close()
        
        return session_id
    
    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Add a message to conversation history.
        
        Args:
            session_id: The session identifier
            role: Message role ('user' or 'assistant')
            content: Message content
            metadata: Optional metadata dictionary
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        try:
            metadata_json = json.dumps(metadata) if metadata else None
            cursor.execute("""
                INSERT INTO conversations (session_id, role, content, metadata)
                VALUES (?, ?, ?, ?)
            """, (session_id, role, content, metadata_json))
            
            # Update session last activity
            cursor.execute("""
                UPDATE sessions 
                SET last_activity = CURRENT_TIMESTAMP 
                WHERE session_id = ?
            """, (session_id,))
            
            conn.commit()
            logger.debug(f"Added {role} message to session {session_id}")
        except Exception as e:
            logger.error(f"Error adding message: {e}")
        finally:
            conn.close()
    
    def get_recent_history(
        self,
        session_id: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get recent conversation history for a session.
        
        Args:
            session_id: The session identifier
            limit: Maximum number of messages to retrieve
            
        Returns:
            List of message dictionaries
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        limit = limit or self.max_history
        
        try:
            cursor.execute("""
                SELECT role, content, metadata, timestamp
                FROM conversations
                WHERE session_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (session_id, limit))
            
            rows = cursor.fetchall()
            
            # Reverse to get chronological order
            messages = []
            for row in reversed(rows):
                role, content, metadata_json, timestamp = row
                metadata = json.loads(metadata_json) if metadata_json else {}
                messages.append({
                    "role": role,
                    "content": content,
                    "metadata": metadata,
                    "timestamp": timestamp
                })
            
            logger.debug(f"Retrieved {len(messages)} messages for session {session_id}")
            return messages
            
        except Exception as e:
            logger.error(f"Error retrieving history: {e}")
            return []
        finally:
            conn.close()
    
    def get_short_term_context(self, session_id: str) -> List[Dict[str, str]]:
        """
        Get short-term context for immediate use in prompts.
        
        Args:
            session_id: The session identifier
            
        Returns:
            List of recent messages in chat format
        """
        history = self.get_recent_history(session_id, self.short_term_window)
        return [
            {"role": msg["role"], "content": msg["content"]}
            for msg in history
        ]
    
    def clear_session(self, session_id: str):
        """
        Clear conversation history for a session.
        
        Args:
            session_id: The session identifier
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        try:
            cursor.execute("DELETE FROM conversations WHERE session_id = ?", (session_id,))
            conn.commit()
            logger.info(f"Cleared history for session {session_id}")
        except Exception as e:
            logger.error(f"Error clearing session: {e}")
        finally:
            conn.close()
    
    def get_all_sessions(self) -> List[Dict[str, Any]]:
        """
        Get all active sessions.
        
        Returns:
            List of session dictionaries
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT session_id, user_name, created_at, last_activity
                FROM sessions
                ORDER BY last_activity DESC
            """)
            
            rows = cursor.fetchall()
            sessions = [
                {
                    "session_id": row[0],
                    "user_name": row[1],
                    "created_at": row[2],
                    "last_activity": row[3]
                }
                for row in rows
            ]
            
            return sessions
            
        except Exception as e:
            logger.error(f"Error retrieving sessions: {e}")
            return []
        finally:
            conn.close()