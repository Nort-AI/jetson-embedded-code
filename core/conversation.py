"""
core/conversation.py — Per-session conversation context for the Nort NL interface.

Stores the last MAX_HISTORY Q/A pairs per session. The history is injected as
prior conversation turns into Claude API calls so follow-up questions work
naturally ("and what about that zone?", "show me more detail").

Sessions are keyed by a client-generated UUID stored in localStorage (tab-persistent).
Sessions expire after SESSION_TTL seconds of inactivity and are evicted LRU when the
cap is reached (Jetson-friendly, bounded memory).

Design
------
  • Text-only history — crop images are NEVER stored, only assistant text responses
  • MAX_HISTORY = 6 pairs → at most ~3,000 input tokens per VLM call (Haiku ~$0.0004)
  • Thread-safe via a single lock around the OrderedDict
  • Zero dependency on Flask / request context — importable anywhere
"""

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional

try:
    from system.logger_setup import setup_logger
    logger = setup_logger(__name__)
except Exception:
    import logging
    logger = logging.getLogger(__name__)

# ── Tunable constants ─────────────────────────────────────────────────────────
MAX_HISTORY  = 6      # Q/A pairs retained per session
SESSION_TTL  = 14400  # 4 hours of inactivity before eviction
MAX_SESSIONS = 100    # LRU evict oldest when exceeded


# ═══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Exchange:
    """A single user→assistant Q/A pair."""
    user_text:      str
    assistant_text: str
    ts:             float = field(default_factory=time.time)


class ConversationSession:
    """Holds the rolling history for one browser session."""

    def __init__(self, session_id: str) -> None:
        self.session_id  = session_id
        self.history:    list[Exchange] = []
        self.last_active = time.time()
        # Named aliases: {"camera 1": "camera_entrance"}, set by the user in chat
        self.named:      dict[str, str] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def append(self, user_text: str, assistant_text: str) -> None:
        """Record a completed exchange and trim to MAX_HISTORY."""
        self.history.append(Exchange(user_text, assistant_text))
        if len(self.history) > MAX_HISTORY:
            self.history = self.history[-MAX_HISTORY:]
        self.last_active = time.time()

    def build_messages_prefix(self) -> list[dict]:
        """
        Return a list of prior {role, content} dicts suitable for prepending
        to a Claude messages[] array.

        The caller appends the current user message *after* this prefix.
        Images are never included — only the text of assistant responses.
        """
        msgs: list[dict] = []
        for ex in self.history:
            msgs.append({"role": "user",      "content": ex.user_text})
            msgs.append({"role": "assistant", "content": ex.assistant_text})
        return msgs

    def is_expired(self) -> bool:
        return time.time() - self.last_active > SESSION_TTL

    def touch(self) -> None:
        """Update last_active without appending an exchange."""
        self.last_active = time.time()

    def __repr__(self) -> str:
        return (f"<ConversationSession id={self.session_id!r} "
                f"exchanges={len(self.history)} "
                f"active={int(time.time() - self.last_active)}s ago>")


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STORE
# ═══════════════════════════════════════════════════════════════════════════════

_sessions: OrderedDict[str, ConversationSession] = OrderedDict()
_sessions_lock = threading.Lock()


def _evict_expired() -> None:
    """Remove sessions that have exceeded SESSION_TTL. Must hold _sessions_lock."""
    expired = [sid for sid, s in _sessions.items() if s.is_expired()]
    for sid in expired:
        del _sessions[sid]
    if expired:
        logger.debug(f"[Conversation] Evicted {len(expired)} expired session(s).")


def get_or_create_session(session_id: str) -> ConversationSession:
    """
    Return an existing session or create a new one.

    Thread-safe. Evicts expired sessions on every call (O(n) but n ≤ 100).
    When the cap is reached the oldest session is removed first.
    """
    if not session_id or not isinstance(session_id, str):
        # Caller passed garbage — return a throwaway session (not stored)
        return ConversationSession("_anonymous")

    session_id = session_id[:64]  # guard against absurdly long IDs

    with _sessions_lock:
        _evict_expired()

        if session_id in _sessions:
            sess = _sessions[session_id]
            # Move to end (most-recently-used) for LRU ordering
            _sessions.move_to_end(session_id)
            return sess

        # New session
        if len(_sessions) >= MAX_SESSIONS:
            oldest = next(iter(_sessions))
            del _sessions[oldest]
            logger.debug(f"[Conversation] LRU-evicted session {oldest!r} (cap={MAX_SESSIONS}).")

        sess = ConversationSession(session_id)
        _sessions[session_id] = sess
        logger.debug(f"[Conversation] New session {session_id!r}.")
        return sess


def get_session(session_id: str) -> Optional[ConversationSession]:
    """Return an existing session or None (does not create)."""
    with _sessions_lock:
        return _sessions.get(session_id)


def drop_session(session_id: str) -> None:
    """Explicitly remove a session (e.g., on logout)."""
    with _sessions_lock:
        _sessions.pop(session_id, None)


def active_session_count() -> int:
    with _sessions_lock:
        _evict_expired()
        return len(_sessions)
