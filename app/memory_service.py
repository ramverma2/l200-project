# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Async Memory Service and History Compactor for Blackjack Multi-Agent system."""

import asyncio
import datetime
import json
from pathlib import Path
from typing import Any

from app.pii_redactor import redact_pii
from app.vector_store import get_vector_store

DEFAULT_MEMORY_FILE = Path("data/player_memory.json")


class HistoryCompactor:
    """Implements history compaction for multi-turn conversational agents.

    Prunes older conversational rounds into a concise structured state summary
    to prevent context window bloat while preserving strategic learning and bankroll stats.
    """

    @classmethod
    async def compact_history_async(
        cls,
        history: list[dict[str, Any]],
        max_turns: int = 6,
        summary_prefix: str = "[HISTORY_COMPACTION_SUMMARY]",
    ) -> list[dict[str, Any]]:
        """Asynchronously summarizes older turns when history length exceeds max_turns.

        Args:
            history: List of conversation event dictionaries (each with role, content/parts).
            max_turns: Maximum raw events to retain before triggering compaction.
            summary_prefix: Identifying tag for compacted historical summaries.

        Returns:
            A compacted list containing the consolidated historical summary followed by the latest active turns.
        """
        if len(history) <= max_turns:
            return history

        # Split into historical events to compact and recent active events to keep verbatim
        cutoff = len(history) - (max_turns // 2)
        older_events = history[:cutoff]
        recent_events = history[cutoff:]

        # Asynchronously extract key stats and decisions from older events
        player_actions = []
        outcomes = []
        bets = []

        for evt in older_events:
            text = str(evt.get("content", evt.get("parts", "")))
            if "bet" in text.lower():
                bets.append(text)
            if any(act in text.lower() for act in ["hit", "stand", "double", "bust"]):
                player_actions.append(text)
            if any(
                res in text.lower()
                for res in ["player_win", "dealer_win", "push", "won", "lost"]
            ):
                outcomes.append(text)

        summary_text = (
            f"{summary_prefix}: Compacted {len(older_events)} previous interaction turns. "
            f"Key Historical Actions: {len(player_actions)} moves recorded. "
            f"Observed Outcomes: {len(outcomes)} completed hands. "
            f"Player decision tendencies: Basic strategy guidance applied. "
            f"State continuity preserved."
        )

        # Scrub PII from summary
        sanitized_summary = redact_pii(summary_text)

        compacted_event = {
            "role": "system",
            "content": sanitized_summary,
            "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
            "compacted_turns_count": len(older_events),
        }

        return [compacted_event, *recent_events]


class AsyncBlackjackMemory:
    """Async memory management service for tracking long-term player statistics, habits, and sessions."""

    def __init__(self, storage_path: Path = DEFAULT_MEMORY_FILE) -> None:
        self.storage_path = storage_path
        self._lock = asyncio.Lock()
        self.memory_data: dict[str, Any] = {"sessions": {}, "player_stats": {}}
        self._load_sync()

    def _load_sync(self) -> None:
        """Loads persistent player memory from disk."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, encoding="utf-8") as f:
                    self.memory_data = json.load(f)
            except Exception:
                self.memory_data = {"sessions": {}, "player_stats": {}}
        else:
            self.memory_data = {"sessions": {}, "player_stats": {}}

    def _save_sync(self) -> None:
        """Persists player memory to disk."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(self.memory_data, f, indent=2)

    async def record_hand_async(
        self,
        session_id: str,
        hand_data: dict[str, Any],
    ) -> None:
        """Asynchronously records a completed Blackjack hand into long-term memory.

        Args:
            session_id: Unique session ID.
            hand_data: Hand details including player cards, dealer upcard, actions, outcome, bet.
        """
        async with self._lock:
            sanitized_data = redact_pii(hand_data)
            sessions = self.memory_data.setdefault("sessions", {})
            session_history = sessions.setdefault(session_id, [])
            session_history.append(
                {
                    "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
                    **sanitized_data,
                }
            )

            # Update aggregate player stats
            stats = self.memory_data.setdefault("player_stats", {})
            stats["hands_played"] = stats.get("hands_played", 0) + 1
            result = sanitized_data.get("result", "").lower()
            if "player_win" in result or "won" in result:
                stats["wins"] = stats.get("wins", 0) + 1
            elif "dealer_win" in result or "bust" in result or "lost" in result:
                stats["losses"] = stats.get("losses", 0) + 1
            elif "push" in result:
                stats["pushes"] = stats.get("pushes", 0) + 1

            # Persist to disk asynchronously
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._save_sync)

            # Also index into vector store for semantic memory search
            vector_store = get_vector_store()
            summary = (
                f"Session {session_id} Hand: Player had {sanitized_data.get('player_cards')} "
                f"(Total: {sanitized_data.get('player_total')}) vs Dealer {sanitized_data.get('dealer_upcard')}. "
                f"Action taken: {sanitized_data.get('action')}. Result: {sanitized_data.get('result')}."
            )
            await vector_store.add_document(
                doc_id=f"hand_{session_id}_{len(session_history)}",
                title=f"Hand Memory {session_id} #{len(session_history)}",
                content=summary,
                category="player_history",
                tags=["hand_history", session_id],
            )

    async def get_player_stats_async(self) -> dict[str, Any]:
        """Asynchronously retrieves overall player statistics."""
        async with self._lock:
            return dict(self.memory_data.get("player_stats", {}))

    async def search_memory_async(
        self, query: str, top_k: int = 3
    ) -> list[dict[str, Any]]:
        """Asynchronously searches long-term player memory and strategic knowledge base."""
        vector_store = get_vector_store()
        return await vector_store.search(query=query, top_k=top_k)


# Global singleton instance
_memory_instance: AsyncBlackjackMemory | None = None


def get_memory_service() -> AsyncBlackjackMemory:
    """Returns the async memory service singleton."""
    global _memory_instance
    if _memory_instance is None:
        _memory_instance = AsyncBlackjackMemory()
    return _memory_instance
