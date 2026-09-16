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

"""Unit tests for PII redaction, OpenTelemetry tracing, Vector Store, and History Compaction."""

from pathlib import Path

import pytest

from app.memory_service import AsyncBlackjackMemory, HistoryCompactor
from app.pii_redactor import PIIRedactor, redact_pii
from app.telemetry import trace_span
from app.vector_store import PersistentVectorStore


def test_pii_redaction_strings_and_dicts() -> None:
    """Tests that PII (emails, phones, SSNs, credit cards, keys) are scrubbed."""
    raw_text = (
        "Contact me at alice@example.com or call 555-123-4567. "
        "My SSN is 123-45-6789 and card is 4111-2222-3333-4444. "
        "My API key is AIzaSyD9u38XkL019284759283748291029."
    )
    redacted = redact_pii(raw_text)
    assert "alice@example.com" not in redacted
    assert "[REDACTED_EMAIL]" in redacted
    assert "555-123-4567" not in redacted
    assert "[REDACTED_PHONE]" in redacted
    assert "123-45-6789" not in redacted
    assert "[REDACTED_SSN]" in redacted
    assert "4111-2222-3333-4444" not in redacted
    assert "[REDACTED_CREDIT_CARD]" in redacted
    assert "AIzaSyD9" not in redacted
    assert "[REDACTED_API_KEY]" in redacted

    # Test recursive dictionary redaction
    payload = {
        "user": {"email": "john.doe@company.org", "phone": "123-456-7890"},
        "details": ["secret@mail.com", 100],
    }
    redacted_dict = PIIRedactor.redact(payload)
    assert redacted_dict["user"]["email"] == "[REDACTED_EMAIL]"
    assert redacted_dict["user"]["phone"] == "[REDACTED_PHONE]"
    assert redacted_dict["details"][0] == "[REDACTED_EMAIL]"
    assert redacted_dict["details"][1] == 100


@pytest.mark.asyncio
async def test_persistent_vector_store(tmp_path: Path) -> None:
    """Tests semantic search and persistent document indexing in the vector store."""
    store_file = tmp_path / "test_store.json"
    store = PersistentVectorStore(storage_path=store_file)

    # Search pre-populated Blackjack strategy docs
    results = await store.search(query="hard 17 vs dealer 10", top_k=2)
    assert len(results) > 0
    top_doc = results[0]
    assert (
        "hard 17" in top_doc["title"].lower() or "hard 17" in top_doc["content"].lower()
    )

    # Search soft 18 rule
    soft_results = await store.search(query="soft 18 ace", top_k=2)
    assert len(soft_results) > 0

    # Add new document with PII to verify scrubbing before storage
    await store.add_document(
        doc_id="test_player_note",
        title="Note for user@casino.com",
        content="Player bet $500, email user@casino.com, phone 555-555-5555.",
        category="notes",
    )
    assert store_file.exists()
    search_res = await store.search(query="player note", top_k=1)
    assert len(search_res) > 0
    assert "[REDACTED_EMAIL]" in search_res[0]["content"]
    assert "user@casino.com" not in search_res[0]["content"]


@pytest.mark.asyncio
async def test_history_compactor_async() -> None:
    """Tests that history compaction condenses older turns while retaining state."""
    raw_history = [
        {"role": "user", "content": "I bet $50"},
        {"role": "model", "content": "Dealt 10 and 6"},
        {"role": "user", "content": "I hit"},
        {"role": "model", "content": "Drew 5, total 21. Player won $100."},
        {"role": "user", "content": "I bet $100"},
        {"role": "model", "content": "Dealt Ace and 7"},
        {"role": "user", "content": "I stand"},
        {"role": "model", "content": "Round over, dealer won."},
    ]

    # Compaction triggered when history exceeds max_turns=4
    compacted = await HistoryCompactor.compact_history_async(raw_history, max_turns=4)

    # First event must be the consolidated summary
    assert len(compacted) < len(raw_history)
    assert compacted[0]["role"] == "system"
    assert "[HISTORY_COMPACTION_SUMMARY]" in compacted[0]["content"]
    assert "Compacted" in compacted[0]["content"]

    # Recent turns must be preserved verbatim
    assert compacted[-1]["content"] == "Round over, dealer won."


@pytest.mark.asyncio
async def test_async_memory_service(tmp_path: Path) -> None:
    """Tests async memory operations and persistence across sessions."""
    mem_file = tmp_path / "test_memory.json"
    memory_svc = AsyncBlackjackMemory(storage_path=mem_file)

    hand_data = {
        "player_cards": ["10 of Spades", "Jack of Hearts"],
        "player_total": 20,
        "dealer_upcard": "9 of Diamonds",
        "action": "stand",
        "result": "player_win",
        "email": "player@gambling.com",  # Should be redacted
    }

    await memory_svc.record_hand_async(session_id="session_123", hand_data=hand_data)
    stats = await memory_svc.get_player_stats_async()
    assert stats["hands_played"] == 1
    assert stats["wins"] == 1
    assert mem_file.exists()


def test_opentelemetry_tracing_span() -> None:
    """Tests OpenTelemetry distributed tracing spans execution and attribute setting."""
    with trace_span(
        "test.blackjack.span", {"hand": "hard 17", "action": "stand"}
    ) as span:
        assert span is not None
        # Verify span is usable as context manager and doesn't raise
        span.set_attribute("custom_attr", "verified")
