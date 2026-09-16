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

"""Persistent Vector Store & Semantic Search for Blackjack Strategy and Player Knowledge."""

import asyncio
import json
import math
import re
from pathlib import Path
from typing import Any

from app.pii_redactor import redact_pii

DEFAULT_STORE_PATH = Path("data/strategy_vector_store.json")

# Pre-populated Blackjack Strategy Knowledge Base
DEFAULT_STRATEGY_DOCS = [
    {
        "id": "strat_hard_17_vs_10",
        "title": "Hard 17 vs Dealer 10 Strategy Rule",
        "content": (
            "Rule: Always STAND on Hard 17 against any dealer upcard, including a 10. "
            "Mathematics: Hitting on Hard 17 has a bust probability exceeding 69.2% (any 5, 6, 7, 8, 9, 10, J, Q, K busts). "
            "Standing gives a ~29% chance of not losing (dealer busts 21.3%, pushes 7.7%). "
            "Expected value of standing is -0.54, compared to -0.68 for hitting. Standing is the defensive play that minimizes long-term loss."
        ),
        "category": "hard_hands",
        "tags": ["hard 17", "dealer 10", "bust rate", "expected value"],
    },
    {
        "id": "strat_soft_18_vs_ace",
        "title": "Soft 18 vs Dealer Ace Strategy Rule",
        "content": (
            "Rule: Always HIT Soft 18 (Ace + 7) against a dealer Ace, 10, or 9. "
            "Mathematics: Against a strong dealer card like an Ace, standing on 18 has an EV of -0.09 because the dealer makes 19-21 over 75% of the time. "
            "Hitting has ZERO bust risk because the Ace can count as 1. Drawing an Ace, 2, or 3 (31% probability) improves your hand to 19, 20, or 21. "
            "Hitting improves Expected Value to -0.01, significantly reducing your expected loss."
        ),
        "category": "soft_hands",
        "tags": ["soft 18", "dealer ace", "dealer 10", "free hit", "expected value"],
    },
    {
        "id": "strat_soft_18_vs_weak",
        "title": "Soft 18 vs Weak Dealer Cards (2 through 8)",
        "content": (
            "Rule: Double down on Soft 18 against dealer 3 through 6. Stand against dealer 2, 7, and 8. "
            "When dealer shows 3-6, their bust probability is highest (35% to 42%). Doubling capitalizes on their weakness."
        ),
        "category": "soft_hands",
        "tags": ["soft 18", "dealer weak", "double down", "stand"],
    },
    {
        "id": "strat_bankroll_management",
        "title": "Bankroll Management & Large Bet Guardrail Rule",
        "content": (
            "Rule: Standard betting unit should never exceed 1-5% of total bankroll. "
            "Wagers exceeding 50% of the active bankroll expose the player to catastrophic ruin and trigger mandatory Human-In-The-Loop (HITL) confirmation: "
            "'Are you sure you want to bet big? Y/N'. Always preserve capital across multiple rounds."
        ),
        "category": "bankroll",
        "tags": ["bankroll", "guardrail", "hitl", "betting units", "ruin risk"],
    },
    {
        "id": "strat_hard_12_to_16",
        "title": "Hard 12 to 16 Stiff Hands Strategy",
        "content": (
            "Rule: Stand on 12-16 against dealer 2-6 (bust cards). Hit on 12-16 against dealer 7-Ace. "
            "Against dealer 2-6, let the dealer take the risk of busting. Against dealer 7-Ace, dealer is heavily favored to make a pat hand, forcing the player to hit."
        ),
        "category": "hard_hands",
        "tags": ["stiff hands", "hard 12-16", "bust cards", "dealer 7-ace"],
    },
]


def _tokenize(text: str) -> list[str]:
    """Simple lowercase word tokenizer."""
    return re.findall(r"\w+", text.lower())


def _compute_tf_idf_vector(
    tokens: list[str], vocab: dict[str, int], idf: dict[str, float]
) -> dict[int, float]:
    """Computes sparse TF-IDF vector."""
    tf: dict[str, int] = {}
    for t in tokens:
        tf[t] = tf.get(t, 0) + 1
    vec: dict[int, float] = {}
    for word, count in tf.items():
        if word in vocab:
            idx = vocab[word]
            vec[idx] = (count / len(tokens)) * idf.get(word, 1.0)
    # Normalize L2
    norm = math.sqrt(sum(v * v for v in vec.values()))
    if norm > 0:
        for idx in vec:
            vec[idx] /= norm
    return vec


def _cosine_similarity(vec_a: dict[int, float], vec_b: dict[int, float]) -> float:
    """Computes cosine similarity between two normalized sparse vectors."""
    dot = 0.0
    for idx, val in vec_a.items():
        if idx in vec_b:
            dot += val * vec_b[idx]
    return dot


class PersistentVectorStore:
    """Persistent vector store with semantic similarity search and Vertex AI Search compatibility."""

    def __init__(self, storage_path: Path = DEFAULT_STORE_PATH) -> None:
        self.storage_path = storage_path
        self._lock = asyncio.Lock()
        self.documents: list[dict[str, Any]] = []
        self.vocab: dict[str, int] = {}
        self.idf: dict[str, float] = {}
        self._load_or_initialize()

    def _load_or_initialize(self) -> None:
        """Loads index from persistent disk or initializes with default strategy docs."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, encoding="utf-8") as f:
                    data = json.load(f)
                    self.documents = data.get("documents", [])
            except Exception:
                self.documents = list(DEFAULT_STRATEGY_DOCS)
        else:
            self.documents = list(DEFAULT_STRATEGY_DOCS)
            self._save_sync()
        self._rebuild_index()

    def _save_sync(self) -> None:
        """Synchronous file write to ensure storage directory exists."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(
                {"documents": self.documents, "count": len(self.documents)}, f, indent=2
            )

    def _rebuild_index(self) -> None:
        """Builds vocabulary, IDF frequencies, and vector embeddings."""
        doc_count = max(len(self.documents), 1)
        doc_freq: dict[str, int] = {}
        for doc in self.documents:
            words = set(
                _tokenize(
                    f"{doc['title']} {doc['content']} {' '.join(doc.get('tags', []))}"
                )
            )
            for w in words:
                doc_freq[w] = doc_freq.get(w, 0) + 1

        self.vocab = {w: i for i, w in enumerate(sorted(doc_freq.keys()))}
        self.idf = {
            w: math.log((doc_count + 1) / (freq + 1)) + 1.0
            for w, freq in doc_freq.items()
        }

        # Compute document vectors
        for doc in self.documents:
            tokens = _tokenize(
                f"{doc['title']} {doc['content']} {' '.join(doc.get('tags', []))}"
            )
            doc["_vector"] = _compute_tf_idf_vector(tokens, self.vocab, self.idf)

    async def add_document(
        self,
        doc_id: str,
        title: str,
        content: str,
        category: str = "general",
        tags: list[str] | None = None,
    ) -> None:
        """Asynchronously indexes a new document, scrubbing PII first.

        Args:
            doc_id: Unique document identifier.
            title: Document title.
            content: Main text content.
            category: Classification category.
            tags: List of keywords.
        """
        async with self._lock:
            # Scrub PII before saving to persistent vector store
            sanitized_content = redact_pii(content)
            sanitized_title = redact_pii(title)

            doc = {
                "id": doc_id,
                "title": sanitized_title,
                "content": sanitized_content,
                "category": category,
                "tags": tags or [],
            }
            # Remove existing doc with same ID if exists
            self.documents = [d for d in self.documents if d["id"] != doc_id]
            self.documents.append(doc)
            self._rebuild_index()

            # Persist to disk asynchronously
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._save_sync)

    async def search(
        self, query: str, top_k: int = 3, threshold: float = 0.05
    ) -> list[dict[str, Any]]:
        """Asynchronously executes semantic vector similarity search against indexed knowledge base.

        Args:
            query: Natural language search query.
            top_k: Number of highest ranking results to retrieve.
            threshold: Minimum cosine similarity score.

        Returns:
            List of matching documents with similarity scores, sorted by descending relevance.
        """
        async with self._lock:
            sanitized_query = redact_pii(query)
            query_tokens = _tokenize(sanitized_query)
            query_vec = _compute_tf_idf_vector(query_tokens, self.vocab, self.idf)

            scored_docs = []
            for doc in self.documents:
                doc_vec = doc.get("_vector", {})
                score = _cosine_similarity(query_vec, doc_vec)
                if score >= threshold or len(query_tokens) == 0:
                    result = {
                        "id": doc["id"],
                        "title": doc["title"],
                        "content": doc["content"],
                        "category": doc.get("category", "general"),
                        "similarity_score": round(score, 4),
                    }
                    scored_docs.append(result)

            scored_docs.sort(key=lambda x: x["similarity_score"], reverse=True)
            return scored_docs[:top_k]


# Global singleton instance
_vector_store_instance: PersistentVectorStore | None = None


def get_vector_store() -> PersistentVectorStore:
    """Returns the persistent vector store singleton."""
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = PersistentVectorStore()
    return _vector_store_instance
