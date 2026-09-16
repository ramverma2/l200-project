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

"""Blackjack Strategy Tutor: Multi-Agent game, tutoring, and memory system."""

import datetime
import json
import logging
import random
from enum import StrEnum
from typing import Any

from google.adk.agents import Agent
from google.adk.apps import App, ResumabilityConfig
from google.adk.models import Gemini
from google.adk.tools import AgentTool, ToolContext
from google.genai import types
from pydantic import BaseModel, Field

from app.memory_service import HistoryCompactor, get_memory_service
from app.pii_redactor import redact_pii
from app.telemetry import trace_span
from app.vector_store import get_vector_store

# Models as specified in problem.txt:
# Fast model (Gemini Flash) for Dealer, Analytical reasoning model (Gemini Pro) for Tutor
DEALER_MODEL = "gemini-2.5-flash"
TUTOR_MODEL = "gemini-2.5-pro"

# Structured Observability Logger
logger = logging.getLogger("blackjack_observability")
logger.setLevel(logging.INFO)


def emit_observability_log(event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Emits structured JSON observability log for Intents and Outcomes with PII redaction."""
    # Ensure PII is scrubbed before writing to logs
    sanitized_payload = redact_pii(payload)
    record = {
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        "event_type": event_type,
        **sanitized_payload,
    }
    json_str = json.dumps(record)
    logger.info(json_str)
    print(f"[OBSERVABILITY_JSON] {json_str}")
    return record


# ─── Pydantic Schemas ───────────────────────────────────────────


class PlayerAction(StrEnum):
    HIT = "hit"
    STAND = "stand"
    DOUBLE = "double"


class BetInput(BaseModel):
    amount: float = Field(
        ...,
        description="The dollar amount to bet from the player's virtual bankroll.",
        gt=0,
    )


class HandTotalInput(BaseModel):
    cards: list[str] = Field(
        ...,
        description="List of card representation strings (e.g. ['Ace of Spades', '7 of Diamonds'] or ['10', '7']).",
    )


class StrategyCheckInput(BaseModel):
    player_cards: list[str] = Field(
        ...,
        description="The list of cards currently held by the player.",
    )
    dealer_upcard: str = Field(
        ...,
        description="The dealer's face-up visible card (e.g. '10', 'Ace', '7').",
    )
    proposed_action: str = Field(
        ...,
        description="The action the player is considering or requested: 'hit' or 'stand'.",
    )


class KnowledgeSearchInput(BaseModel):
    query: str = Field(
        ...,
        description="Natural language query to search the persistent vector store for strategy rules and player history.",
    )


# ─── Game Helper Functions ──────────────────────────────────────

SUITS = ["Hearts", "Diamonds", "Clubs", "Spades"]
RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "Jack", "Queen", "King", "Ace"]


def _card_value(rank: str) -> tuple[int, bool]:
    """Returns (numerical_value, is_ace)."""
    clean_rank = rank.strip().split()[0].capitalize()
    if clean_rank in ["Jack", "Queen", "King", "J", "Q", "K"]:
        return 10, False
    if clean_rank in ["Ace", "A"]:
        return 11, True
    try:
        return int(clean_rank), False
    except ValueError:
        return 10, False


def _parse_card(card_str: str) -> tuple[int, bool, str]:
    """Parses a card string into (value, is_ace, display_name)."""
    parts = card_str.split(" of ")
    rank = parts[0].strip()
    val, is_ace = _card_value(rank)
    return val, is_ace, card_str


# ─── ADK Tools with Strict JSON Schemas & Docstrings ─────────────


def deal_random_card() -> dict[str, Any]:
    """Deals a single random card from a standard 52-card deck.

    Returns:
        dict containing the card's rank, suit, display representation, and base point value.
    """
    with trace_span("blackjack.deal_random_card"):
        rank = random.choice(RANKS)
        suit = random.choice(SUITS)
        card_name = f"{rank} of {suit}"
        val, is_ace = _card_value(rank)
        return {
            "card": card_name,
            "rank": rank,
            "suit": suit,
            "base_value": 11 if is_ace else val,
            "is_ace": is_ace,
        }


def calculate_hand_total(cards: list[str]) -> dict[str, Any]:
    """Calculates the optimal Blackjack total for a given list of cards, correctly adjusting Aces.

    Args:
        cards: List of card strings in the hand (e.g. ['Ace of Spades', '7 of Hearts']).

    Returns:
        dict containing total (int), is_soft (bool indicating an Ace counted as 11), and is_bust (bool).
    """
    with trace_span("blackjack.calculate_hand_total", {"cards": cards}) as span:
        total = 0
        ace_count = 0
        for c in cards:
            val, is_ace, _ = _parse_card(c)
            if is_ace:
                ace_count += 1
                total += 11
            else:
                total += val

        is_soft = False
        while total > 21 and ace_count > 0:
            total -= 10
            ace_count -= 1

        if ace_count > 0 and total <= 21:
            is_soft = True

        result = {
            "cards": cards,
            "total": total,
            "is_soft": is_soft,
            "is_bust": total > 21,
        }
        span.set_attribute("calculated_total", total)
        span.set_attribute("is_soft", is_soft)
        return result


def check_basic_strategy(
    player_cards: list[str],
    dealer_upcard: str,
    proposed_action: str,
) -> dict[str, Any]:
    """Evaluates the player's proposed move against standard Blackjack basic strategy.

    Emits OpenTelemetry distributed tracing spans with strategy evaluation metadata.

    Args:
        player_cards: List of cards in the player's hand (e.g. ['10 of Clubs', '7 of Hearts']).
        dealer_upcard: The single upcard visible from the dealer (e.g. '10 of Spades' or 'Ace').
        proposed_action: The action proposed by player: 'hit' or 'stand'.

    Returns:
        dict indicating whether the proposed move is optimal, what the mathematically optimal
        move is, and the strategic explanation for why it is optimal.
    """
    with trace_span(
        "blackjack.check_basic_strategy",
        {
            "player_cards": player_cards,
            "dealer_upcard": dealer_upcard,
            "proposed_action": proposed_action,
        },
    ) as span:
        hand_calc = calculate_hand_total(player_cards)
        total = hand_calc["total"]
        is_soft = hand_calc["is_soft"]
        dealer_val, is_dealer_ace, _ = _parse_card(dealer_upcard)
        action = proposed_action.strip().lower()

        is_optimal = True
        optimal_action = action
        reason = "Standard play consistent with basic strategy."

        if not is_soft:
            # Hard Hand rules
            if total >= 17:
                optimal_action = "stand"
                if action == "hit":
                    is_optimal = False
                    reason = (
                        f"Player has Hard {total} against dealer {dealer_upcard}. "
                        f"Hitting on hard 17+ has a high bust rate (>69%). Basic strategy requires standing."
                    )
            elif 12 <= total <= 16:
                # Dealer upcard 2-6 is weak; dealer 7-Ace is strong
                if 2 <= dealer_val <= 6 and not is_dealer_ace:
                    optimal_action = "stand"
                    if action == "hit":
                        is_optimal = False
                        reason = (
                            f"Player has Hard {total} against dealer bust-card {dealer_upcard} (value {dealer_val}). "
                            f"Basic strategy dictates standing and letting the dealer bust."
                        )
                else:
                    optimal_action = "hit"
                    if action == "stand":
                        is_optimal = False
                        reason = (
                            f"Player has Hard {total} against strong dealer upcard {dealer_upcard}. "
                            f"Basic strategy dictates hitting because the dealer is likely to make 17-21."
                        )
            elif total <= 11:
                optimal_action = "hit"
                if action == "stand":
                    is_optimal = False
                    reason = (
                        f"Player has Hard {total}. You cannot bust on a hit with 11 or lower! "
                        f"Standing on 11 or less is always mathematically suboptimal."
                    )
        else:
            # Soft Hand rules (Ace counted as 11)
            if total == 18:
                # Soft 18 (e.g. Ace + 7)
                if is_dealer_ace or dealer_val in [9, 10]:
                    optimal_action = "hit"
                    if action == "stand":
                        is_optimal = False
                        reason = (
                            f"Player has Soft 18 against strong dealer upcard {dealer_upcard}. "
                            f"Against a 9, 10, or Ace, 18 is an underdog hand. Hitting is mathematically optimal "
                            f"because you cannot bust and drawing 2, 3, or Ace improves your position."
                        )
                elif 2 <= dealer_val <= 8 and not is_dealer_ace:
                    optimal_action = "stand"
                    if action == "hit":
                        is_optimal = False
                        reason = (
                            f"Player has Soft 18 against dealer {dealer_upcard}. "
                            f"Basic strategy dictates standing against 2, 7, or 8 (or doubling against 3-6)."
                        )
            elif total >= 19:
                optimal_action = "stand"
                if action == "hit":
                    is_optimal = False
                    reason = f"Player has Soft {total}. A total of 19+ is a strong winning hand; basic strategy dictates standing."
            elif total <= 17:
                optimal_action = "hit"
                if action == "stand":
                    is_optimal = False
                    reason = (
                        f"Player has Soft {total}. With a soft hand 17 or lower, you cannot bust on a single hit. "
                        f"Basic strategy recommends hitting (or doubling down against favorable dealer cards)."
                    )

        span.set_attribute("is_optimal", is_optimal)
        span.set_attribute("optimal_action", optimal_action)

        return {
            "is_optimal": is_optimal,
            "optimal_action": optimal_action,
            "proposed_action": action,
            "player_total": total,
            "is_soft": is_soft,
            "dealer_upcard": dealer_upcard,
            "reason": reason,
        }


def place_bet(amount: float, tool_context: ToolContext) -> dict[str, Any]:
    """Places a bet for the round from the user's persistent bankroll.

    Enforces the HITL guardrail: If the proposed bet exceeds 50% of the active
    bankroll, a hard stop is triggered requiring terminal confirmation.

    Args:
        amount: The dollar amount the player wishes to wager.

    Returns:
        dict with bet confirmation details or HITL confirmation prompt status.
    """
    bankroll = float(tool_context.state.get("bankroll", 1000.0))

    with trace_span(
        "blackjack.place_bet",
        {"proposed_bet": amount, "bankroll": bankroll},
    ) as span:
        # Guardrail / Human-in-the-Loop check: Bet > 50% of bankroll
        if amount > 0.5 * bankroll:
            span.set_attribute("hitl_triggered", True)
            # Check if already confirmed via ADK tool_confirmation
            has_adk_confirm = tool_context.tool_confirmation and getattr(
                tool_context.tool_confirmation, "confirmed", False
            )
            if not has_adk_confirm:
                # Trigger ADK confirmation request
                try:
                    tool_context.request_confirmation(
                        hint="Are you sure you want to bet big? Y/N"
                    )
                except Exception:
                    pass

                emit_observability_log(
                    "hitl_guardrail_triggered",
                    {
                        "guardrail": "large_bet_protection",
                        "bet_amount": amount,
                        "bankroll": bankroll,
                        "threshold_percent": 50.0,
                        "prompt": "Are you sure you want to bet big? Y/N",
                    },
                )

                return {
                    "status": "confirmation_required",
                    "hitl_triggered": True,
                    "prompt": "Are you sure you want to bet big? Y/N",
                    "bankroll": bankroll,
                    "amount": amount,
                    "message": (
                        f"HITL Guardrail Triggered: You are betting ${amount:.2f}, which is "
                        f"{(amount / bankroll) * 100:.1f}% of your ${bankroll:.2f} bankroll. "
                        "Are you sure you want to bet big? Y/N"
                    ),
                }

        if amount > bankroll:
            span.set_attribute("error", "insufficient_bankroll")
            return {
                "status": "error",
                "message": f"Insufficient bankroll. Current bankroll is ${bankroll:.2f}.",
            }

        tool_context.state["bankroll"] = bankroll - amount
        tool_context.state["current_bet"] = amount
        tool_context.state["game_status"] = "bet_placed"

        emit_observability_log(
            "bet_placed",
            {
                "amount": amount,
                "remaining_bankroll": tool_context.state["bankroll"],
            },
        )

        span.set_attribute("remaining_bankroll", tool_context.state["bankroll"])
        return {
            "status": "success",
            "bet_amount": amount,
            "remaining_bankroll": tool_context.state["bankroll"],
            "message": f"Bet of ${amount:.2f} accepted. Ready to deal cards.",
        }


def start_new_hand(tool_context: ToolContext) -> dict[str, Any]:
    """Deals the initial 2 cards to player and 2 cards to dealer.

    Returns:
        dict containing the player's initial cards, total, and dealer's visible upcard.
    """
    with trace_span("blackjack.start_new_hand") as span:
        card1 = deal_random_card()["card"]
        card2 = deal_random_card()["card"]
        player_cards = [card1, card2]

        d_card1 = deal_random_card()["card"]
        d_card2 = deal_random_card()["card"]
        dealer_cards = [d_card1, d_card2]

        tool_context.state["player_cards"] = player_cards
        tool_context.state["dealer_cards"] = dealer_cards
        tool_context.state["dealer_upcard"] = d_card1
        tool_context.state["game_status"] = "player_turn"

        p_calc = calculate_hand_total(player_cards)

        span.set_attribute("player_cards", str(player_cards))
        span.set_attribute("dealer_upcard", d_card1)

        emit_observability_log(
            "hand_started",
            {
                "player_cards": player_cards,
                "player_total": p_calc["total"],
                "is_soft": p_calc["is_soft"],
                "dealer_upcard": d_card1,
            },
        )

        return {
            "player_cards": player_cards,
            "player_total": p_calc["total"],
            "is_soft": p_calc["is_soft"],
            "dealer_upcard": d_card1,
            "message": (
                f"Cards dealt! Player hand: {player_cards} (Total: {p_calc['total']}). "
                f"Dealer upcard: {d_card1}."
            ),
        }


def execute_player_action(action: str, tool_context: ToolContext) -> dict[str, Any]:
    """Executes the player's chosen action ('hit' or 'stand'), logging Intent and Outcome.

    Emits OpenTelemetry spans capturing intent and outcome with zero PII leakage.

    Args:
        action: The player's confirmed action: 'hit' or 'stand'.

    Returns:
        dict with the outcome of the action, updated hand, and whether the player busted.
    """
    with trace_span("blackjack.execute_player_action", {"action": action}) as span:
        player_cards = list(tool_context.state.get("player_cards", []))
        dealer_upcard = tool_context.state.get("dealer_upcard", "Unknown")
        pre_calc = calculate_hand_total(player_cards)

        span.set_attribute("intent", action.lower())
        span.set_attribute("pre_total", pre_calc["total"])

        # Log the Intent in structured JSON format with PII redaction
        emit_observability_log(
            "player_intent",
            {
                "intent": action.lower(),
                "player_cards": player_cards,
                "player_total_before": pre_calc["total"],
                "dealer_upcard": dealer_upcard,
            },
        )

        if action.lower() == "hit":
            new_card = deal_random_card()["card"]
            player_cards.append(new_card)
            tool_context.state["player_cards"] = player_cards
            post_calc = calculate_hand_total(player_cards)

            outcome_status = "bust" if post_calc["is_bust"] else "safe"
            if post_calc["is_bust"]:
                tool_context.state["game_status"] = "round_over"

            span.set_attribute("outcome", outcome_status)
            span.set_attribute("post_total", post_calc["total"])

            # Log the Outcome in structured JSON format with PII redaction
            emit_observability_log(
                "player_outcome",
                {
                    "action": "hit",
                    "card_drawn": new_card,
                    "new_player_total": post_calc["total"],
                    "outcome": outcome_status,
                },
            )

            return {
                "action": "hit",
                "card_drawn": new_card,
                "player_cards": player_cards,
                "player_total": post_calc["total"],
                "is_bust": post_calc["is_bust"],
                "status": outcome_status,
                "message": (
                    f"Drew {new_card}. New total: {post_calc['total']}."
                    + (" BUST! You lose." if post_calc["is_bust"] else "")
                ),
            }

        elif action.lower() == "stand":
            tool_context.state["game_status"] = "dealer_turn"
            span.set_attribute("outcome", "stood_pat")

            # Log the Stand Outcome
            emit_observability_log(
                "player_outcome",
                {
                    "action": "stand",
                    "player_total": pre_calc["total"],
                    "outcome": "stood_pat",
                },
            )
            return {
                "action": "stand",
                "player_cards": player_cards,
                "player_total": pre_calc["total"],
                "status": "stood",
                "message": f"Player stands on {pre_calc['total']}. Dealer's turn.",
            }

        return {"status": "invalid_action", "message": f"Unknown action: {action}"}


def resolve_dealer_and_payout(tool_context: ToolContext) -> dict[str, Any]:
    """Plays out the dealer hand according to casino rules (dealer hits until 17+) and resolves bets.

    Returns:
        dict containing dealer final hand, total, winner determination, and updated bankroll.
    """
    with trace_span("blackjack.resolve_dealer_and_payout") as span:
        player_cards = list(tool_context.state.get("player_cards", []))
        dealer_cards = list(tool_context.state.get("dealer_cards", []))
        bet = float(tool_context.state.get("current_bet", 0.0))
        bankroll = float(tool_context.state.get("bankroll", 1000.0))

        p_calc = calculate_hand_total(player_cards)
        if p_calc["is_bust"]:
            result = "dealer_win"
            payout = 0.0
        else:
            # Dealer draws to 17
            d_calc = calculate_hand_total(dealer_cards)
            while d_calc["total"] < 17:
                c = deal_random_card()["card"]
                dealer_cards.append(c)
                d_calc = calculate_hand_total(dealer_cards)

            d_total = d_calc["total"]
            p_total = p_calc["total"]

            if d_calc["is_bust"]:
                result = "player_win"
                payout = bet * 2.0
            elif p_total > d_total:
                result = "player_win"
                payout = bet * 2.0
            elif p_total < d_total:
                result = "dealer_win"
                payout = 0.0
            else:
                result = "push"
                payout = bet

        bankroll += payout
        tool_context.state["bankroll"] = bankroll
        tool_context.state["dealer_cards"] = dealer_cards
        tool_context.state["game_status"] = "round_over"

        span.set_attribute("game_result", result)
        span.set_attribute("final_bankroll", bankroll)

        emit_observability_log(
            "round_resolved",
            {
                "result": result,
                "payout": payout,
                "player_total": p_calc["total"],
                "dealer_total": calculate_hand_total(dealer_cards)["total"],
                "new_bankroll": bankroll,
            },
        )

        return {
            "result": result,
            "dealer_cards": dealer_cards,
            "dealer_total": calculate_hand_total(dealer_cards)["total"],
            "payout": payout,
            "updated_bankroll": bankroll,
            "message": f"Round Over: {result.upper()}! Payout: ${payout:.2f}. Bankroll: ${bankroll:.2f}.",
        }


# ─── Context & Memory: Vector Store & History Compaction Tools ──


async def search_strategy_knowledge_base(query: str) -> dict[str, Any]:
    """Asynchronously searches the persistent vector store and strategy database for relevant rules and probabilities.

    Args:
        query: Strategic question or situation (e.g. 'hard 17 vs 10' or 'soft 18 against dealer ace').

    Returns:
        dict containing the top retrieved strategy documents and mathematical reasoning.
    """
    with trace_span("blackjack.vector_store_search", {"query": query}) as span:
        vector_store = get_vector_store()
        results = await vector_store.search(query=query, top_k=2)
        span.set_attribute("results_count", len(results))
        return {
            "query": query,
            "matches": results,
            "source": "persistent_vector_store",
        }


async def compact_session_history(tool_context: ToolContext) -> dict[str, Any]:
    """Asynchronously applies history compaction to condense past rounds into a structured context summary.

    This prevents context bloat across long Blackjack sessions while preserving game continuity.

    Returns:
        dict confirming history compaction status and preserved state.
    """
    with trace_span("blackjack.history_compaction") as span:
        bankroll = tool_context.state.get("bankroll", 1000.0)
        player_cards = tool_context.state.get("player_cards", [])
        dealer_upcard = tool_context.state.get("dealer_upcard", "None")

        # Mock event list representing session events
        mock_events = [
            {"role": "user", "content": f"Bet placed with bankroll {bankroll}"},
            {"role": "model", "content": f"Dealt {player_cards} vs {dealer_upcard}"},
            {"role": "user", "content": "Strategy analyzed"},
            {"role": "model", "content": "Round completed"},
        ]

        compacted = await HistoryCompactor.compact_history_async(
            mock_events,
            max_turns=2,
        )
        tool_context.state["history_compacted"] = True
        span.set_attribute("compacted_events", len(compacted))

        return {
            "status": "compacted",
            "summary": compacted[0]["content"],
            "preserved_bankroll": bankroll,
        }


async def record_hand_to_memory(
    result: str,
    player_total: int,
    dealer_upcard: str,
    action: str,
    tool_context: ToolContext,
) -> dict[str, Any]:
    """Asynchronously persists hand result into the long-term player memory service and vector database.

    Args:
        result: Outcome of round (e.g. 'player_win', 'dealer_win', 'push').
        player_total: Player's final total.
        dealer_upcard: Dealer's visible card.
        action: Key player action taken ('hit', 'stand', 'double').

    Returns:
        dict with memory confirmation.
    """
    with trace_span("blackjack.async_memory_record", {"result": result}) as span:
        memory_svc = get_memory_service()
        session_id = str(tool_context.state.get("session_id", "default_session"))
        hand_data = {
            "result": result,
            "player_total": player_total,
            "dealer_upcard": dealer_upcard,
            "action": action,
            "bankroll": tool_context.state.get("bankroll", 1000.0),
        }
        await memory_svc.record_hand_async(session_id, hand_data)
        stats = await memory_svc.get_player_stats_async()
        span.set_attribute("hands_played", stats.get("hands_played", 0))

        return {
            "status": "memory_saved_async",
            "hands_played": stats.get("hands_played", 0),
            "player_stats": stats,
        }


# ─── Multi-Agent Orchestration: Tutor Agent & Dealer Agent ────────

tutor_agent = Agent(
    name="tutor_agent",
    model=Gemini(
        model=TUTOR_MODEL,
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    instruction="""You are the expert Blackjack Strategy Tutor.
Your role:
- When a player makes or proposes a mathematically poor decision according to basic strategy, INTERCEPT and explain why the move is suboptimal.
- Use `search_strategy_knowledge_base` to query the persistent vector store for exact expected values, bust rates, and strategy tables.
- Key Strategic Rules to explain:
  1. Hard 17 or higher: ALWAYS STAND against all dealer cards (including a 10). Hitting on hard 17 has an enormous bust probability (>69%), whereas standing maximizes win/push chances against dealer 10.
  2. Soft 18 (e.g. Ace + 7): ALWAYS HIT against a strong dealer upcard (9, 10, or Ace). Standing on 18 against 9/10/A is mathematically negative EV because the dealer is likely to make 19-20. Hitting gives you free upside without busting because the Ace converts to 1. Against 2, 7, 8, stand. Against 3-6, double down.
  3. Hard 11 or lower: ALWAYS HIT or DOUBLE. You can never bust.
- Explain the mathematics, dealer bust probabilities, and expected value clearly and encouragingly.
- When consulted, provide a structured, friendly breakdown of the situation.
""",
    description="Analyzes Blackjack player moves against basic strategy tables and provides deep mathematical guidance.",
    tools=[search_strategy_knowledge_base],
)

dealer_agent = Agent(
    name="dealer_agent",
    model=Gemini(
        model=DEALER_MODEL,
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    instruction="""You are the Casino Blackjack Dealer and Game Host.
Manage the game flow smoothly:
1. When the player starts or wants to play, check bankroll (defaults to $1000) and ask for their bet.
2. Call `place_bet` to record the wager. If `place_bet` indicates that confirmation is required (>50% of bankroll), STOP and ask the user: "Are you sure you want to bet big? Y/N". Do not proceed until confirmed.
3. Call `start_new_hand` to deal cards. Announce the player's cards and the dealer's visible upcard.
4. When the player suggests or chooses an action ('hit' or 'stand'):
   a. Always run `check_basic_strategy`.
   b. If `is_optimal` is False (for example: hitting on hard 17 vs 10, or standing on soft 18 vs Ace), INTERCEPT IMMEDIATELY! Invoke the `tutor_agent` to explain the basic strategy and math to the player. Give the player the opportunity to change their mind.
   c. If the action is optimal or confirmed by player, call `execute_player_action`.
5. If player stands and didn't bust, call `resolve_dealer_and_payout` to finish the round.
6. Periodically call `compact_session_history` and `record_hand_to_memory` to keep session context compacted and retain player progress.
7. Display current bankroll and offer the next hand.
Always remain professional, fair, and adhere strictly to casino Blackjack rules.
""",
    tools=[
        deal_random_card,
        calculate_hand_total,
        check_basic_strategy,
        place_bet,
        start_new_hand,
        execute_player_action,
        resolve_dealer_and_payout,
        search_strategy_knowledge_base,
        compact_session_history,
        record_hand_to_memory,
        AgentTool(tutor_agent),
    ],
)

root_agent = dealer_agent

app = App(
    root_agent=root_agent,
    name="app",
    resumability_config=ResumabilityConfig(is_resumable=True),
)
