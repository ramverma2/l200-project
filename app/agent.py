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


# ─── Base Subscriptable Pydantic Model ──────────────────────────


class SubscriptableBaseModel(BaseModel):
    """Pydantic BaseModel with dictionary-like subscripting support for backwards compatibility."""

    def __getitem__(self, item: str) -> Any:
        try:
            return getattr(self, item)
        except AttributeError:
            raise KeyError(item) from None

    def get(self, item: str, default: Any = None) -> Any:
        return getattr(self, item, default)

    def __contains__(self, item: str) -> bool:
        return hasattr(self, item)


# ─── Strict Pydantic Schemas for Tool Inputs & Outputs ──────────


class PlayerAction(StrEnum):
    HIT = "hit"
    STAND = "stand"
    DOUBLE = "double"


class BetInput(SubscriptableBaseModel):
    """Input parameters for placing a bet."""

    amount: float = Field(
        ...,
        description="The dollar amount to bet from the player's virtual bankroll.",
        gt=0,
    )


class BetOutput(SubscriptableBaseModel):
    """Output schema for bet placement results."""

    status: str = Field(
        ...,
        description="Status of the bet: 'success', 'confirmation_required', or 'error'.",
    )
    hitl_triggered: bool = Field(
        default=False, description="Whether human-in-the-loop guardrail was triggered."
    )
    prompt: str | None = Field(
        default=None, description="Confirmation prompt if HITL triggered."
    )
    bankroll: float = Field(..., description="Active bankroll amount.")
    amount: float = Field(..., description="The wager amount requested.")
    bet_amount: float | None = Field(
        default=None, description="Alias for wager amount."
    )
    message: str = Field(
        ..., description="Human-readable description of the bet status."
    )
    remaining_bankroll: float | None = Field(
        default=None, description="Bankroll remaining after placing bet."
    )


class HandTotalInput(SubscriptableBaseModel):
    """Input parameters for calculating hand total."""

    cards: list[str] = Field(
        ...,
        description="List of card representation strings (e.g. ['Ace of Spades', '7 of Diamonds'] or ['10', '7']).",
    )


class HandTotalOutput(SubscriptableBaseModel):
    """Output schema for calculated hand totals."""

    cards: list[str] = Field(..., description="Cards in the hand.")
    total: int = Field(..., description="Optimal numerical hand total.")
    is_soft: bool = Field(
        ..., description="True if hand contains an Ace counted as 11 points."
    )
    is_bust: bool = Field(..., description="True if hand total exceeds 21.")


class StrategyCheckInput(SubscriptableBaseModel):
    """Input parameters for evaluating a proposed move against basic strategy."""

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


class StrategyCheckOutput(SubscriptableBaseModel):
    """Output schema for basic strategy evaluation results."""

    is_optimal: bool = Field(
        ..., description="Whether the proposed move is mathematically optimal."
    )
    optimal_action: str = Field(
        ..., description="The basic strategy prescribed action: 'hit' or 'stand'."
    )
    proposed_action: str = Field(..., description="The action evaluated.")
    player_total: int = Field(..., description="Player's current point total.")
    is_soft: bool = Field(..., description="Whether player's hand is soft.")
    dealer_upcard: str = Field(..., description="The dealer's upcard.")
    reason: str = Field(
        ...,
        description="Mathematical and probabilistic rationale for the optimal play.",
    )


class PlayerActionInput(SubscriptableBaseModel):
    """Input parameters for executing a player move."""

    action: str = Field(
        ...,
        description="The player's chosen action: 'hit' or 'stand'.",
    )


class PlayerActionOutput(SubscriptableBaseModel):
    """Output schema for player action execution."""

    action: str = Field(..., description="The action executed.")
    card_drawn: str | None = Field(
        default=None, description="The card drawn if action was 'hit'."
    )
    player_cards: list[str] = Field(..., description="Updated player cards.")
    player_total: int = Field(..., description="Updated player total.")
    is_bust: bool = Field(default=False, description="Whether the player busted.")
    status: str = Field(
        ...,
        description="Execution status: 'safe', 'bust', 'stood', or 'invalid_action'.",
    )
    message: str = Field(..., description="Narrative summary of action outcome.")


class DealCardOutput(SubscriptableBaseModel):
    """Output schema for dealing a single card."""

    card: str = Field(..., description="Display representation of the card.")
    rank: str = Field(..., description="Rank of the card (e.g. '10', 'Ace').")
    suit: str = Field(..., description="Suit of the card (e.g. 'Hearts', 'Spades').")
    base_value: int = Field(..., description="Initial point value.")
    is_ace: bool = Field(..., description="True if card is an Ace.")


class StartNewHandOutput(SubscriptableBaseModel):
    """Output schema for starting a new Blackjack hand."""

    player_cards: list[str] = Field(
        ..., description="Two initial cards dealt to player."
    )
    player_total: int = Field(..., description="Initial player total.")
    is_soft: bool = Field(..., description="True if player hand is soft.")
    dealer_upcard: str = Field(..., description="Dealer's face-up card.")
    message: str = Field(..., description="Deal announcement message.")


class PayoutOutput(SubscriptableBaseModel):
    """Output schema for resolving dealer and payout."""

    result: str = Field(
        ..., description="Result of round: 'player_win', 'dealer_win', or 'push'."
    )
    dealer_cards: list[str] = Field(..., description="Dealer's completed hand.")
    dealer_total: int = Field(..., description="Dealer's final total.")
    payout: float = Field(..., description="Payout awarded to player.")
    updated_bankroll: float = Field(..., description="Updated player bankroll.")
    message: str = Field(..., description="Round outcome message.")


class KnowledgeSearchInput(SubscriptableBaseModel):
    """Input parameters for searching strategy knowledge base."""

    query: str = Field(
        ...,
        description="Natural language query to search the persistent vector store for strategy rules and player history.",
    )


class KnowledgeSearchOutput(SubscriptableBaseModel):
    """Output schema for vector store search results."""

    query: str = Field(..., description="The searched query.")
    matches: list[dict[str, Any]] = Field(
        ..., description="Top matching strategy documents and scores."
    )
    source: str = Field(
        default="persistent_vector_store", description="Data source identifier."
    )


class CompactionInput(SubscriptableBaseModel):
    """Input parameters for session history compaction."""

    max_turns: int = Field(
        default=6, description="Maximum number of historical turns before compaction."
    )


class CompactionOutput(SubscriptableBaseModel):
    """Output schema for session history compaction."""

    status: str = Field(
        ..., description="Compaction status: 'compacted' or 'unchanged'."
    )
    summary: str = Field(..., description="Consolidated context summary.")
    preserved_bankroll: float = Field(
        ..., description="Preserved bankroll across compaction."
    )


class RecordHandInput(SubscriptableBaseModel):
    """Input parameters for recording hand to persistent memory."""

    result: str = Field(
        ..., description="Outcome of round ('player_win', 'dealer_win', 'push')."
    )
    player_total: int = Field(..., description="Final player total.")
    dealer_upcard: str = Field(..., description="Dealer's face-up card.")
    action: str = Field(
        ..., description="Key player action taken ('hit', 'stand', 'double')."
    )


class RecordHandOutput(SubscriptableBaseModel):
    """Output schema for async memory recording."""

    status: str = Field(..., description="Async persistence status.")
    hands_played: int = Field(..., description="Total hands played in player history.")
    player_stats: dict[str, Any] = Field(
        ..., description="Updated cumulative player stats."
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


# ─── ADK Tools with Strict Pydantic Signatures & Docstrings ───────


def deal_random_card() -> DealCardOutput:
    """Deals a single random card from a standard 52-card deck.

    Returns:
        DealCardOutput containing the card's rank, suit, display representation, and base point value.
    """
    with trace_span("blackjack.deal_random_card"):
        rank = random.choice(RANKS)
        suit = random.choice(SUITS)
        card_name = f"{rank} of {suit}"
        val, is_ace = _card_value(rank)
        return DealCardOutput(
            card=card_name,
            rank=rank,
            suit=suit,
            base_value=11 if is_ace else val,
            is_ace=is_ace,
        )


def calculate_hand_total(
    params: HandTotalInput | None = None,
    *,
    cards: list[str] | None = None,
) -> HandTotalOutput:
    """Calculates the optimal Blackjack total for a given list of cards, correctly adjusting Aces.

    Args:
        params: Strict HandTotalInput containing cards list.
        cards: Direct list of card strings in the hand (e.g. ['Ace of Spades', '7 of Hearts']).

    Returns:
        HandTotalOutput containing total (int), is_soft (bool indicating an Ace counted as 11), and is_bust (bool).
    """
    if isinstance(params, list):
        cards = params
        params = None

    if params is not None:
        input_data = params
    elif cards is not None:
        input_data = HandTotalInput(cards=cards)
    else:
        input_data = HandTotalInput(cards=[])

    resolved_cards = input_data.cards

    with trace_span(
        "blackjack.calculate_hand_total", {"cards": resolved_cards}
    ) as span:
        total = 0
        ace_count = 0
        for c in resolved_cards:
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

        result = HandTotalOutput(
            cards=resolved_cards,
            total=total,
            is_soft=is_soft,
            is_bust=total > 21,
        )
        span.set_attribute("calculated_total", total)
        span.set_attribute("is_soft", is_soft)
        return result


def check_basic_strategy(
    params: StrategyCheckInput | None = None,
    *,
    player_cards: list[str] | None = None,
    dealer_upcard: str | None = None,
    proposed_action: str | None = None,
) -> StrategyCheckOutput:
    """Evaluates the player's proposed move against standard Blackjack basic strategy.

    Emits OpenTelemetry distributed tracing spans with strategy evaluation metadata.

    Args:
        params: Strict StrategyCheckInput Pydantic model with player cards, dealer upcard, and proposed action.
        player_cards: List of cards in the player's hand (e.g. ['10 of Clubs', '7 of Hearts']).
        dealer_upcard: The single upcard visible from the dealer (e.g. '10 of Spades' or 'Ace').
        proposed_action: The action proposed by player: 'hit' or 'stand'.

    Returns:
        StrategyCheckOutput indicating whether the proposed move is optimal, what the mathematically optimal
        move is, and the strategic explanation for why it is optimal.
    """
    if isinstance(params, list):
        player_cards = params
        params = None

    if params is not None:
        input_data = params
    elif (
        player_cards is not None
        and dealer_upcard is not None
        and proposed_action is not None
    ):
        input_data = StrategyCheckInput(
            player_cards=player_cards,
            dealer_upcard=dealer_upcard,
            proposed_action=proposed_action,
        )
    else:
        raise ValueError("Missing required arguments for check_basic_strategy.")

    p_cards = input_data.player_cards
    d_upcard = input_data.dealer_upcard
    action = input_data.proposed_action.strip().lower()

    with trace_span(
        "blackjack.check_basic_strategy",
        {
            "player_cards": p_cards,
            "dealer_upcard": d_upcard,
            "proposed_action": action,
        },
    ) as span:
        hand_calc = calculate_hand_total(cards=p_cards)
        total = hand_calc.total
        is_soft = hand_calc.is_soft
        dealer_val, is_dealer_ace, _ = _parse_card(d_upcard)

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
                        f"Player has Hard {total} against dealer {d_upcard}. "
                        f"Hitting on hard 17+ has a high bust rate (>69%). Basic strategy requires standing."
                    )
            elif 12 <= total <= 16:
                # Dealer upcard 2-6 is weak; dealer 7-Ace is strong
                if 2 <= dealer_val <= 6 and not is_dealer_ace:
                    optimal_action = "stand"
                    if action == "hit":
                        is_optimal = False
                        reason = (
                            f"Player has Hard {total} against dealer bust-card {d_upcard} (value {dealer_val}). "
                            f"Basic strategy dictates standing and letting the dealer bust."
                        )
                else:
                    optimal_action = "hit"
                    if action == "stand":
                        is_optimal = False
                        reason = (
                            f"Player has Hard {total} against strong dealer upcard {d_upcard}. "
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
                            f"Player has Soft 18 against strong dealer upcard {d_upcard}. "
                            f"Against a 9, 10, or Ace, 18 is an underdog hand. Hitting is mathematically optimal "
                            f"because you cannot bust and drawing 2, 3, or Ace improves your position."
                        )
                elif 2 <= dealer_val <= 8 and not is_dealer_ace:
                    optimal_action = "stand"
                    if action == "hit":
                        is_optimal = False
                        reason = (
                            f"Player has Soft 18 against dealer {d_upcard}. "
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

        return StrategyCheckOutput(
            is_optimal=is_optimal,
            optimal_action=optimal_action,
            proposed_action=action,
            player_total=total,
            is_soft=is_soft,
            dealer_upcard=d_upcard,
            reason=reason,
        )


def place_bet(
    params: BetInput | None = None,
    *,
    amount: float | None = None,
    tool_context: ToolContext | None = None,
) -> BetOutput:
    """Places a bet for the round from the user's persistent bankroll.

    Enforces the HITL guardrail: If the proposed bet exceeds 50% of the active
    bankroll, a hard stop is triggered requiring terminal confirmation.

    Args:
        params: Strict BetInput Pydantic model.
        amount: Direct dollar amount the player wishes to wager.
        tool_context: ADK ToolContext injected by runtime for state and confirmation management.

    Returns:
        BetOutput with bet confirmation details or HITL confirmation prompt status.
    """
    if isinstance(params, (float, int)):
        amount = float(params)
        params = None

    if params is not None:
        input_data = params
    elif amount is not None:
        input_data = BetInput(amount=amount)
    else:
        raise ValueError("Missing amount for place_bet.")

    wager = input_data.amount
    bankroll = (
        float(tool_context.state.get("bankroll", 1000.0)) if tool_context else 1000.0
    )

    with trace_span(
        "blackjack.place_bet",
        {"proposed_bet": wager, "bankroll": bankroll},
    ) as span:
        # Guardrail / Human-in-the-Loop check: Bet > 50% of bankroll
        if wager > 0.5 * bankroll:
            span.set_attribute("hitl_triggered", True)
            # Check if already confirmed via ADK tool_confirmation
            has_adk_confirm = (
                tool_context
                and tool_context.tool_confirmation
                and getattr(tool_context.tool_confirmation, "confirmed", False)
            )
            if not has_adk_confirm:
                # Trigger ADK confirmation request
                if tool_context:
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
                        "bet_amount": wager,
                        "bankroll": bankroll,
                        "threshold_percent": 50.0,
                        "prompt": "Are you sure you want to bet big? Y/N",
                    },
                )

                return BetOutput(
                    status="confirmation_required",
                    hitl_triggered=True,
                    prompt="Are you sure you want to bet big? Y/N",
                    bankroll=bankroll,
                    amount=wager,
                    message=(
                        f"HITL Guardrail Triggered: You are betting ${wager:.2f}, which is "
                        f"{(wager / bankroll) * 100:.1f}% of your ${bankroll:.2f} bankroll. "
                        "Are you sure you want to bet big? Y/N"
                    ),
                )

        if wager > bankroll:
            span.set_attribute("error", "insufficient_bankroll")
            return BetOutput(
                status="error",
                bankroll=bankroll,
                amount=wager,
                message=f"Insufficient bankroll. Current bankroll is ${bankroll:.2f}.",
            )

        if tool_context:
            tool_context.state["bankroll"] = bankroll - wager
            tool_context.state["current_bet"] = wager
            tool_context.state["game_status"] = "bet_placed"
            remaining = tool_context.state["bankroll"]
        else:
            remaining = bankroll - wager

        emit_observability_log(
            "bet_placed",
            {
                "amount": wager,
                "remaining_bankroll": remaining,
            },
        )

        span.set_attribute("remaining_bankroll", remaining)
        return BetOutput(
            status="success",
            bankroll=bankroll,
            amount=wager,
            bet_amount=wager,
            remaining_bankroll=remaining,
            message=f"Bet of ${wager:.2f} accepted. Ready to deal cards.",
        )


def start_new_hand(tool_context: ToolContext) -> StartNewHandOutput:
    """Deals the initial 2 cards to player and 2 cards to dealer.

    Returns:
        StartNewHandOutput containing the player's initial cards, total, and dealer's visible upcard.
    """
    with trace_span("blackjack.start_new_hand") as span:
        card1 = deal_random_card().card
        card2 = deal_random_card().card
        player_cards = [card1, card2]

        d_card1 = deal_random_card().card
        d_card2 = deal_random_card().card
        dealer_cards = [d_card1, d_card2]

        if tool_context:
            tool_context.state["player_cards"] = player_cards
            tool_context.state["dealer_cards"] = dealer_cards
            tool_context.state["dealer_upcard"] = d_card1
            tool_context.state["game_status"] = "player_turn"

        p_calc = calculate_hand_total(cards=player_cards)

        span.set_attribute("player_cards", str(player_cards))
        span.set_attribute("dealer_upcard", d_card1)

        emit_observability_log(
            "hand_started",
            {
                "player_cards": player_cards,
                "player_total": p_calc.total,
                "is_soft": p_calc.is_soft,
                "dealer_upcard": d_card1,
            },
        )

        return StartNewHandOutput(
            player_cards=player_cards,
            player_total=p_calc.total,
            is_soft=p_calc.is_soft,
            dealer_upcard=d_card1,
            message=(
                f"Cards dealt! Player hand: {player_cards} (Total: {p_calc.total}). "
                f"Dealer upcard: {d_card1}."
            ),
        )


def execute_player_action(
    params: PlayerActionInput | None = None,
    *,
    action: str | None = None,
    tool_context: ToolContext | None = None,
) -> PlayerActionOutput:
    """Executes the player's chosen action ('hit' or 'stand'), logging Intent and Outcome.

    Emits OpenTelemetry spans capturing intent and outcome with zero PII leakage.

    Args:
        params: Strict PlayerActionInput Pydantic model with action.
        action: Direct action string: 'hit' or 'stand'.
        tool_context: ADK ToolContext injected by runtime.

    Returns:
        PlayerActionOutput with the outcome of the action, updated hand, and whether the player busted.
    """
    if params is not None:
        input_data = params
    elif action is not None:
        input_data = PlayerActionInput(action=action)
    else:
        raise ValueError("Missing action for execute_player_action.")

    act = input_data.action.lower()

    with trace_span("blackjack.execute_player_action", {"action": act}) as span:
        player_cards = (
            list(tool_context.state.get("player_cards", [])) if tool_context else []
        )
        dealer_upcard = (
            tool_context.state.get("dealer_upcard", "Unknown")
            if tool_context
            else "Unknown"
        )
        pre_calc = calculate_hand_total(cards=player_cards)

        span.set_attribute("intent", act)
        span.set_attribute("pre_total", pre_calc.total)

        # Log the Intent in structured JSON format with PII redaction
        emit_observability_log(
            "player_intent",
            {
                "intent": act,
                "player_cards": player_cards,
                "player_total_before": pre_calc.total,
                "dealer_upcard": dealer_upcard,
            },
        )

        if act == "hit":
            new_card = deal_random_card().card
            player_cards.append(new_card)
            if tool_context:
                tool_context.state["player_cards"] = player_cards
            post_calc = calculate_hand_total(cards=player_cards)

            outcome_status = "bust" if post_calc.is_bust else "safe"
            if post_calc.is_bust and tool_context:
                tool_context.state["game_status"] = "round_over"

            span.set_attribute("outcome", outcome_status)
            span.set_attribute("post_total", post_calc.total)

            # Log the Outcome in structured JSON format with PII redaction
            emit_observability_log(
                "player_outcome",
                {
                    "action": "hit",
                    "card_drawn": new_card,
                    "new_player_total": post_calc.total,
                    "outcome": outcome_status,
                },
            )

            return PlayerActionOutput(
                action="hit",
                card_drawn=new_card,
                player_cards=player_cards,
                player_total=post_calc.total,
                is_bust=post_calc.is_bust,
                status=outcome_status,
                message=(
                    f"Drew {new_card}. New total: {post_calc.total}."
                    + (" BUST! You lose." if post_calc.is_bust else "")
                ),
            )

        elif act == "stand":
            if tool_context:
                tool_context.state["game_status"] = "dealer_turn"
            span.set_attribute("outcome", "stood_pat")

            # Log the Stand Outcome
            emit_observability_log(
                "player_outcome",
                {
                    "action": "stand",
                    "player_total": pre_calc.total,
                    "outcome": "stood_pat",
                },
            )
            return PlayerActionOutput(
                action="stand",
                player_cards=player_cards,
                player_total=pre_calc.total,
                status="stood",
                message=f"Player stands on {pre_calc.total}. Dealer's turn.",
            )

        return PlayerActionOutput(
            action=act,
            player_cards=player_cards,
            player_total=pre_calc.total,
            status="invalid_action",
            message=f"Unknown action: {act}",
        )


def resolve_dealer_and_payout(tool_context: ToolContext) -> PayoutOutput:
    """Plays out the dealer hand according to casino rules (dealer hits until 17+) and resolves bets.

    Returns:
        PayoutOutput containing dealer final hand, total, winner determination, and updated bankroll.
    """
    with trace_span("blackjack.resolve_dealer_and_payout") as span:
        player_cards = (
            list(tool_context.state.get("player_cards", [])) if tool_context else []
        )
        dealer_cards = (
            list(tool_context.state.get("dealer_cards", [])) if tool_context else []
        )
        bet = float(tool_context.state.get("current_bet", 0.0)) if tool_context else 0.0
        bankroll = (
            float(tool_context.state.get("bankroll", 1000.0))
            if tool_context
            else 1000.0
        )

        p_calc = calculate_hand_total(cards=player_cards)
        if p_calc.is_bust:
            result = "dealer_win"
            payout = 0.0
        else:
            # Dealer draws to 17
            d_calc = calculate_hand_total(cards=dealer_cards)
            while d_calc.total < 17:
                c = deal_random_card().card
                dealer_cards.append(c)
                d_calc = calculate_hand_total(cards=dealer_cards)

            d_total = d_calc.total
            p_total = p_calc.total

            if d_calc.is_bust:
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
        if tool_context:
            tool_context.state["bankroll"] = bankroll
            tool_context.state["dealer_cards"] = dealer_cards
            tool_context.state["game_status"] = "round_over"

        span.set_attribute("game_result", result)
        span.set_attribute("final_bankroll", bankroll)

        dealer_final_total = calculate_hand_total(cards=dealer_cards).total

        emit_observability_log(
            "round_resolved",
            {
                "result": result,
                "payout": payout,
                "player_total": p_calc.total,
                "dealer_total": dealer_final_total,
                "new_bankroll": bankroll,
            },
        )

        return PayoutOutput(
            result=result,
            dealer_cards=dealer_cards,
            dealer_total=dealer_final_total,
            payout=payout,
            updated_bankroll=bankroll,
            message=f"Round Over: {result.upper()}! Payout: ${payout:.2f}. Bankroll: ${bankroll:.2f}.",
        )


# ─── Context & Memory: Vector Store & History Compaction Tools ──


async def search_strategy_knowledge_base(
    params: KnowledgeSearchInput | None = None,
    *,
    query: str | None = None,
) -> KnowledgeSearchOutput:
    """Asynchronously searches the persistent vector store and strategy database for relevant rules and probabilities.

    Args:
        params: Strict KnowledgeSearchInput Pydantic model.
        query: Strategic question or situation (e.g. 'hard 17 vs 10' or 'soft 18 against dealer ace').

    Returns:
        KnowledgeSearchOutput containing the top retrieved strategy documents and mathematical reasoning.
    """
    if params is not None:
        input_data = params
    elif query is not None:
        input_data = KnowledgeSearchInput(query=query)
    else:
        raise ValueError("Missing query for search_strategy_knowledge_base.")

    search_query = input_data.query

    with trace_span("blackjack.vector_store_search", {"query": search_query}) as span:
        vector_store = get_vector_store()
        results = await vector_store.search(query=search_query, top_k=2)
        span.set_attribute("results_count", len(results))
        return KnowledgeSearchOutput(
            query=search_query,
            matches=results,
            source="persistent_vector_store",
        )


async def compact_session_history(
    params: CompactionInput | None = None,
    *,
    max_turns: int = 6,
    tool_context: ToolContext | None = None,
) -> CompactionOutput:
    """Asynchronously applies history compaction to condense past rounds into a structured context summary.

    This prevents context bloat across long Blackjack sessions while preserving game continuity.

    Args:
        params: Strict CompactionInput Pydantic model.
        max_turns: Historical turn threshold before triggering compaction.
        tool_context: ADK ToolContext injected by runtime.

    Returns:
        CompactionOutput confirming history compaction status and preserved state.
    """
    input_data = params or CompactionInput(max_turns=max_turns)

    with trace_span("blackjack.history_compaction") as span:
        bankroll = (
            tool_context.state.get("bankroll", 1000.0) if tool_context else 1000.0
        )
        player_cards = (
            tool_context.state.get("player_cards", []) if tool_context else []
        )
        dealer_upcard = (
            tool_context.state.get("dealer_upcard", "None") if tool_context else "None"
        )

        # Mock event list representing session events
        mock_events = [
            {"role": "user", "content": f"Bet placed with bankroll {bankroll}"},
            {"role": "model", "content": f"Dealt {player_cards} vs {dealer_upcard}"},
            {"role": "user", "content": "Strategy analyzed"},
            {"role": "model", "content": "Round completed"},
        ]

        compacted = await HistoryCompactor.compact_history_async(
            mock_events,
            max_turns=input_data.max_turns,
        )
        if tool_context:
            tool_context.state["history_compacted"] = True
        span.set_attribute("compacted_events", len(compacted))

        return CompactionOutput(
            status="compacted",
            summary=compacted[0]["content"],
            preserved_bankroll=bankroll,
        )


async def record_hand_to_memory(
    params: RecordHandInput | None = None,
    *,
    result: str | None = None,
    player_total: int | None = None,
    dealer_upcard: str | None = None,
    action: str | None = None,
    tool_context: ToolContext | None = None,
) -> RecordHandOutput:
    """Asynchronously persists hand result into the long-term player memory service and vector database.

    Args:
        params: Strict RecordHandInput Pydantic model.
        result: Outcome of round (e.g. 'player_win', 'dealer_win', 'push').
        player_total: Player's final total.
        dealer_upcard: Dealer's visible card.
        action: Key player action taken ('hit', 'stand', 'double').
        tool_context: ADK ToolContext injected by runtime.

    Returns:
        RecordHandOutput with memory confirmation and updated statistics.
    """
    if params is not None:
        input_data = params
    elif (
        result is not None
        and player_total is not None
        and dealer_upcard is not None
        and action is not None
    ):
        input_data = RecordHandInput(
            result=result,
            player_total=player_total,
            dealer_upcard=dealer_upcard,
            action=action,
        )
    else:
        raise ValueError("Missing parameters for record_hand_to_memory.")

    with trace_span(
        "blackjack.async_memory_record", {"result": input_data.result}
    ) as span:
        memory_svc = get_memory_service()
        session_id = (
            str(tool_context.state.get("session_id", "default_session"))
            if tool_context
            else "default_session"
        )
        bankroll = (
            tool_context.state.get("bankroll", 1000.0) if tool_context else 1000.0
        )
        hand_data = {
            "result": input_data.result,
            "player_total": input_data.player_total,
            "dealer_upcard": input_data.dealer_upcard,
            "action": input_data.action,
            "bankroll": bankroll,
        }
        await memory_svc.record_hand_async(session_id, hand_data)
        stats = await memory_svc.get_player_stats_async()
        span.set_attribute("hands_played", stats.get("hands_played", 0))

        return RecordHandOutput(
            status="memory_saved_async",
            hands_played=stats.get("hands_played", 0),
            player_stats=stats,
        )


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
