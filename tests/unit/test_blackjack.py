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

"""Unit tests for the Blackjack Strategy Tutor tools and game logic."""

from unittest.mock import MagicMock

from app.agent import (
    calculate_hand_total,
    check_basic_strategy,
    deal_random_card,
    place_bet,
)


def test_deal_random_card() -> None:
    """Tests that deal_random_card returns valid card data."""
    card_info = deal_random_card()
    assert "card" in card_info
    assert "rank" in card_info
    assert "suit" in card_info
    assert "base_value" in card_info


def test_calculate_hand_total_hard_and_soft() -> None:
    """Tests hand calculation for hard and soft hands."""
    # Hard 17: 10 + 7
    hard_17 = calculate_hand_total(["10 of Hearts", "7 of Spades"])
    assert hard_17["total"] == 17
    assert not hard_17["is_soft"]
    assert not hard_17["is_bust"]

    # Soft 18: Ace + 7
    soft_18 = calculate_hand_total(["Ace of Spades", "7 of Diamonds"])
    assert soft_18["total"] == 18
    assert soft_18["is_soft"]
    assert not soft_18["is_bust"]

    # Bust hand: 10 + 8 + 5 = 23
    bust_hand = calculate_hand_total(["10 of Clubs", "8 of Diamonds", "5 of Hearts"])
    assert bust_hand["total"] == 23
    assert bust_hand["is_bust"]


def test_case_1_hard_17_vs_dealer_10() -> None:
    """Test Case 1: Dealer upcard is a 10 and player hand is a hard 17."""
    player_cards = ["10 of Spades", "7 of Hearts"]
    dealer_upcard = "10 of Diamonds"

    # Hitting on hard 17 vs 10 is mathematically suboptimal
    check_hit = check_basic_strategy(
        player_cards=player_cards,
        dealer_upcard=dealer_upcard,
        proposed_action="hit",
    )
    assert not check_hit["is_optimal"]
    assert check_hit["optimal_action"] == "stand"
    assert "hard 17" in check_hit["reason"].lower()

    # Standing on hard 17 vs 10 is optimal
    check_stand = check_basic_strategy(
        player_cards=player_cards,
        dealer_upcard=dealer_upcard,
        proposed_action="stand",
    )
    assert check_stand["is_optimal"]
    assert check_stand["optimal_action"] == "stand"


def test_case_2_soft_18_vs_dealer_ace() -> None:
    """Test Case 2: Dealer upcard is an Ace and player hand is a soft 18."""
    player_cards = ["Ace of Spades", "7 of Clubs"]
    dealer_upcard = "Ace of Diamonds"

    # Standing on soft 18 vs Ace is mathematically suboptimal
    check_stand = check_basic_strategy(
        player_cards=player_cards,
        dealer_upcard=dealer_upcard,
        proposed_action="stand",
    )
    assert not check_stand["is_optimal"]
    assert check_stand["optimal_action"] == "hit"
    assert "soft 18" in check_stand["reason"].lower()

    # Hitting on soft 18 vs Ace is optimal
    check_hit = check_basic_strategy(
        player_cards=player_cards,
        dealer_upcard=dealer_upcard,
        proposed_action="hit",
    )
    assert check_hit["is_optimal"]
    assert check_hit["optimal_action"] == "hit"


def test_case_3_hitl_triggered_on_large_bet() -> None:
    """Test Case 3: Bet $600 out of $1000 bankroll triggers HITL confirmation."""
    mock_context = MagicMock()
    mock_context.state = {"bankroll": 1000.0}
    mock_context.tool_confirmation = None

    # Bet $600 (> 50% of $1000 bankroll)
    result = place_bet(amount=600.0, tool_context=mock_context)

    # Must trigger HITL
    assert result["status"] == "confirmation_required"
    assert result["hitl_triggered"] is True
    assert result["prompt"] == "Are you sure you want to bet big? Y/N"
    assert result["amount"] == 600.0
    assert result["bankroll"] == 1000.0
    mock_context.request_confirmation.assert_called_once_with(
        hint="Are you sure you want to bet big? Y/N"
    )

    # Standard safe bet ($100 <= 50%) should not trigger HITL
    safe_result = place_bet(amount=100.0, tool_context=mock_context)
    assert safe_result["status"] == "success"
    assert safe_result["bet_amount"] == 100.0
    assert mock_context.state["bankroll"] == 900.0
