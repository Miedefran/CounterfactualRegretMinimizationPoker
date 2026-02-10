"""Tests für hand_strength."""

import pytest

from gui.utils.hand_strength import hand_strength_text


def test_leduc_pair_wird_erkannt():
    """Leduc: Zwei gleiche Ränge (Pair) werden korrekt erkannt."""
    result = hand_strength_text(private_cards=["Js"], public_cards=["Jh"], game="leduc")
    assert result == "Pair"


def test_leduc_high_card_wird_erkannt():
    """Leduc: Verschiedene Ränge (High Card) werden korrekt erkannt."""
    result = hand_strength_text(private_cards=["Js"], public_cards=["Qh"], game="leduc")
    assert result == "High Card"
