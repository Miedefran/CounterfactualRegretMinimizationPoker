"""Tests für Validation."""

import pytest

from gui.server.validation import (
    validate_player_id,
    validate_action,
    validate_bet_size,
    validate_player_name,
)


def test_ungültige_eingaben_werden_abgelehnt_gültige_akzeptiert():
    """Ungültige Eingaben werden abgelehnt, gültige akzeptiert."""
    ok, val, err = validate_player_id(0)
    assert ok is True and val == 0
    
    ok, val, err = validate_player_id(2)
    assert ok is False and "0 or 1" in err
    
    ok, val, err = validate_player_id(None)
    assert ok is False and "required" in err
    
    ok, err = validate_action("check", ["check", "bet"])
    assert ok is True
    
    ok, err = validate_action("invalid", ["check", "bet"])
    assert ok is False and "check" in err or "bet" in err
    
    ok, val, err = validate_bet_size(50, min_bet=0, max_bet=100)
    assert ok is True and val == 50
    
    ok, val, err = validate_bet_size(150, min_bet=0, max_bet=100)
    assert ok is False and "<=" in err
    
    ok, val, err = validate_player_name("<script>alert(1)</script>")
    assert ok is True
    assert "&lt;" in val
    assert "<script>" not in val
