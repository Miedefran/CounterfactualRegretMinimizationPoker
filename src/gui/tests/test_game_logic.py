"""Tests für PokerGameLogic."""

import pytest
from unittest.mock import patch

from envs.kuhn_poker.game import KuhnPokerGame
from gui.server.game_logic import PokerGameLogic


@pytest.fixture
def game_logic():
    """Erstellt eine frische GameLogic-Instanz für jeden Test."""
    game = KuhnPokerGame()
    return PokerGameLogic(game)


def test_zwei_clients_registrieren_sich_erfolgreich(game_logic):
    """Zwei Clients können sich registrieren; jeder bekommt eine eindeutige player_id."""
    assert game_logic.register_client(0) is True
    assert game_logic.register_client(1) is True
    assert len(game_logic.clients) == 2
    assert 0 in game_logic.clients
    assert 1 in game_logic.clients


def test_dritter_client_wird_abgelehnt(game_logic):
    """Wenn zwei Clients bereits registriert sind, wird ein dritter abgelehnt."""
    assert game_logic.register_client(0) is True
    assert game_logic.register_client(1) is True
    assert game_logic.register_client(0) is False  # Bereits registriert
    assert len(game_logic.clients) == 2


def test_client_mit_name_wird_gespeichert_und_escaped(game_logic):
    """Client registriert sich mit Name; Name wird gespeichert und HTML-escaped."""
    assert game_logic.register_client(0, name="<script>alert(1)</script>") is True
    assert "&lt;" in game_logic.client_names[0]
    assert "<script>" not in game_logic.client_names[0]


def test_reset_setzt_spiel_zurück_und_erhöht_reset_id(game_logic):
    """Reset setzt das Spiel zurück, erhöht reset_id und leert history_events."""
    game_logic.register_client(0)
    game_logic.register_client(1)
    game_logic.reset_game(starting_player=0)
    reset_id_1 = game_logic.reset_id
    assert reset_id_1 == 1
    assert len(game_logic.history_events) == 0
    assert game_logic.game.done is False

    game_logic.reset_game(starting_player=1)
    assert game_logic.reset_id == 2
    assert len(game_logic.history_events) == 0


def test_gültige_aktion_wird_ausgeführt_spieler_wechselt(game_logic):
    """Spieler führt gültige Aktion aus; State ändert sich, current_player wechselt."""
    game_logic.register_client(0)
    game_logic.register_client(1)
    game_logic.reset_game(starting_player=0)

    state_before = game_logic.get_state_update(0)
    assert state_before["current_player"] == 0
    assert "check" in state_before["legal_actions"]

    success = game_logic.handle_action(player_id=0, action="check", bet_size=0)
    assert success is True

    state_after = game_logic.get_state_update(0)
    assert state_after["current_player"] == 1
    assert len(game_logic.history_events) > 0
    assert game_logic.history_events[-1]["type"] == "action"
    assert game_logic.history_events[-1]["player_id"] == 0
    assert game_logic.history_events[-1]["action"] == "check"


def test_aktion_von_falschem_spieler_wird_abgelehnt(game_logic):
    """Spieler, der nicht dran ist, kann keine Aktion ausführen."""
    game_logic.register_client(0)
    game_logic.register_client(1)
    game_logic.reset_game(starting_player=0)

    success = game_logic.handle_action(player_id=1, action="check", bet_size=0)
    assert success is False


def test_ungültige_aktion_wird_abgelehnt(game_logic):
    """Eine nicht-legale Aktion wird abgelehnt."""
    game_logic.register_client(0)
    game_logic.register_client(1)
    game_logic.reset_game(starting_player=0)

    success = game_logic.handle_action(player_id=0, action="invalid_action", bet_size=0)
    assert success is False


def test_aktion_nach_spiel_ende_wird_abgelehnt(game_logic):
    """Nach Spielende können keine Aktionen mehr ausgeführt werden."""
    game_logic.register_client(0)
    game_logic.register_client(1)
    game_logic.reset_game(starting_player=0)

    game_logic.handle_action(0, "check", 0)
    game_logic.handle_action(1, "check", 0)

    assert game_logic.game.done is True
    success = game_logic.handle_action(player_id=0, action="check", bet_size=0)
    assert success is False


def test_state_update_enthält_wichtige_felder(game_logic):
    """State-Update enthält alle wichtigen Felder: private_cards, legal_actions, current_player, etc."""
    game_logic.register_client(0, name="Alice")
    game_logic.register_client(1, name="Bob")
    game_logic.reset_game(starting_player=0)

    state = game_logic.get_state_update(0)
    assert "private_cards" in state
    assert "legal_actions" in state
    assert "current_player" in state
    assert "done" in state
    assert "reset_id" in state
    assert "history_events" in state
    assert "player_names" in state
    assert state["player_names"][0] == "Alice"
    assert state["player_names"][1] == "Bob"


def test_spiel_ende_enthält_payoffs(game_logic):
    """Wenn das Spiel endet, enthält der State Payoffs für beide Spieler."""
    game_logic.register_client(0)
    game_logic.register_client(1)
    game_logic.reset_game(starting_player=0)

    # Spiel zu Ende bringen
    game_logic.handle_action(0, "check", 0)
    game_logic.handle_action(1, "check", 0)

    assert game_logic.game.done is True
    state = game_logic.get_state_update(0)
    assert "payoffs" in state
    assert len(state["payoffs"]) == 2


def test_unregister_entfernt_client(game_logic):
    """Unregister entfernt einen Client; er kann sich danach wieder registrieren."""
    assert game_logic.register_client(0) is True
    assert 0 in game_logic.clients

    game_logic.unregister_client(0)
    assert 0 not in game_logic.clients
    assert 0 not in game_logic.client_names

    assert game_logic.register_client(0) is True
