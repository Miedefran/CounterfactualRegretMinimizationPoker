"""Tests für den HTTP-Server."""

import socket
import threading
import time

import pytest
import requests

from envs.kuhn_poker.game import KuhnPokerGame
from gui.server.http_server import PokerHTTPServer


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def http_server_url():
    """Startet HTTP-Server auf freiem Port, liefert Base-URL."""
    port = _free_port()
    game = KuhnPokerGame()
    server = PokerHTTPServer(game, host="localhost", port=port)
    thread = threading.Thread(target=server.start, daemon=True)
    thread.start()
    time.sleep(0.5)
    yield f"http://localhost:{port}"
    # Daemon-Thread endet mit Prozess, kein explizites Stoppen nötig


def test_player_id_first(http_server_url):
    r = requests.get(f"{http_server_url}/player_id", timeout=2)
    assert r.status_code == 200
    assert r.json()["player_id"] == 0


def test_player_id_second(http_server_url):
    r = requests.get(f"{http_server_url}/player_id", timeout=2)
    assert r.status_code == 200
    assert r.json()["player_id"] == 1


def test_reset(http_server_url):
    r = requests.post(
        f"{http_server_url}/reset",
        json={"starting_player": 0},
        timeout=2,
    )
    assert r.status_code == 200


def test_state_after_reset_has_cards(http_server_url):
    requests.post(f"{http_server_url}/reset", json={"starting_player": 0}, timeout=2)
    r = requests.get(f"{http_server_url}/state?player_id=0", timeout=2)
    assert r.status_code == 200
    state = r.json()
    assert "current_player" in state
    assert "done" in state
    assert "pot" in state
    assert "private_cards" in state
    assert len(state["private_cards"]) > 0


def test_action_then_state_updates(http_server_url):
    requests.post(f"{http_server_url}/reset", json={"starting_player": 0}, timeout=2)
    r1 = requests.post(
        f"{http_server_url}/action",
        json={"player_id": 0, "action": "bet", "bet_size": 0},
        timeout=2,
    )
    assert r1.status_code == 200
    r2 = requests.get(f"{http_server_url}/state?player_id=0", timeout=2)
    assert r2.status_code == 200
    assert r2.json().get("current_player") == 1


def test_action_wrong_player_rejected(http_server_url):
    requests.post(f"{http_server_url}/reset", json={"starting_player": 0}, timeout=2)
    requests.post(
        f"{http_server_url}/action",
        json={"player_id": 0, "action": "bet", "bet_size": 0},
        timeout=2,
    )
    r = requests.post(
        f"{http_server_url}/action",
        json={"player_id": 0, "action": "bet", "bet_size": 0},
        timeout=2,
    )
    assert r.status_code == 400
