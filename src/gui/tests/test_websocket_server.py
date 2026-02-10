"""Tests für den WebSocket-Server."""

import asyncio
import json
import socket
import threading
import time

import pytest
import websockets

from envs.kuhn_poker.game import KuhnPokerGame
from gui.server.websocket_server import PokerWebSocketServer


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def ws_server_url():
    """Startet WebSocket-Server im Hintergrund-Thread, liefert ws://localhost:port."""
    port = _free_port()
    game = KuhnPokerGame()
    server = PokerWebSocketServer(game, host="localhost", port=port)
    state = {}

    def run_server():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        server.loop = loop
        stop_future = loop.create_future()
        state["loop"] = loop
        state["stop"] = stop_future
        loop.run_until_complete(server._run_server(stop_future))
        loop.close()

    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    time.sleep(0.4)
    url = f"ws://localhost:{port}"
    yield url
    loop = state.get("loop")
    stop = state.get("stop")
    if loop and stop and not stop.done():
        loop.call_soon_threadsafe(stop.set_result, None)
    thread.join(timeout=2.0)


async def _recv_json(ws):
    raw = await ws.recv()
    return json.loads(raw)


async def _recv_n(ws, n):
    """Empfängt genau n JSON-Nachrichten."""
    out = []
    for _ in range(n):
        out.append(await _recv_json(ws))
    return out


async def test_zwei_clients_verbinden_bekommen_player_id_und_state(ws_server_url):
    """Zwei Clients verbinden sich; jeder bekommt eine player_id und einen gültigen State."""
    async with websockets.connect(ws_server_url) as ws1, websockets.connect(ws_server_url) as ws2:
        msgs1 = await _recv_n(ws1, 3)  # player_id, state, state (broadcast)
        msgs2 = await _recv_n(ws2, 3)

    player_id_msg1 = next(m for m in msgs1 if m.get("type") == "player_id")
    player_id_msg2 = next(m for m in msgs2 if m.get("type") == "player_id")
    assert player_id_msg1["player_id"] == 0
    assert player_id_msg2["player_id"] == 1

    state_msg1 = next(m for m in msgs1 if m.get("type") == "state")
    state_msg2 = next(m for m in msgs2 if m.get("type") == "state")
    for msg in (state_msg1, state_msg2):
        assert "state" in msg
        st = msg["state"]
        assert "legal_actions" in st
        assert "current_player" in st
        assert "done" in st


async def test_reset_broadcast_an_alle_verbundenen_clients(ws_server_url):
    """Ein Client sendet Reset; alle verbundenen Clients bekommen einen neuen State."""
    async with websockets.connect(ws_server_url) as ws1, websockets.connect(ws_server_url) as ws2:
        await _recv_n(ws1, 3)
        await _recv_n(ws2, 3)

        await ws1.send(json.dumps({"type": "reset", "starting_player": 0}))

        msg1 = await _recv_json(ws1)
        msg2 = await _recv_json(ws2)
    assert msg1["type"] == "state"
    assert msg2["type"] == "state"
    assert "state" in msg1 and "state" in msg2
    assert "legal_actions" in msg1["state"] and "legal_actions" in msg2["state"]


async def test_ungültige_aktion_gibt_error(ws_server_url):
    """Client sendet eine ungültige Aktion; Server antwortet mit Fehler."""
    async with websockets.connect(ws_server_url) as ws:
        await _recv_n(ws, 3)
        await ws.send(json.dumps({"type": "action", "action": "invalid_action", "bet_size": 0}))

        msg = await _recv_json(ws)
    assert msg["type"] == "error"
    assert "invalid" in msg.get("message", "").lower() or "Invalid" in msg.get("message", "")


async def test_ungültiges_json_gibt_error(ws_server_url):
    """Client sendet keinen gültigen JSON; Server antwortet mit Invalid-JSON-Fehler."""
    async with websockets.connect(ws_server_url) as ws:
        await _recv_n(ws, 3)
        await ws.send("kein json {")

        msg = await _recv_json(ws)
    assert msg["type"] == "error"
    assert "json" in msg.get("message", "").lower()


async def test_dritter_client_bekommt_server_full(ws_server_url):
    """Sind schon zwei Spieler verbunden, bekommt ein dritter Client 'Server full' und wird abgewiesen."""
    async with websockets.connect(ws_server_url) as ws1, websockets.connect(ws_server_url) as ws2:
        await _recv_n(ws1, 3)
        await _recv_n(ws2, 3)
        async with websockets.connect(ws_server_url) as ws3:
            msg = await _recv_json(ws3)
        assert msg["type"] == "error"
        assert "full" in msg.get("message", "").lower()
