import asyncio
import json
import logging
import signal
import time
from typing import Optional, Dict
from websockets.server import serve
from websockets.exceptions import ConnectionClosed
from gui.server.game_logic import PokerGameLogic

DEBUG_LOG = "/Users/friedemanndoll/CounterfactualRegretMinimizationPoker/.cursor/debug.log"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PokerWebSocketServer:
    """WebSocket-Server für Multiplayer Poker.
    
    Nutzt PokerGameLogic für die gemeinsame Spiellogik.
    Push-basiert: State wird automatisch an beide Clients gesendet bei Änderungen.
    """
    
    def __init__(self, game, host='0.0.0.0', port=8888, game_id: Optional[str] = None):
        self.host = host
        self.port = port
        self.game_logic = PokerGameLogic(game, game_id=game_id)
        # WebSocket-Verbindungen: player_id -> websocket
        self.connections: Dict[int, any] = {}
        self.server = None
        self.loop = None

    async def _handle_connection(self, websocket, path):
        """Behandelt eine neue WebSocket-Verbindung.
        
        Args:
            websocket: WebSocket-Verbindung
            path: URL-Pfad (wird ignoriert)
        """
        player_id = None
        try:
            # Versuche Spieler zu registrieren
            for pid in range(self.game_logic.max_players):
                if pid not in self.connections:
                    if self.game_logic.register_client(pid):
                        player_id = pid
                        self.connections[player_id] = websocket
                        break
            
            if player_id is None:
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": "Server full (2 players already connected)."
                }))
                await websocket.close()
                return

            # Sende player_id an Client
            await websocket.send(json.dumps({
                "type": "player_id",
                "player_id": player_id
            }))

            # Sende initialen State
            state = self.game_logic.get_state_update(player_id)
            await websocket.send(json.dumps({
                "type": "state",
                "state": state
            }))

            # Broadcast State an beide Clients (falls beide verbunden)
            await self._broadcast_state()

            # Empfange Nachrichten vom Client
            async for message in websocket:
                try:
                    data = json.loads(message)
                    msg_type = data.get("type")

                    if msg_type == "register":
                        # Name setzen
                        name = data.get("name")
                        if isinstance(name, str) and name.strip():
                            with self.game_logic.lock:
                                self.game_logic.client_names[player_id] = name.strip()
                        await self._broadcast_state()

                    elif msg_type == "action":
                        action = data.get("action")
                        bet_size = data.get("bet_size", 0)
                        success = self.game_logic.handle_action(player_id, action, bet_size)
                        # #region agent log
                        try:
                            with open(DEBUG_LOG, "a") as f:
                                f.write(json.dumps({"location": "websocket_server:action", "message": "action handled", "data": {"player_id": player_id, "action": action, "bet_size": bet_size, "success": success}, "hypothesisId": "H4", "timestamp": int(time.time() * 1000)}) + "\n")
                        except Exception:
                            pass
                        # #endregion
                        if success:
                            # State an beide Clients pushen
                            await self._broadcast_state()
                        else:
                            await websocket.send(json.dumps({
                                "type": "error",
                                "message": "Invalid action"
                            }))

                    elif msg_type == "reset":
                        # Spiel zurücksetzen
                        starting_player = data.get("starting_player", 0)
                        self.game_logic.reset_game(starting_player)
                        await self._broadcast_state()

                    elif msg_type == "get_state":
                        # State on-demand senden
                        state = self.game_logic.get_state_update(player_id)
                        await websocket.send(json.dumps({
                            "type": "state",
                            "state": state
                        }))

                except json.JSONDecodeError:
                    await websocket.send(json.dumps({
                        "type": "error",
                        "message": "Invalid JSON"
                    }))
                except Exception as e:
                    logger.error(f"Error handling message from player {player_id}: {e}")
                    await websocket.send(json.dumps({
                        "type": "error",
                        "message": str(e)
                    }))

        except ConnectionClosed:
            logger.info(f"Connection closed for player {player_id}")
        except Exception as e:
            logger.error(f"Error in connection handler: {e}")
        finally:
            # Cleanup
            if player_id is not None:
                self.connections.pop(player_id, None)
                self.game_logic.unregister_client(player_id)
                # Informiere anderen Client über Disconnect
                await self._broadcast_state()

    async def _broadcast_state(self):
        """Sendet State an alle verbundenen Clients."""
        for pid, ws in list(self.connections.items()):
            try:
                state = self.game_logic.get_state_update(pid)
                await ws.send(json.dumps({
                    "type": "state",
                    "state": state
                }))
            except Exception as e:
                logger.error(f"Error sending state to player {pid}: {e}")
                # Verbindung entfernen bei Fehler
                self.connections.pop(pid, None)
                self.game_logic.unregister_client(pid)

    async def _run_server(self, stop_future: asyncio.Future):
        """Startet den WebSocket-Server. Beendet sich, wenn stop_future gesetzt wird."""
        server = await serve(self._handle_connection, self.host, self.port)
        logger.info(f"WebSocket Server läuft auf ws://{self.host}:{self.port}")
        try:
            await stop_future
        finally:
            logger.info("Shutting down server...")
            server.close()
            await server.wait_closed()

    def start(self):
        """Startet den Server (blockierend). Ctrl+C löst sauberen Shutdown aus."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        stop_future = self.loop.create_future()

        def _request_stop(*_args):
            if not stop_future.done():
                stop_future.set_result(None)

        try:
            if hasattr(signal, "SIGINT"):
                self.loop.add_signal_handler(signal.SIGINT, _request_stop)
        except NotImplementedError:
            # Windows: add_signal_handler nicht verfügbar; Handler ggf. in anderem Kontext
            def _sigint_handler(sig, frame):
                self.loop.call_soon_threadsafe(_request_stop)

            signal.signal(signal.SIGINT, _sigint_handler)

        try:
            self.loop.run_until_complete(self._run_server(stop_future))
        except KeyboardInterrupt:
            logger.info("Shutting down server...")
        finally:
            self.loop.close()
