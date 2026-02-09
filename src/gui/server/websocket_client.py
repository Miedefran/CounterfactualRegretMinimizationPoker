import asyncio
import json
import logging
import time
from typing import Optional
from PyQt6.QtCore import QObject, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import QApplication
from websockets.client import connect
from websockets.exceptions import ConnectionClosed, InvalidURI

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class WebSocketWorker(QThread):
    """Worker-Thread für WebSocket-Verbindung (asyncio-Loop läuft hier)."""
    
    state_received = pyqtSignal(dict)  # State-Dict
    error_occurred = pyqtSignal(str)  # Fehlermeldung
    connected_signal = pyqtSignal(int)  # player_id
    disconnected_signal = pyqtSignal()

    def __init__(self, server_url: str, player_name: Optional[str] = None):
        super().__init__()
        self.server_url = server_url
        self.player_name = player_name
        self.player_id: Optional[int] = None
        self.websocket = None
        self.loop = None
        self.running = False
        self.send_queue = asyncio.Queue()

    def run(self):
        """Startet die asyncio-Event-Loop."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_until_complete(self._connect_and_listen())
        except Exception as e:
            logger.error(f"WebSocket worker error: {e}")
            self.error_occurred.emit(str(e))
        finally:
            self.loop.close()

    async def _connect_and_listen(self):
        """Verbindet zum Server und lauscht auf Nachrichten."""
        try:
            self.websocket = await connect(self.server_url)
            self.running = True

            # Empfange player_id (oder error z.B. Server full)
            msg = await self.websocket.recv()
            data = json.loads(msg)
            if data.get("type") == "error":
                self.error_occurred.emit(data.get("message", "Unknown error"))
                return
            if data.get("type") != "player_id":
                self.error_occurred.emit("Expected player_id, got " + str(data.get("type")))
                return

            self.player_id = data.get("player_id")
            self.connected_signal.emit(self.player_id)

            # Sende Name falls vorhanden
            if self.player_name:
                await self.websocket.send(json.dumps({
                    "type": "register",
                    "name": self.player_name
                }))

            # Empfange initialen State
            msg = await self.websocket.recv()
            data = json.loads(msg)
            if data.get("type") == "state" and data.get("state") is not None:
                self.state_received.emit(data["state"])

            # Starte Task für ausgehende Nachrichten
            send_task = asyncio.create_task(self._send_worker())

            # Lausche auf eingehende Nachrichten
            try:
                async for message in self.websocket:
                    try:
                        data = json.loads(message)
                        msg_type = data.get("type")

                        if msg_type == "state":
                            self.state_received.emit(data.get("state"))
                        elif msg_type == "error":
                            error_msg = data.get("message", "Unknown error")
                            self.error_occurred.emit(error_msg)

                    except json.JSONDecodeError:
                        logger.error("Invalid JSON received")
                    except Exception as e:
                        logger.error(f"Error processing message: {e}")

            finally:
                send_task.cancel()
                try:
                    await send_task
                except asyncio.CancelledError:
                    pass

        except ConnectionClosed:
            logger.info("Connection closed")
            self.disconnected_signal.emit()
        except Exception as e:
            logger.error(f"Connection error: {e}")
            self.error_occurred.emit(str(e))
            self.disconnected_signal.emit()
        finally:
            self.running = False
            if self.websocket:
                try:
                    await self.websocket.close()
                except Exception:
                    pass

    async def _send_worker(self):
        """Verarbeitet ausgehende Nachrichten aus der Queue."""
        while self.running:
            try:
                # Warte auf Nachricht mit Timeout
                message = await asyncio.wait_for(self.send_queue.get(), timeout=0.1)
                if self.websocket and not self.websocket.closed:
                    await self.websocket.send(json.dumps(message))
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error sending message: {e}")

    def send_message(self, message: dict):
        """Fügt eine Nachricht zur Send-Queue hinzu (thread-safe)."""
        # #region agent log
        if message.get("type") == "action":
            try:
                with open("/Users/friedemanndoll/CounterfactualRegretMinimizationPoker/.cursor/debug.log", "a") as f:
                    f.write(json.dumps({"location": "websocket_client:send_message", "message": "queueing action", "data": {"action": message.get("action"), "loop": self.loop is not None, "running": self.running}, "hypothesisId": "H3", "timestamp": int(time.time() * 1000)}) + "\n")
            except Exception:
                pass
        # #endregion
        if self.loop and self.running and not self.loop.is_closed():
            try:
                asyncio.run_coroutine_threadsafe(
                    self.send_queue.put(message),
                    self.loop
                )
            except Exception as e:
                logger.error(f"Error queueing message: {e}")

    async def _request_close(self):
        """Wird im Loop aufgerufen, um Verbindung zu schließen (beendet async for)."""
        self.running = False
        if self.websocket and not getattr(self.websocket, "closed", True):
            try:
                await self.websocket.close()
            except Exception:
                pass

    def stop(self):
        """Stoppt den Worker sauber: schließt WebSocket, dann Loop."""
        self.running = False
        if self.loop and not self.loop.is_closed():
            try:
                asyncio.run_coroutine_threadsafe(self._request_close(), self.loop)
            except Exception:
                pass
            self.loop.call_soon_threadsafe(self.loop.stop)


class WebSocketClient(QObject):
    """WebSocket-Client für Multiplayer Poker (kompatibel mit HTTPClient-Interface)."""
    
    state_update_received = pyqtSignal(dict)
    connection_error = pyqtSignal(str)

    def __init__(self, server_url: str, player_id: Optional[int] = None, player_name: Optional[str] = None, parent=None):
        super().__init__(parent)
        self.server_url = server_url
        self.player_id = player_id
        self.player_name = player_name
        self.connected = False
        self.worker: Optional[WebSocketWorker] = None

    def connect(self) -> bool:
        """Verbindet zum Server. Gibt True zurück wenn erfolgreich."""
        if self.connected:
            return True

        try:
            self.worker = WebSocketWorker(self.server_url, self.player_name)
            self.worker.state_received.connect(self._on_state_received)
            self.worker.error_occurred.connect(self._on_error)
            self.worker.connected_signal.connect(self._on_connected)
            self.worker.disconnected_signal.connect(self._on_disconnected)
            self.worker.start()
            
            # Warte auf Verbindung (Worker verbindet sich asynchron).
            # processEvents nötig, damit connected_signal im GUI-Thread ankommt.
            import time
            for _ in range(50):  # Max 5 Sekunden
                QApplication.processEvents()
                if self.connected:
                    return True
                if not self.worker.isRunning():
                    return False
                time.sleep(0.1)
            
            # Timeout: Verbindung nicht innerhalb von 5 Sekunden hergestellt
            if not self.connected:
                self.connection_error.emit("Connection timeout")
                return False
            return True
        except Exception as e:
            self.connection_error.emit(str(e))
            return False

    def _on_state_received(self, state: dict):
        """Wird aufgerufen wenn State empfangen wurde."""
        self.state_update_received.emit(state)

    def _on_error(self, error_msg: str):
        """Wird aufgerufen bei Fehlern."""
        self.connection_error.emit(error_msg)
        self.connected = False

    def _on_connected(self, player_id: int):
        """Wird aufgerufen wenn Verbindung hergestellt wurde."""
        self.player_id = player_id
        self.connected = True

    def _on_disconnected(self):
        """Wird aufgerufen wenn Verbindung getrennt wurde."""
        self.connected = False

    def send_action(self, action: str, bet_size: int = 0) -> bool:
        """Sendet eine Spieler-Aktion."""
        if not self.connected or not self.worker:
            return False

        try:
            self.worker.send_message({
                "type": "action",
                "action": action,
                "bet_size": bet_size
            })
            return True
        except Exception as e:
            self.connection_error.emit(str(e))
            return False

    def send_reset_request(self, starting_player: int = 0) -> bool:
        """Sendet einen Reset-Request."""
        if not self.connected or not self.worker:
            return False

        try:
            self.worker.send_message({
                "type": "reset",
                "starting_player": starting_player
            })
            return True
        except Exception as e:
            self.connection_error.emit(str(e))
            return False

    def disconnect(self):
        """Trennt die Verbindung."""
        if self.worker:
            self.worker.stop()
            self.worker.wait(2000)  # Warte max 2 Sekunden
            self.worker = None
        self.connected = False
