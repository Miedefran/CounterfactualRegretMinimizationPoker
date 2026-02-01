from PyQt6.QtCore import QObject, QTimer, pyqtSignal
import requests
from requests.exceptions import Timeout


class HTTPClient(QObject):
    state_update_received = pyqtSignal(dict)
    connection_error = pyqtSignal(str)

    def __init__(self, server_url, player_id=None, player_name: str | None = None, parent=None):
        super().__init__(parent)
        self.server_url = server_url
        self.player_id = player_id
        self.player_name = player_name
        self.poll_timer = QTimer()
        self.poll_timer.timeout.connect(self._poll_state)
        self.poll_timer.setInterval(100)
        self.connected = False

    def connect(self):
        if self.player_id is None:
            try:
                params = {}
                if isinstance(self.player_name, str) and self.player_name.strip():
                    params["name"] = self.player_name.strip()
                response = requests.get(f"{self.server_url}/player_id", params=params, timeout=2)
                self.player_id = response.json()['player_id']
            except Exception as e:
                self.connection_error.emit(str(e))
                return False

        self.connected = True
        self.poll_timer.start()
        return True

    def _poll_state(self):
        if not self.connected:
            return

        try:
            response = requests.get(
                f"{self.server_url}/state",
                params={'player_id': self.player_id},
                timeout=1
            )
            state = response.json()
            self.state_update_received.emit(state)
        except Timeout:
            # transient network hiccup (esp. ngrok). Don't drop the connection.
            return
        except Exception as e:
            self.connection_error.emit(str(e))
            self.poll_timer.stop()
            self.connected = False

    def send_action(self, action, bet_size=0):
        if not self.connected:
            return False

        try:
            requests.post(
                f"{self.server_url}/action",
                json={
                    'player_id': self.player_id,
                    'action': action,
                    'bet_size': bet_size
                },
                timeout=2
            )
            return True
        except Exception as e:
            self.connection_error.emit(str(e))
            return False

    def send_reset_request(self, starting_player=0):
        if not self.connected:
            return False

        try:
            response = requests.post(
                f"{self.server_url}/reset",
                json={'starting_player': starting_player},
                timeout=2
            )

            if response.status_code == 200:
                return True
            else:
                return False
        except Exception as e:
            self.connection_error.emit(str(e))
            return False

    def disconnect(self):
        # Best-effort: tell server to free our player slot immediately.
        if self.connected and self.player_id is not None:
            try:
                requests.post(
                    f"{self.server_url}/disconnect",
                    json={'player_id': int(self.player_id)},
                    timeout=1,
                )
            except Exception:
                pass

        self.connected = False
        self.poll_timer.stop()
