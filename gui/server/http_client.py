from PyQt6.QtCore import QObject, QTimer, pyqtSignal
import requests

class HTTPClient(QObject):
    state_update_received = pyqtSignal(dict)
    connection_error = pyqtSignal(str)
    
    def __init__(self, server_url, player_id=None, parent=None):
        super().__init__(parent)
        self.server_url = server_url
        self.player_id = player_id
        self.poll_timer = QTimer()
        self.poll_timer.timeout.connect(self._poll_state)
        self.poll_timer.setInterval(100)
        self.connected = False
    
    def connect(self):
        if self.player_id is None:
            try:
                response = requests.get(f"{self.server_url}/player_id", timeout=2)
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
        self.connected = False
        self.poll_timer.stop()

