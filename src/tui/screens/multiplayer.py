import sys
import os
import socket
import subprocess
from typing import Optional
from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal
from textual.widgets import Button, Input, Label, Select, Static
from textual import on
from textual.timer import Timer

# Add src directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Games supported by WebSocket server
# Format: (display_label, value) - value must match what the server expects
SERVER_GAMES = [
    ("Kuhn Poker", "kuhn"),
    ("Leduc Hold'em", "leduc"),
    ("Twelve Card Poker", "twelve_card"),
    ("Rhode Island Hold'em", "rhode_island"),
    ("Royal Hold'em", "royal_holdem"),
    ("Limit Hold'em", "limit_holdem"),
]


def get_local_ip() -> str:
    """Ermittelt die lokale IP-Adresse."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "localhost"


class MultiplayerScreen(Vertical):
    """Multiplayer-Screen für Server- und Client-Management."""

    def __init__(self):
        super().__init__()
        self.server_process: Optional[subprocess.Popen] = None
        self.server_port: int = 8888
        self.server_game: str = "leduc"
        self.server_host: str = "localhost"
        self.local_ip: str = ""
        self.status_timer: Optional[Timer] = None

    def compose(self) -> ComposeResult:
        """Erstellt die UI-Komponenten."""
        # Server-Sektion
        yield Label("Server", classes="section-title")
        with Vertical(classes="server-section"):
            with Horizontal():
                yield Label("Game:", classes="field-label")
                yield Select(SERVER_GAMES, value="leduc", id="select-game", classes="field-input")
            with Horizontal():
                yield Label("Port:", classes="field-label")
                yield Input(value="8888", type="integer", id="input-port", classes="field-input")
            with Horizontal():
                yield Label("Host:", classes="field-label")
                yield Select([
                    ("localhost (nur lokal)", "localhost"),
                    ("0.0.0.0 (alle Interfaces)", "0.0.0.0")
                ], value="localhost", id="select-host", classes="field-input")
            with Horizontal():
                yield Button("Start Server", id="btn-start-server", variant="primary")
                yield Button("Stop Server", id="btn-stop-server", variant="error", disabled=True)
            yield Label("", classes="spacer-small")  # Abstand nach Buttons
            yield Static("Status: Not running", id="status-label")
            yield Static("Local IP: -", id="ip-label")
            yield Static("Connection URL: -", id="url-label")

        yield Label("", classes="spacer")  # Abstand

        # Client-Sektion
        yield Label("Client", classes="section-title")
        with Vertical(classes="client-section"):
            yield Button("Start Local Client", id="btn-start-local", variant="success")
            yield Label("", classes="spacer-small")
            yield Label("Remote Connection:", classes="field-label")
            with Horizontal():
                yield Label("Server IP:", classes="field-label")
                yield Input(placeholder="192.168.0.131", id="input-remote-ip", classes="field-input")
            with Horizontal():
                yield Label("Port:", classes="field-label")
                yield Input(value="8888", type="integer", id="input-remote-port", classes="field-input")
            with Horizontal():
                yield Label("Player Name:", classes="field-label")
                yield Input(value="Player", id="input-player-name", classes="field-input")
            yield Button("Start Remote Client", id="btn-start-remote", variant="primary")

        yield Label("", classes="spacer")  # Abstand

        # Info-Box
        yield Label("Connection Info", classes="section-title")
        with Vertical(classes="info-section"):
            yield Static("", id="info-text")

    def on_mount(self) -> None:
        """Wird beim Mount aufgerufen - initialisiert lokale IP."""
        self.local_ip = get_local_ip()
        ip_label = self.query_one("#ip-label", Static)
        ip_label.update(f"Local IP: {self.local_ip} (für Remote-Verbindung)")
        self._update_info_text()

    def on_unmount(self) -> None:
        """Cleanup beim Unmount - stoppt Server falls noch aktiv."""
        if self.status_timer:
            self.status_timer.stop()
        if self.server_process:
            self.stop_server()

    @on(Select.Changed, "#select-game")
    def on_game_changed(self, event: Select.Changed) -> None:
        """Wird aufgerufen wenn Game geändert wird."""
        if event.value != Select.BLANK:
            self.server_game = str(event.value)

    @on(Input.Changed, "#input-port")
    def on_port_changed(self, event: Input.Changed) -> None:
        """Wird aufgerufen wenn Port geändert wird."""
        try:
            self.server_port = int(event.value) if event.value else 8888
        except ValueError:
            pass

    @on(Select.Changed, "#select-host")
    def on_host_changed(self, event: Select.Changed) -> None:
        """Wird aufgerufen wenn Host geändert wird."""
        if event.value != Select.BLANK:
            self.server_host = str(event.value)
            # Warnung wenn 0.0.0.0 gewählt wird
            if event.value == "0.0.0.0":
                self.notify("WARNING: 0.0.0.0 macht den Server von außen erreichbar!", severity="warning")

    @on(Button.Pressed, "#btn-start-server")
    def on_start_server(self) -> None:
        """Startet den Server."""
        if self.server_process:
            return  # Bereits gestartet

        port_input = self.query_one("#input-port", Input)
        try:
            port = int(port_input.value) if port_input.value else 8888
            self.server_port = port
        except ValueError:
            self.notify("Invalid port number", severity="error")
            return

        self.start_server()

    @on(Button.Pressed, "#btn-stop-server")
    def on_stop_server(self) -> None:
        """Stoppt den Server."""
        self.stop_server()

    @on(Button.Pressed, "#btn-start-local")
    def on_start_local_client(self) -> None:
        """Startet lokalen Client."""
        self.start_local_client()

    @on(Button.Pressed, "#btn-start-remote")
    def on_start_remote_client(self) -> None:
        """Startet Remote-Client."""
        self.start_remote_client()

    def start_server(self) -> None:
        """Startet den WebSocket-Server als Subprozess."""
        project_root = Path(__file__).parent.parent.parent.parent
        script_path = project_root / "src" / "gui" / "runner" / "run_ws_server.py"

        cmd = [
            sys.executable,
            str(script_path),
            "--game", self.server_game,
            "--port", str(self.server_port),
            "--host", self.server_host
        ]

        try:
            self.server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=str(project_root)
            )
            self._update_server_ui(running=True)
            self._start_status_timer()
            self.notify(f"Server started on port {self.server_port}", severity="success")
        except Exception as e:
            self.notify(f"Failed to start server: {e}", severity="error")

    def stop_server(self) -> None:
        """Stoppt den WebSocket-Server."""
        if not self.server_process:
            return

        try:
            self.server_process.terminate()
            self.server_process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.server_process.kill()
            self.server_process.wait()
        except Exception:
            pass

        self.server_process = None
        self._update_server_ui(running=False)
        if self.status_timer:
            self.status_timer.stop()
            self.status_timer = None
        self.notify("Server stopped", severity="info")

    def start_local_client(self) -> None:
        """Startet lokalen Client mit localhost."""
        project_root = Path(__file__).parent.parent.parent.parent
        script_path = project_root / "src" / "gui" / "runner" / "run_ws_client.py"

        cmd = [
            sys.executable,
            str(script_path),
            "--ip", "localhost",
            "--port", str(self.server_port),
            "--name", "Player1"
        ]

        try:
            subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=str(project_root)
            )
            self.notify("Local client started", severity="success")
        except Exception as e:
            self.notify(f"Failed to start client: {e}", severity="error")

    def start_remote_client(self) -> None:
        """Startet Remote-Client mit eingegebener IP/Port."""
        remote_ip_input = self.query_one("#input-remote-ip", Input)
        remote_port_input = self.query_one("#input-remote-port", Input)
        player_name_input = self.query_one("#input-player-name", Input)

        remote_ip = remote_ip_input.value.strip()
        if not remote_ip:
            self.notify("Please enter server IP", severity="error")
            return

        try:
            remote_port = int(remote_port_input.value) if remote_port_input.value else 8888
        except ValueError:
            self.notify("Invalid port number", severity="error")
            return

        player_name = player_name_input.value.strip() or "Player"

        project_root = Path(__file__).parent.parent.parent.parent
        script_path = project_root / "src" / "gui" / "runner" / "run_ws_client.py"

        cmd = [
            sys.executable,
            str(script_path),
            "--ip", remote_ip,
            "--port", str(remote_port),
            "--name", player_name
        ]

        try:
            subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=str(project_root)
            )
            self.notify(f"Remote client started (connecting to {remote_ip}:{remote_port})", severity="success")
        except Exception as e:
            self.notify(f"Failed to start client: {e}", severity="error")

    def _update_server_ui(self, running: bool) -> None:
        """Aktualisiert die Server-UI basierend auf Status."""
        start_btn = self.query_one("#btn-start-server", Button)
        stop_btn = self.query_one("#btn-stop-server", Button)
        status_label = self.query_one("#status-label", Static)
        url_label = self.query_one("#url-label", Static)

        if running:
            start_btn.disabled = True
            stop_btn.disabled = False
            status_label.update("Status: Running")
            # URL basierend auf gewähltem Host
            if self.server_host == "localhost":
                url_label.update(f"Connection URL: ws://localhost:{self.server_port}")
            else:
                url_label.update(f"Connection URL: ws://{self.local_ip}:{self.server_port}")
        else:
            start_btn.disabled = False
            stop_btn.disabled = True
            status_label.update("Status: Not running")
            url_label.update("Connection URL: -")

        self._update_info_text()

    def _start_status_timer(self) -> None:
        """Startet Timer zum Prüfen des Server-Status."""
        if self.status_timer:
            self.status_timer.stop()

        def check_status():
            if self.server_process:
                poll_result = self.server_process.poll()
                if poll_result is not None:
                    # Server wurde beendet
                    self.server_process = None
                    self._update_server_ui(running=False)
                    self.notify("Server stopped unexpectedly", severity="warning")

        self.status_timer = self.set_interval(2.0, check_status)

    def _update_info_text(self) -> None:
        """Aktualisiert Info-Text mit CLI-Befehl für Remote-Verbindung."""
        info_text = self.query_one("#info-text", Static)
        if self.server_process:
            if self.server_host == "localhost":
                info_text.update("Server läuft auf localhost (nur lokal erreichbar).\nFür Remote-Verbindung: Host auf 0.0.0.0 ändern.")
            else:
                cmd = f"python src/gui/runner/run_ws_client.py --ip {self.local_ip} --port {self.server_port} --name Spieler2"
                info_text.update(f"Remote clients connect with:\n{cmd}")
        else:
            info_text.update("Start server to see connection command")
