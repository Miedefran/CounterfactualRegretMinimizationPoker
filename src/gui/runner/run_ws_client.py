import sys
import argparse
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from PyQt6.QtWidgets import QApplication
from gui.human_vs_human import HumanVsHumanGUI
from gui.server.websocket_client import WebSocketClient


def main():
    parser = argparse.ArgumentParser(description='Start WebSocket Client for Human vs Human mode')
    parser.add_argument('--ip', default='localhost',
                        help='Server IP-Adresse (default: localhost)')
    parser.add_argument('--port', type=int, default=8888,
                        help='Server Port (default: 8888)')
    parser.add_argument('--name', default='Player',
                        help='Player name (default: Player)')
    parser.add_argument('--no-ssl', action='store_true',
                        help='Deaktiviert SSL/WSS (verwendet unverschlüsseltes WS)')

    args = parser.parse_args()

    protocol = "ws" if args.no_ssl else "wss"
    server_url = f"{protocol}://{args.ip}:{args.port}"

    app = QApplication(sys.argv)

    print(f"Connecting to WebSocket server: {server_url}")
    print(f"Player name: {args.name}")

    # Erstelle WebSocket-Client und verbinde
    ws_client = WebSocketClient(server_url, player_name=args.name)
    
    # Erstelle GUI mit WebSocket-Client
    window = HumanVsHumanGUI(server_url, human_name=args.name, client=ws_client)
    window.showMaximized()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
