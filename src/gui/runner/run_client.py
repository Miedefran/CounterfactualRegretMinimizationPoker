import sys
import argparse
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from PyQt6.QtWidgets import QApplication
from gui.human_vs_human import HumanVsHumanGUI


def main():
    parser = argparse.ArgumentParser(description='Start Client for Human vs Human mode')
    parser.add_argument('--ip', default='localhost',
                        help='Server IP-Adresse (default: localhost)')
    parser.add_argument('--port', type=int, default=8888,
                        help='Server Port (default: 8888)')
    parser.add_argument('--name', default='Player',
                        help='Player name (default: Player)')
    parser.add_argument('--no-ssl', action='store_true',
                        help='Deaktiviert SSL/HTTPS (verwendet unverschlüsseltes HTTP)')

    args = parser.parse_args()

    protocol = "http" if args.no_ssl else "https"
    server_url = f"{protocol}://{args.ip}:{args.port}"

    app = QApplication(sys.argv)

    print(f"Connecting to server: {server_url}")
    print(f"Player name: {args.name}")

    window = HumanVsHumanGUI(server_url, human_name=args.name)
    window.showMaximized()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
