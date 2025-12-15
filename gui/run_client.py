import sys
import argparse
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from PyQt6.QtWidgets import QApplication
from gui.human_vs_human import HumanVsHumanGUI

def main():
    parser = argparse.ArgumentParser(description='Start Client for Human vs Human mode')
    parser.add_argument('--server', required=True,
                       help='Server URL (e.g. http://localhost:8888 or http://192.168.1.100:8888)')
    parser.add_argument('--name', default='Player',
                       help='Player name (default: Player)')
    
    args = parser.parse_args()
    
    app = QApplication(sys.argv)
    
    print(f"Connecting to server: {args.server}")
    print(f"Player name: {args.name}")
    
    window = HumanVsHumanGUI(args.server, human_name=args.name)
    window.showMaximized()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()

