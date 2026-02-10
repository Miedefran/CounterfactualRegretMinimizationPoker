import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from PyQt6.QtWidgets import QApplication
from gui.layouts.base_poker_layout import BasePokerLayout

if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = BasePokerLayout()
    window.show()

    sys.exit(app.exec())
