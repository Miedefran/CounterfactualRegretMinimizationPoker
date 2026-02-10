import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from gui.components.visual_card import VisualCard


class TestCardWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Card Visual Test")
        self.setMinimumSize(600, 400)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        title = QLabel("Visual Card Test")
        title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        layout.addWidget(title)

        cards_layout = QHBoxLayout()
        cards_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        test_cards = ['Ks', 'Kh', 'Qd', 'Jc', 'As']
        for card in test_cards:
            card_widget = VisualCard(card)
            cards_layout.addWidget(card_widget)

        layout.addLayout(cards_layout)

        hidden_layout = QHBoxLayout()
        hidden_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hidden_card = VisualCard(None)
        hidden_layout.addWidget(hidden_card)
        layout.addLayout(hidden_layout)

        layout.addStretch()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TestCardWindow()
    window.show()
    sys.exit(app.exec())
