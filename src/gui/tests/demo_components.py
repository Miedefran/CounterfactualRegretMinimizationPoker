import sys
import os
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from gui.components.poker_table import PokerTable
from gui.components.hidden_card import HiddenCard
from gui.components.visual_card import VisualCard


class TestComponentsWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Component Test - Poker Table & Cards")
        self.setMinimumSize(1000, 700)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0a1929;
            }
            QLabel {
                color: white;
            }
        """)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        title = QLabel("Component Test")
        title.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        layout.addWidget(title)

        table_container = QWidget()
        table_container.setStyleSheet("background-color: #0a1929;")
        table_layout = QVBoxLayout(table_container)
        table_layout.setContentsMargins(50, 50, 50, 50)

        self.poker_table = PokerTable()
        table_layout.addWidget(self.poker_table)

        layout.addWidget(table_container, 2)

        cards_section = QHBoxLayout()
        cards_section.setAlignment(Qt.AlignmentFlag.AlignCenter)

        cards_label = QLabel("Visible Cards:")
        cards_section.addWidget(cards_label)

        test_cards = ['Ks', 'Kh', 'Qd']
        for card in test_cards:
            card_widget = VisualCard(card)
            cards_section.addWidget(card_widget)

        cards_section.addStretch()

        hidden_label = QLabel("Hidden Card:")
        cards_section.addWidget(hidden_label)

        hidden_card = HiddenCard()
        cards_section.addWidget(hidden_card)

        layout.addLayout(cards_section)
        layout.addStretch()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TestComponentsWindow()
    window.show()
    sys.exit(app.exec())
