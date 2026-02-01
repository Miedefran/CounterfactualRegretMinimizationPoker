from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLabel
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QPalette, QColor


class CardDisplay(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QHBoxLayout(self)
        self.layout.setSpacing(10)
        self.cards = []
        self.setup_style()

    def setup_style(self):
        self.setStyleSheet("""
            QLabel {
                background-color: white;
                border: 2px solid #333;
                border-radius: 8px;
                padding: 10px;
                font-size: 18px;
                font-weight: bold;
                min-width: 60px;
                min-height: 80px;
            }
        """)

    def set_cards(self, cards):
        for card_label in self.cards:
            self.layout.removeWidget(card_label)
            card_label.deleteLater()

        self.cards = []

        if not cards:
            return

        if isinstance(cards, str):
            cards = [cards]

        for card in cards:
            label = QLabel(self.format_card(card))
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.layout.addWidget(label)
            self.cards.append(label)

    def format_card(self, card):
        if card is None:
            return "??"
        if isinstance(card, str):
            return card
        return str(card)

    def add_hidden_card(self):
        label = QLabel("??")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet("""
            QLabel {
                background-color: #1a5490;
                border: 2px solid #333;
                border-radius: 8px;
                padding: 10px;
                font-size: 18px;
                font-weight: bold;
                min-width: 60px;
                min-height: 80px;
                color: white;
            }
        """)
        self.layout.addWidget(label)
        self.cards.append(label)
