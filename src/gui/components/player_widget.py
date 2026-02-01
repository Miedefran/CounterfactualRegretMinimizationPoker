from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from gui.components.visual_card import VisualCard


class TopPlayerWidget(QWidget):
    """Player widget for top position - Name above cards"""

    def __init__(self, player_id, player_name=None, parent=None):
        super().__init__(parent)
        self.player_id = player_id
        self.player_name = player_name or f"Player {player_id}"
        self.is_current_player = False
        self.is_folded = False
        self.chips_stack = 0

        self.setup_ui()
        self.setup_style()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(5)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setContentsMargins(0, 0, 0, 0)

        # Name label at top
        self.name_label = QLabel(self.player_name)
        self.name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.name_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        self.name_label.setVisible(True)
        layout.addWidget(self.name_label)

        # Cards
        self.cards_layout = QHBoxLayout()
        self.cards_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cards_layout.setSpacing(5)
        self.cards_layout.setContentsMargins(0, 0, 0, 0)
        self.cards = []
        layout.addLayout(self.cards_layout)

        # Info layout (chips, status)
        self.info_layout = QVBoxLayout()
        self.info_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.info_layout.setContentsMargins(0, 0, 0, 0)

        # Hand label (text-only)
        self.hand_label = QLabel("")
        self.hand_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.hand_label.setFont(QFont("Arial", 11))
        self.hand_label.setStyleSheet("color: #ffd700;")
        self.hand_label.setVisible(False)
        self.info_layout.addWidget(self.hand_label)

        self.chips_label = QLabel("Chips: 0")
        self.chips_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.chips_label.setFont(QFont("Arial", 12))
        self.chips_label.setVisible(False)
        self.info_layout.addWidget(self.chips_label)

        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setFont(QFont("Arial", 10))
        self.status_label.setVisible(False)
        self.info_layout.addWidget(self.status_label)

        layout.addLayout(self.info_layout)

    def set_hand_text(self, text):
        text = (text or "").strip()
        if not text:
            self.hand_label.setText("")
            self.hand_label.setVisible(False)
            return
        self.hand_label.setText(text)
        self.hand_label.setVisible(True)

    def setup_style(self):
        self.setStyleSheet("""
            QWidget {
                background-color: transparent;
                border: none;
                padding: 0px;
            }
            QLabel {
                color: white;
                background-color: transparent;
            }
        """)
        self.update_highlight()

    def set_cards(self, cards, reveal=False):
        for card_widget in self.cards:
            self.cards_layout.removeWidget(card_widget)
            card_widget.deleteLater()

        self.cards = []

        if not cards:
            return

        if isinstance(cards, str):
            cards = [cards]

        for card in cards:
            if reveal:
                card_widget = VisualCard(card)
            else:
                card_widget = VisualCard(None)
            self.cards_layout.addWidget(card_widget)
            self.cards.append(card_widget)

    def set_chips_stack(self, chips):
        self.chips_stack = chips
        self.chips_label.setText(f"Chips: {chips}")

    def set_player_name(self, name):
        self.player_name = name
        if self.is_current_player:
            self.name_label.setText(f"{self.player_name} (Your Turn)")
        else:
            self.name_label.setText(self.player_name)

    def set_current_player(self, is_current):
        self.is_current_player = is_current
        self.update_highlight()
        if is_current:
            self.name_label.setText(f"{self.player_name} (Your Turn)")
        else:
            self.name_label.setText(self.player_name)

    def set_folded(self, folded):
        self.is_folded = folded
        if folded:
            self.status_label.setText("FOLDED")
            self.status_label.setStyleSheet("color: #dc143c; font-weight: bold;")
            self.status_label.setVisible(True)
        else:
            self.status_label.setText("")
            self.status_label.setVisible(False)
        self.update_highlight()

    def set_all_in(self, all_in):
        if all_in:
            self.status_label.setText("ALL-IN")
            self.status_label.setStyleSheet("color: #ffd700; font-weight: bold;")
            self.status_label.setVisible(True)
        elif not self.is_folded:
            self.status_label.setText("")
            self.status_label.setVisible(False)

    def update_highlight(self):
        self.setStyleSheet("""
            QWidget {
                background-color: transparent;
                border: none;
                padding: 0px;
            }
            QLabel {
                color: white;
                background-color: transparent;
            }
        """)


class BottomPlayerWidget(QWidget):
    """Player widget for bottom position - Name below cards"""

    def __init__(self, player_id, player_name=None, parent=None):
        super().__init__(parent)
        self.player_id = player_id
        self.player_name = player_name or f"Player {player_id}"
        self.is_current_player = False
        self.is_folded = False
        self.chips_stack = 0

        self.setup_ui()
        self.setup_style()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(5)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setContentsMargins(0, 0, 0, 0)

        # Cards at top
        self.cards_layout = QHBoxLayout()
        self.cards_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cards_layout.setSpacing(5)
        self.cards_layout.setContentsMargins(0, 0, 0, 0)
        self.cards = []
        layout.addLayout(self.cards_layout)

        # Name label below cards
        self.name_label = QLabel(self.player_name)
        self.name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.name_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        self.name_label.setVisible(True)
        layout.addWidget(self.name_label)

        # Info layout (chips, status)
        self.info_layout = QVBoxLayout()
        self.info_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.info_layout.setContentsMargins(0, 0, 0, 0)

        # Hand label (text-only)
        self.hand_label = QLabel("")
        self.hand_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.hand_label.setFont(QFont("Arial", 11))
        self.hand_label.setStyleSheet("color: #ffd700;")
        self.hand_label.setVisible(False)
        self.info_layout.addWidget(self.hand_label)

        self.chips_label = QLabel("Chips: 0")
        self.chips_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.chips_label.setFont(QFont("Arial", 12))
        self.chips_label.setVisible(False)
        self.info_layout.addWidget(self.chips_label)

        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setFont(QFont("Arial", 10))
        self.status_label.setVisible(False)
        self.info_layout.addWidget(self.status_label)

        layout.addLayout(self.info_layout)

    def set_hand_text(self, text):
        text = (text or "").strip()
        if not text:
            self.hand_label.setText("")
            self.hand_label.setVisible(False)
            return
        self.hand_label.setText(text)
        self.hand_label.setVisible(True)

    def setup_style(self):
        self.setStyleSheet("""
            QWidget {
                background-color: transparent;
                border: none;
                padding: 0px;
            }
            QLabel {
                color: white;
                background-color: transparent;
            }
        """)
        self.update_highlight()

    def set_cards(self, cards, reveal=False):
        for card_widget in self.cards:
            self.cards_layout.removeWidget(card_widget)
            card_widget.deleteLater()

        self.cards = []

        if not cards:
            return

        if isinstance(cards, str):
            cards = [cards]

        for card in cards:
            if reveal:
                card_widget = VisualCard(card)
            else:
                card_widget = VisualCard(None)
            self.cards_layout.addWidget(card_widget)
            self.cards.append(card_widget)

    def set_chips_stack(self, chips):
        self.chips_stack = chips
        self.chips_label.setText(f"Chips: {chips}")

    def set_player_name(self, name):
        self.player_name = name
        if self.is_current_player:
            self.name_label.setText(f"{self.player_name} (Your Turn)")
        else:
            self.name_label.setText(self.player_name)

    def set_current_player(self, is_current):
        self.is_current_player = is_current
        self.update_highlight()
        if is_current:
            self.name_label.setText(f"{self.player_name} (Your Turn)")
        else:
            self.name_label.setText(self.player_name)

    def set_folded(self, folded):
        self.is_folded = folded
        if folded:
            self.status_label.setText("FOLDED")
            self.status_label.setStyleSheet("color: #dc143c; font-weight: bold;")
            self.status_label.setVisible(True)
        else:
            self.status_label.setText("")
            self.status_label.setVisible(False)
        self.update_highlight()

    def set_all_in(self, all_in):
        if all_in:
            self.status_label.setText("ALL-IN")
            self.status_label.setStyleSheet("color: #ffd700; font-weight: bold;")
            self.status_label.setVisible(True)
        elif not self.is_folded:
            self.status_label.setText("")
            self.status_label.setVisible(False)

    def update_highlight(self):
        self.setStyleSheet("""
            QWidget {
                background-color: transparent;
                border: none;
                padding: 0px;
            }
            QLabel {
                color: white;
                background-color: transparent;
            }
        """)
