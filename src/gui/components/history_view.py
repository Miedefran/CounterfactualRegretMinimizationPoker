from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QScrollArea
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont


class HistoryView(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.history_items = []
        self.setup_ui()
        self.setup_style()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        title = QLabel("History")
        title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        layout.addWidget(title)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.content_layout.setSpacing(5)

        scroll_area.setWidget(self.content_widget)
        layout.addWidget(scroll_area)

    def setup_style(self):
        self.setStyleSheet("""
            QWidget {
                background-color: rgba(0, 0, 0, 0.3);
                border-radius: 5px;
                padding: 5px;
            }
            QLabel {
                color: white;
                background-color: transparent;
            }
            QScrollArea {
                border: none;
                background-color: transparent;
            }
        """)

    def add_action(self, player_id, action, bet_size=None, player_name=None):
        if isinstance(action, tuple):
            action, bet_size = action

        if player_name is None:
            player_name = f"Player {player_id}"

        action_text = self.format_action(action, bet_size)

        label = QLabel(f"{player_name}: {action_text}")
        label.setFont(QFont("Arial", 11))
        label.setWordWrap(True)
        label.setStyleSheet("""
            QLabel {
                color: #e0e0e0;
                padding: 3px;
                background-color: rgba(255, 255, 255, 0.05);
                border-radius: 3px;
            }
        """)

        self.content_layout.addWidget(label)
        self.history_items.append(label)

        scroll_area = self.findChild(QScrollArea)
        if scroll_area:
            scroll_bar = scroll_area.verticalScrollBar()
            if scroll_bar:
                scroll_bar.setValue(scroll_bar.maximum())

    def format_action(self, action, bet_size=None):
        action_lower = action.lower()

        if action_lower == 'check':
            return "Check"
        elif action_lower == 'bet':
            if bet_size:
                return f"Bet {bet_size}"
            return "Bet"
        elif action_lower == 'call':
            if bet_size:
                return f"Call {bet_size}"
            return "Call"
        elif action_lower == 'fold':
            return "Fold"
        elif action_lower == 'raise':
            if bet_size:
                return f"Raise to {bet_size}"
            return "Raise"
        elif action_lower == '|':
            return "--- New Round ---"
        else:
            if bet_size:
                return f"{action} {bet_size}"
            return str(action)

    def add_round_separator(self, round_name=None):
        separator = QLabel("─" * 30)
        separator.setFont(QFont("Arial", 10))
        separator.setStyleSheet("""
            QLabel {
                color: #888;
                padding: 5px;
            }
        """)
        self.content_layout.addWidget(separator)
        self.history_items.append(separator)

        if round_name:
            round_label = QLabel(f"Round: {round_name}")
            round_label.setFont(QFont("Arial", 11, QFont.Weight.Bold))
            round_label.setStyleSheet("""
                QLabel {
                    color: #ffd700;
                    padding: 3px;
                }
            """)
            self.content_layout.addWidget(round_label)
            self.history_items.append(round_label)

    def add_game_result(self, winner_id, pot_amount, winner_name=None):
        if winner_name is None:
            winner_name = f"Player {winner_id}"
        result_label = QLabel(f"{winner_name} wins {pot_amount}!")
        result_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        result_label.setStyleSheet("""
            QLabel {
                color: #ffd700;
                padding: 5px;
                background-color: rgba(255, 215, 0, 0.2);
                border-radius: 3px;
            }
        """)
        self.content_layout.addWidget(result_label)
        self.history_items.append(result_label)

    def clear(self):
        for item in self.history_items:
            self.content_layout.removeWidget(item)
            item.deleteLater()
        self.history_items = []

    def set_history(self, history_list, player_names=None):
        self.clear()
        if player_names is None:
            player_names = {}

        for entry in history_list:
            player_id, action = entry
            player_name = player_names.get(player_id, f"Player {player_id}")
            self.add_action(player_id, action, player_name=player_name)