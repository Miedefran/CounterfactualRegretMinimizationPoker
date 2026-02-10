import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from gui.components.player_widget import PlayerWidget


class TestPlayerWidgetWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Player Widget Test")
        self.setMinimumSize(1000, 700)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0a1929;
            }
            QLabel {
                color: white;
            }
            QPushButton {
                background-color: #1a5490;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #2a6ab0;
            }
        """)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        title = QLabel("Player Widget Test")
        title.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        layout.addWidget(title)

        players_layout = QHBoxLayout()
        players_layout.setSpacing(20)

        self.player0 = PlayerWidget(0, "Alice")
        self.player0.set_cards(['As', 'Kh'], reveal=True)
        self.player0.set_chips_stack(1000)
        self.player0.set_current_bet(50)
        self.player0.set_current_player(True)
        players_layout.addWidget(self.player0)

        self.player1 = PlayerWidget(1, "Bob")
        self.player1.set_cards(['Qd', 'Jc'], reveal=False)
        self.player1.set_chips_stack(950)
        self.player1.set_current_bet(0)
        players_layout.addWidget(self.player1)

        layout.addLayout(players_layout)

        controls_layout = QHBoxLayout()

        btn1 = QPushButton("Switch Current Player")
        btn1.clicked.connect(self.switch_current)
        controls_layout.addWidget(btn1)

        btn2 = QPushButton("Reveal Player 1 Cards")
        btn2.clicked.connect(lambda: self.player1.set_cards(['Qd', 'Jc'], reveal=True))
        controls_layout.addWidget(btn2)

        btn3 = QPushButton("Fold Player 0")
        btn3.clicked.connect(lambda: self.player0.set_folded(True))
        controls_layout.addWidget(btn3)

        btn4 = QPushButton("Unfold Player 0")
        btn4.clicked.connect(lambda: self.player0.set_folded(False))
        controls_layout.addWidget(btn4)

        btn5 = QPushButton("Set Bet Player 1 (100)")
        btn5.clicked.connect(lambda: self.player1.set_current_bet(100))
        controls_layout.addWidget(btn5)

        btn6 = QPushButton("All-In Player 0")
        btn6.clicked.connect(lambda: self.player0.set_all_in(True))
        controls_layout.addWidget(btn6)

        controls_layout.addStretch()
        layout.addLayout(controls_layout)

        layout.addStretch()

    def switch_current(self):
        current0 = self.player0.is_current_player
        self.player0.set_current_player(not current0)
        self.player1.set_current_player(current0)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TestPlayerWidgetWindow()
    window.show()
    sys.exit(app.exec())
