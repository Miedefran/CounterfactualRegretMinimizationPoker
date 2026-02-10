import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from gui.components.history_view import HistoryView


class TestHistoryViewWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("History View Test")
        self.setMinimumSize(600, 800)
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

        title = QLabel("History View Test")
        title.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        layout.addWidget(title)

        self.history_view = HistoryView()
        self.history_view.setMaximumHeight(500)
        layout.addWidget(self.history_view)

        controls_layout = QHBoxLayout()

        btn1 = QPushButton("Add: Player 0 Check")
        btn1.clicked.connect(lambda: self.history_view.add_action(0, 'check'))
        controls_layout.addWidget(btn1)

        btn2 = QPushButton("Add: Player 1 Bet (50)")
        btn2.clicked.connect(lambda: self.history_view.add_action(1, 'bet', 50))
        controls_layout.addWidget(btn2)

        btn3 = QPushButton("Add: Player 0 Call (50)")
        btn3.clicked.connect(lambda: self.history_view.add_action(0, 'call', 50))
        controls_layout.addWidget(btn3)

        btn4 = QPushButton("Add: Player 1 Fold")
        btn4.clicked.connect(lambda: self.history_view.add_action(1, 'fold'))
        controls_layout.addWidget(btn4)

        layout.addLayout(controls_layout)

        controls2_layout = QHBoxLayout()

        btn5 = QPushButton("Add Round Separator")
        btn5.clicked.connect(lambda: self.history_view.add_round_separator("Round 2"))
        controls2_layout.addWidget(btn5)

        btn6 = QPushButton("Add Game Result")
        btn6.clicked.connect(lambda: self.history_view.add_game_result(0, 100))
        controls2_layout.addWidget(btn6)

        btn7 = QPushButton("Clear History")
        btn7.clicked.connect(self.history_view.clear)
        controls2_layout.addWidget(btn7)

        btn8 = QPushButton("Load Sample Game")
        btn8.clicked.connect(self.load_sample_game)
        controls2_layout.addWidget(btn8)

        controls2_layout.addStretch()
        layout.addLayout(controls2_layout)

        layout.addStretch()

    def load_sample_game(self):
        self.history_view.clear()
        sample_history = [
            'check',
            'bet',
            'call',
            '|',
            'bet',
            'call',
            '|',
            'check',
            'check'
        ]
        self.history_view.set_history(sample_history, {0: "Alice", 1: "Bob"})


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TestHistoryViewWindow()
    window.show()
    sys.exit(app.exec())
