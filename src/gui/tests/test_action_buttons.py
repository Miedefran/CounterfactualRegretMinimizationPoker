import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QPushButton
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from gui.components.action_buttons import ActionButtons


class TestActionButtonsWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Action Buttons Test")
        self.setMinimumSize(800, 600)
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

        title = QLabel("Action Buttons Test")
        title.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        layout.addWidget(title)

        self.action_buttons = ActionButtons()
        self.action_buttons.action_selected.connect(self.on_action_selected)
        layout.addWidget(self.action_buttons)

        self.status_label = QLabel("No action selected yet")
        self.status_label.setFont(QFont("Arial", 14))
        layout.addWidget(self.status_label)

        test_scenarios = QVBoxLayout()
        test_scenarios_label = QLabel("Test Scenarios:")
        test_scenarios_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        test_scenarios.addWidget(test_scenarios_label)

        btn1 = QPushButton("Scenario 1: Check/Bet available")
        btn1.clicked.connect(lambda: self.action_buttons.update_legal_actions(['check', 'bet']))
        test_scenarios.addWidget(btn1)

        btn2 = QPushButton("Scenario 2: Call/Fold available")
        btn2.clicked.connect(lambda: self.action_buttons.update_legal_actions(['call', 'fold']))
        test_scenarios.addWidget(btn2)

        btn3 = QPushButton("Scenario 3: All actions available")
        btn3.clicked.connect(lambda: self.action_buttons.update_legal_actions(['check', 'bet', 'call', 'fold']))
        test_scenarios.addWidget(btn3)

        btn4 = QPushButton("Scenario 4: Bet with size range (10-100)")
        btn4.clicked.connect(lambda: self.action_buttons.update_legal_actions(['bet'], bet_size_range=(10, 100)))
        test_scenarios.addWidget(btn4)

        btn5 = QPushButton("Scenario 5: Call with amount (50)")
        btn5.clicked.connect(lambda: (
            self.action_buttons.update_legal_actions(['call', 'fold']),
            self.action_buttons.set_amount_to_call(50)
        ))
        test_scenarios.addWidget(btn5)

        btn6 = QPushButton("Disable All")
        btn6.clicked.connect(self.action_buttons.disable_all)
        test_scenarios.addWidget(btn6)

        layout.addLayout(test_scenarios)
        layout.addStretch()

    def on_action_selected(self, action, bet_size):
        if bet_size > 0:
            self.status_label.setText(f"Action selected: {action} (bet size: {bet_size})")
        else:
            self.status_label.setText(f"Action selected: {action}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TestActionButtonsWindow()
    window.show()
    sys.exit(app.exec())
