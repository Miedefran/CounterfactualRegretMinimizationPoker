from PyQt6.QtWidgets import QWidget, QLabel, QVBoxLayout
from PyQt6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import QFont


class ResultOverlay(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.hide()

        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.result_label = QLabel()
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_label.setFont(QFont("Arial", 32, QFont.Weight.Bold))

        self.amount_label = QLabel()
        self.amount_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.amount_label.setFont(QFont("Arial", 24))

        layout.addWidget(self.result_label)
        layout.addWidget(self.amount_label)

    def show_result(self, is_winner, amount, auto_hide_ms=3000):
        if is_winner:
            self.result_label.setText("You Won!")
            self.result_label.setStyleSheet("""
                QLabel {
                    color: #4ade80;
                    background-color: transparent;
                    padding: 10px;
                }
            """)
            self.amount_label.setText(f"+{amount}")
            self.amount_label.setStyleSheet("""
                QLabel {
                    color: #22c55e;
                    background-color: transparent;
                    padding: 5px;
                }
            """)
            bg_color = "rgba(34, 197, 94, 0.2)"
            border_color = "#22c55e"
        else:
            self.result_label.setText("You Lost")
            self.result_label.setStyleSheet("""
                QLabel {
                    color: #f87171;
                    background-color: transparent;
                    padding: 10px;
                }
            """)
            self.amount_label.setText(f"-{amount}")
            self.amount_label.setStyleSheet("""
                QLabel {
                    color: #ef4444;
                    background-color: transparent;
                    padding: 5px;
                }
            """)
            bg_color = "rgba(239, 68, 68, 0.2)"
            border_color = "#ef4444"

        self.setStyleSheet(f"""
            QWidget {{
                background-color: {bg_color};
                border: 3px solid {border_color};
                border-radius: 15px;
            }}
        """)

        self.adjustSize()
        self.setMinimumSize(300, 150)
        self.show()
        self.raise_()

        if auto_hide_ms > 0:
            QTimer.singleShot(auto_hide_ms, self.hide)

    def show_tie(self, auto_hide_ms=7500):
        self.result_label.setText("Tie!")
        self.result_label.setStyleSheet("""
            QLabel {
                color: #fbbf24;
                background-color: transparent;
                padding: 10px;
            }
        """)
        self.amount_label.setText("Push")
        self.amount_label.setStyleSheet("""
            QLabel {
                color: #f59e0b;
                background-color: transparent;
                padding: 5px;
            }
        """)

        self.setStyleSheet("""
            QWidget {
                background-color: rgba(251, 191, 36, 0.2);
                border: 3px solid #f59e0b;
                border-radius: 15px;
            }
        """)

        self.adjustSize()
        self.setMinimumSize(300, 150)
        self.show()
        self.raise_()

        if auto_hide_ms > 0:
            QTimer.singleShot(auto_hide_ms, self.hide)
