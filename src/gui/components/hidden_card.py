from PyQt6.QtWidgets import QLabel
from PyQt6.QtCore import Qt


class HiddenCard(QLabel):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(80, 120)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("""
            QLabel {
                background-color: #1a5490;
                border: 2px solid #333;
                border-radius: 8px;
            }
        """)
