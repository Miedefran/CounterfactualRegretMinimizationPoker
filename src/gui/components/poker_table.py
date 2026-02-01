from PyQt6.QtWidgets import QWidget, QSizePolicy
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPainter, QBrush, QColor, QPen


class PokerTable(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(800, 500)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        width = self.width()
        height = self.height()

        brush = QBrush(QColor(13, 93, 47))
        painter.setBrush(brush)
        painter.setPen(QPen(QColor(26, 122, 62), 3))

        margin = 30
        painter.drawRoundedRect(
            margin, margin,
            width - 2 * margin, height - 2 * margin,
            100, 100
        )
