from PyQt6.QtWidgets import QWidget, QLabel
from PyQt6.QtCore import Qt, QPropertyAnimation, QPoint, QEasingCurve, pyqtProperty
from PyQt6.QtGui import QPainter, QColor, QBrush, QPen, QFont


class Chip(QWidget):

    def __init__(self, value=1, color=None, parent=None):
        super().__init__(parent)
        self.value = value
        self.setFixedSize(50, 50)

        chip_colors = {
            1: '#E8E8E8',  # Hellblau/Weiß
            5: '#C84A4A',  # Rot
            10: '#5A8FC2',  # Blau
            20: '#6BA86B',  # Grün
            25: '#5A8F5A',  # Dunkelgrün
            50: '#4A6A9F',  # Dunkelblau
            100: '#2A2A2A',  # Schwarz
            500: '#8A6A9F',  # Lila
            1000: '#D4A574',  # Gold/Tan
            5000: '#D49FAF',  # Pink
            10000: '#D4C4A4'  # Beige/Tan
        }

        if color is None:
            self.ring_color = chip_colors.get(value, '#808080')
        else:
            self.ring_color = color

        self._position = QPoint(0, 0)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        width = self.width()
        height = self.height()
        center_x = width // 2
        center_y = height // 2

        outer_radius = min(width, height) // 2 - 2
        inner_radius = int(outer_radius * 0.6)

        ring_width = outer_radius - inner_radius

        # Äußerer Ring
        ring_brush = QBrush(QColor(self.ring_color))
        painter.setBrush(ring_brush)
        painter.setPen(QPen(QColor('#333333'), 1))
        painter.drawEllipse(int(center_x - outer_radius), int(center_y - outer_radius),
                            int(outer_radius * 2), int(outer_radius * 2))

        # Weißer innerer Kreis
        inner_brush = QBrush(QColor('#FFFFFF'))
        painter.setBrush(inner_brush)
        painter.setPen(QPen(QColor('#CCCCCC'), 1))
        painter.drawEllipse(int(center_x - inner_radius), int(center_y - inner_radius),
                            int(inner_radius * 2), int(inner_radius * 2))

        # Wert-Text
        painter.setPen(QPen(QColor('#000000')))
        font = QFont("Arial", 10, QFont.Weight.Bold)
        painter.setFont(font)

        text_rect = painter.fontMetrics().boundingRect(str(self.value))
        text_x = center_x - text_rect.width() // 2
        text_y = center_y + text_rect.height() // 2

        painter.drawText(text_x, text_y, str(self.value))

    def animate_to(self, target_pos, duration=500):
        animation = QPropertyAnimation(self, b"pos")
        animation.setDuration(duration)
        animation.setStartValue(self.pos())
        animation.setEndValue(target_pos)
        animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        animation.start()
        return animation
