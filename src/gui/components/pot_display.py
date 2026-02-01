from PyQt6.QtWidgets import QWidget, QLabel, QVBoxLayout
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QPainter, QColor, QBrush, QPen
from gui.components.chip import Chip


class PotDisplay(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.pot_value = 0
        self.setMinimumSize(150, 100)
        self.setMaximumSize(200, 120)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.pot_label = QLabel("Pot: 0")
        self.pot_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pot_label.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        self.pot_label.setStyleSheet("""
            QLabel {
                color: white;
                background-color: rgba(0, 0, 0, 0.5);
                border-radius: 10px;
                padding: 10px;
            }
        """)
        layout.addWidget(self.pot_label)

    def set_pot(self, value):
        """Setzt den Pot-Wert und aktualisiert die Anzeige"""
        self.pot_value = value
        self.pot_label.setText(f"Pot: {value}")

    def add_to_pot(self, amount):
        """Fügt Betrag zum Pot hinzu"""
        self.set_pot(self.pot_value + amount)

    def get_center_position(self):
        """Gibt die zentrale Position des Pot Displays zurück (für Chip-Animationen)"""
        return self.mapToGlobal(self.rect().center())
