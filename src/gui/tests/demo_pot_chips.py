import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont
from gui.components.poker_table import PokerTable
from gui.components.pot_display import PotDisplay
from gui.components.chip import Chip


class TestPotChipsWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pot & Chips Test")
        self.setMinimumSize(1200, 800)
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

        title = QLabel("Pot Display & Chip Animation Test")
        title.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        layout.addWidget(title)

        # Tisch-Container mit Overlay für Pot
        table_container = QWidget()
        table_container.setStyleSheet("background-color: #0a1929;")
        table_layout = QVBoxLayout(table_container)
        table_layout.setContentsMargins(50, 50, 50, 50)

        self.poker_table = PokerTable()
        table_layout.addWidget(self.poker_table)

        # Pot Display über dem Tisch positionieren
        self.pot_display = PotDisplay()
        self.pot_display.setParent(self.poker_table)
        self.pot_display.set_pot(0)

        layout.addWidget(table_container, 2)

        # Chip-Bereich und Buttons
        controls_layout = QHBoxLayout()

        # Chips zum Testen
        self.chips = []
        chip_values = [1, 5, 10, 25, 100]
        for value in chip_values:
            chip = Chip(value)
            chip.setParent(self.poker_table)
            chip.move(100 + len(self.chips) * 60, 200)
            chip.show()
            self.chips.append(chip)

        # Buttons
        add_pot_btn = QPushButton("Add 50 to Pot")
        add_pot_btn.clicked.connect(self.add_to_pot)
        controls_layout.addWidget(add_pot_btn)

        animate_chip_btn = QPushButton("Animate Chip to Pot")
        animate_chip_btn.clicked.connect(self.animate_chip_to_pot)
        controls_layout.addWidget(animate_chip_btn)

        reset_btn = QPushButton("Reset Pot")
        reset_btn.clicked.connect(self.reset_pot)
        controls_layout.addWidget(reset_btn)

        controls_layout.addStretch()
        layout.addLayout(controls_layout)

        # Positioniere Pot Display in der Mitte des Tisches
        QTimer.singleShot(100, self.position_pot_display)

    def position_pot_display(self):
        """Positioniert das Pot Display in der Mitte des Tisches"""
        table_rect = self.poker_table.rect()
        pot_rect = self.pot_display.rect()
        x = (table_rect.width() - pot_rect.width()) // 2
        y = (table_rect.height() - pot_rect.height()) // 2
        self.pot_display.move(x, y)
        self.pot_display.show()

    def add_to_pot(self):
        """Fügt Betrag zum Pot hinzu"""
        self.pot_display.add_to_pot(50)

    def animate_chip_to_pot(self):
        """Animiert einen Chip zum Pot"""
        if not self.chips:
            return

        chip = self.chips[0]  # Nimm ersten Chip

        # Berechne Zielposition: Pot-Mitte relativ zum Tisch
        pot_rect = self.pot_display.geometry()
        pot_center_x = pot_rect.x() + pot_rect.width() // 2
        pot_center_y = pot_rect.y() + pot_rect.height() // 2

        # Zielposition: Pot-Mitte minus halbe Chip-Größe (damit Chip zentriert ist)
        target_pos = self.poker_table.mapFromGlobal(
            self.pot_display.mapToGlobal(
                self.pot_display.rect().center()
            )
        ) - chip.rect().center()

        animation = chip.animate_to(target_pos, duration=800)

        # Nach Animation: Chip entfernen und Pot erhöhen
        def on_finished():
            chip.hide()
            if chip in self.chips:
                self.chips.remove(chip)
            self.pot_display.add_to_pot(chip.value)

        animation.finished.connect(on_finished)

    def reset_pot(self):
        """Setzt Pot zurück"""
        self.pot_display.set_pot(0)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TestPotChipsWindow()
    window.show()
    sys.exit(app.exec())
