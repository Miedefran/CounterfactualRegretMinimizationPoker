from PyQt6.QtWidgets import QWidget, QHBoxLayout, QPushButton, QSlider, QLabel
from PyQt6.QtCore import Qt, pyqtSignal


class PlaybackControls(QWidget):
    step_forward_requested = pyqtSignal()
    step_backward_requested = pyqtSignal()
    play_pause_toggled = pyqtSignal(bool)
    speed_changed = pyqtSignal(float)
    fast_forward_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.is_playing = False
        self.layout = QHBoxLayout(self)
        self.layout.setSpacing(10)

        self.setup_buttons()
        self.setup_connections()
        self.setup_style()

    def setup_buttons(self):
        self.play_pause_btn = QPushButton("▶ PLAY")
        self.play_pause_btn.setMinimumSize(120, 50)
        self.layout.addWidget(self.play_pause_btn)

        self.step_back_btn = QPushButton("⏮ STEP BACK")
        self.step_back_btn.setMinimumSize(120, 50)
        self.layout.addWidget(self.step_back_btn)

        self.step_forward_btn = QPushButton("STEP ⏭")
        self.step_forward_btn.setMinimumSize(120, 50)
        self.layout.addWidget(self.step_forward_btn)

        self.fast_forward_btn = QPushButton("⏭⏭ FAST FORWARD")
        self.fast_forward_btn.setMinimumSize(140, 50)
        self.layout.addWidget(self.fast_forward_btn)

        self.layout.addStretch()

        speed_label = QLabel("Speed:")
        speed_label.setStyleSheet("color: white; font-size: 14px;")
        self.layout.addWidget(speed_label)

        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setMinimum(0)
        self.speed_slider.setMaximum(6)
        self.speed_slider.setValue(3)
        self.speed_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.speed_slider.setTickInterval(1)
        self.speed_slider.setMinimumWidth(150)
        self.speed_slider.setMaximumWidth(200)
        self.layout.addWidget(self.speed_slider)

        self.speed_value_label = QLabel("x1.0")
        self.speed_value_label.setStyleSheet("color: white; font-size: 14px; min-width: 50px;")
        self.speed_value_label.setMinimumWidth(50)
        self.layout.addWidget(self.speed_value_label)

        self.update_speed_display()

    def setup_connections(self):
        self.play_pause_btn.clicked.connect(self.toggle_play_pause)
        self.step_back_btn.clicked.connect(lambda: self.step_backward_requested.emit())
        self.step_forward_btn.clicked.connect(lambda: self.step_forward_requested.emit())
        self.fast_forward_btn.clicked.connect(lambda: self.fast_forward_requested.emit())
        self.speed_slider.valueChanged.connect(self.on_speed_changed)

    def setup_style(self):
        self.setStyleSheet("""
            QPushButton {
                background-color: #1a5490;
                color: white;
                border: none;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover:enabled {
                background-color: #2a6ab0;
            }
            QPushButton:disabled {
                background-color: #555;
                color: #999;
            }
            QPushButton:pressed {
                background-color: #0a3a70;
            }
            QSlider::groove:horizontal {
                border: 1px solid #333;
                height: 8px;
                background: #333;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #1a5490;
                border: 2px solid #fff;
                width: 18px;
                height: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
            QSlider::handle:horizontal:hover {
                background: #2a6ab0;
            }
            QSlider::sub-page:horizontal {
                background: #1a5490;
                border-radius: 4px;
            }
        """)

        self.play_pause_btn.setStyleSheet("""
            QPushButton {
                background-color: #388e3c;
                color: white;
                border: none;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover:enabled {
                background-color: #4caf50;
            }
            QPushButton:disabled {
                background-color: #555;
                color: #999;
            }
            QPushButton:pressed {
                background-color: #2e7d32;
            }
        """)

    def toggle_play_pause(self):
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.play_pause_btn.setText("⏸ PAUSE")
            self.play_pause_btn.setStyleSheet("""
                QPushButton {
                    background-color: #d32f2f;
                    color: white;
                    border: none;
                    border-radius: 5px;
                    font-size: 14px;
                    font-weight: bold;
                }
                QPushButton:hover:enabled {
                    background-color: #f44336;
                }
                QPushButton:disabled {
                    background-color: #555;
                    color: #999;
                }
                QPushButton:pressed {
                    background-color: #b71c1c;
                }
            """)
        else:
            self.play_pause_btn.setText("▶ PLAY")
            self.play_pause_btn.setStyleSheet("""
                QPushButton {
                    background-color: #388e3c;
                    color: white;
                    border: none;
                    border-radius: 5px;
                    font-size: 14px;
                    font-weight: bold;
                }
                QPushButton:hover:enabled {
                    background-color: #4caf50;
                }
                QPushButton:disabled {
                    background-color: #555;
                    color: #999;
                }
                QPushButton:pressed {
                    background-color: #2e7d32;
                }
            """)
        self.play_pause_toggled.emit(self.is_playing)

    def on_speed_changed(self, value):
        self.update_speed_display()
        speeds = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
        speed = speeds[value]
        self.speed_changed.emit(speed)

    def update_speed_display(self):
        speeds = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
        current_value = self.speed_slider.value()
        speed = speeds[current_value]
        self.speed_value_label.setText(f"x{speed}")

    def set_playing(self, playing):
        if self.is_playing != playing:
            self.toggle_play_pause()

    def set_enabled(self, enabled):
        self.play_pause_btn.setEnabled(enabled)
        self.step_back_btn.setEnabled(enabled)
        self.step_forward_btn.setEnabled(enabled)
        self.fast_forward_btn.setEnabled(enabled)
        self.speed_slider.setEnabled(enabled)

    def get_speed(self):
        speeds = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
        return speeds[self.speed_slider.value()]
