from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QPushButton
from gui.layouts.base_poker_layout import BasePokerLayout
from gui.components.action_buttons import ActionButtons

class AgentVsHumanLayout(BasePokerLayout):
    
    def __init__(self, parent=None):
        super().__init__(parent)
        QTimer.singleShot(100, self.setup_action_buttons)
        QTimer.singleShot(150, self.setup_restart_button_in_layout)
    
    def setup_action_buttons(self):
        game_area = None
        for i in range(self.main_layout.count()):
            item = self.main_layout.itemAt(i)
            if item:
                widget = item.widget()
                if widget:
                    game_area = widget
                    break
        
        if not game_area:
            return
        
        left_area = None
        game_area_layout = game_area.layout()
        if game_area_layout:
            for i in range(game_area_layout.count()):
                item = game_area_layout.itemAt(i)
                if item:
                    widget = item.widget()
                    if widget:
                        left_area = widget
                        break
        
        if not left_area:
            return
        
        left_area_layout = left_area.layout()
        if not left_area_layout:
            return
        
        for i in range(left_area_layout.count()):
            item = left_area_layout.itemAt(i)
            if item and item.widget() == self.control_area:
                left_area_layout.removeWidget(self.control_area)
                self.control_area.deleteLater()
                
                self.action_buttons = ActionButtons()
                left_area_layout.addWidget(self.action_buttons)
                break
    
    def setup_restart_button_in_layout(self):
        if not hasattr(self, 'main_layout'):
            QTimer.singleShot(100, lambda: self.setup_restart_button_in_layout())
            return
        
        self.restart_button = QPushButton("🔄 New Hand")
        self.restart_button.setFixedSize(120, 40)
        self.restart_button.setStyleSheet("""
            QPushButton {
                background-color: #1a5490;
                color: white;
                border: none;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2a6ab0;
            }
            QPushButton:pressed {
                background-color: #0a3a70;
            }
        """)
        self.restart_button.setParent(self)
        self.restart_button.show()
        self.restart_button.raise_()
        
        def position_button():
            if hasattr(self, 'restart_button') and self.restart_button:
                button_x = self.width() - self.restart_button.width() - 20
                button_y = 20
                self.restart_button.move(button_x, button_y)
                self.restart_button.raise_()
        
        QTimer.singleShot(200, position_button)
        QTimer.singleShot(500, position_button)
        
        original_resize = self.resizeEvent
        def resize_with_button(event):
            original_resize(event)
            position_button()
        self.resizeEvent = resize_with_button

