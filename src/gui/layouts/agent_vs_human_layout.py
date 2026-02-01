from PyQt6.QtCore import QTimer
from gui.layouts.base_poker_layout import BasePokerLayout
from gui.components.action_buttons import ActionButtons


class AgentVsHumanLayout(BasePokerLayout):

    def __init__(self, parent=None):
        super().__init__(parent)
        QTimer.singleShot(100, self.setup_action_buttons)

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
