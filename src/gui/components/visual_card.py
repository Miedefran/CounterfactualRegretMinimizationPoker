from PyQt6.QtWidgets import QLabel
from PyQt6.QtCore import Qt


class VisualCard(QLabel):

    def __init__(self, card, parent=None):
        super().__init__(parent)
        self.setFixedSize(80, 120)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("""
            QLabel {
                background-color: white;
                border: 2px solid #333;
                border-radius: 8px;
            }
        """)
        self.set_card(card)

    def set_card(self, card):
        if card is None:
            self.setText("")
            self.setStyleSheet("""
                QLabel {
                    background-color: #1a5490;
                    border: 2px solid #333;
                    border-radius: 8px;
                }
            """)
            return

        rank, suit, color = self.parse_card(card)

        if suit:
            html = f"""
            <div style="
                font-family: Arial;
                font-weight: bold;
                color: {color};
                padding: 5px;
                height: 100%;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
            ">
                <div style="font-size: 20px; text-align: left;">{rank}</div>
                <div style="font-size: 32px; text-align: center; margin: 10px 0;">{suit}</div>
                <div style="font-size: 20px; text-align: right; transform: rotate(180deg);">{rank}</div>
            </div>
            """
        else:
            html = f"""
            <div style="
                font-family: Arial;
                font-weight: bold;
                color: {color};
                padding: 5px;
                height: 100%;
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
            ">
                <div style="font-size: 48px; text-align: center;">{rank}</div>
            </div>
            """

        self.setText(html)
        self.setStyleSheet(f"""
            QLabel {{
                background-color: white;
                border: 2px solid #333;
                border-radius: 8px;
            }}
        """)

    def parse_card(self, card):
        if isinstance(card, str):
            if len(card) == 1:
                return (card, "", "#000")

            rank = card[0]
            suit_char = card[1] if len(card) > 1 else ""

            suit_symbols = {
                's': '♠',
                'h': '♥',
                'd': '♦',
                'c': '♣'
            }

            suit_colors = {
                's': '#000000',
                'c': '#000000',
                'h': '#dc143c',
                'd': '#dc143c'
            }

            suit = suit_symbols.get(suit_char, suit_char)
            color = suit_colors.get(suit_char, '#000000')

            return (rank, suit, color)

        return (str(card), "", "#000")
