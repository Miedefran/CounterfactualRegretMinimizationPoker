from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtCore import QUrl
from pathlib import Path
import os


class SoundManager:

    def __init__(self, sounds_dir=None):
        if sounds_dir is None:
            sounds_dir = Path(__file__).parent / 'sounds'
        self.sounds_dir = Path(sounds_dir)

        self.audio_output = QAudioOutput()
        self.media_player = QMediaPlayer()
        self.media_player.setAudioOutput(self.audio_output)

        self.volume = 0.7
        self.audio_output.setVolume(self.volume)
        self.enabled = True

        self.sound_files = {
            'check': 'SamanthaCheck.wav',
            'bet': 'SamanthaRaise.wav',
            'call': 'SamanthaCall.wav',
            'fold': 'SamanthaFold.wav',
            'card_deal': 'card_deal.wav',
            'game_start': 'game_start.wav',
            'game_end': 'game_end.wav',
        }

    def play_sound(self, sound_name):
        if not self.enabled:
            return

        if sound_name not in self.sound_files:
            return

        sound_file = self.sounds_dir / self.sound_files[sound_name]

        if not sound_file.exists():
            return

        url = QUrl.fromLocalFile(str(sound_file.absolute()))
        self.media_player.setSource(url)
        self.media_player.play()

    def play_action(self, action):
        action_lower = action.lower()

        if action_lower == 'check':
            self.play_sound('check')
        elif action_lower == 'bet':
            self.play_sound('bet')
        elif action_lower == 'call':
            self.play_sound('call')
        elif action_lower == 'fold':
            self.play_sound('fold')

    def set_volume(self, volume):
        self.volume = max(0.0, min(1.0, volume))
        self.audio_output.setVolume(self.volume)

    def get_volume(self):
        return self.volume

    def set_enabled(self, enabled):
        self.enabled = enabled

    def is_enabled(self):
        return self.enabled

    def toggle(self):
        self.enabled = not self.enabled
        return self.enabled
