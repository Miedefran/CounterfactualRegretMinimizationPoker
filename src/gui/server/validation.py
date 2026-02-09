"""Validierungs-Funktionen für Server-Inputs."""

from typing import Optional, Tuple, List
import html


def validate_player_id(player_id) -> Tuple[bool, Optional[int], Optional[str]]:
    """Validiert player_id. Gibt (is_valid, int_value, error_msg) zurück."""
    if player_id is None:
        return False, None, "player_id is required"
    try:
        pid = int(player_id)
        if pid not in (0, 1):
            return False, None, "player_id must be 0 or 1"
        return True, pid, None
    except (ValueError, TypeError):
        return False, None, "player_id must be an integer"


def validate_action(action, legal_actions: List[str]) -> Tuple[bool, Optional[str]]:
    """Validiert action gegen legal_actions."""
    if not isinstance(action, str):
        return False, "action must be a string"
    if action not in legal_actions:
        return False, f"action must be one of {legal_actions}"
    return True, None


def validate_bet_size(bet_size, min_bet: int = 0, max_bet: int = 10000) -> Tuple[bool, Optional[int], Optional[str]]:
    """Validiert bet_size."""
    if bet_size is None:
        return True, 0, None  # Default ist 0
    try:
        bet = int(bet_size)
        if bet < min_bet:
            return False, None, f"bet_size must be >= {min_bet}"
        if bet > max_bet:
            return False, None, f"bet_size must be <= {max_bet}"
        return True, bet, None
    except (ValueError, TypeError):
        return False, None, "bet_size must be an integer"


def validate_player_name(name, max_length: int = 50) -> Tuple[bool, Optional[str], Optional[str]]:
    """Validiert und säubert Player-Name."""
    if name is None:
        return True, None, None  # Name ist optional
    if not isinstance(name, str):
        return False, None, "name must be a string"
    name = name.strip()
    if len(name) == 0:
        return True, None, None  # Leerer Name ist OK (wird ignoriert)
    if len(name) > max_length:
        return False, None, f"name must be <= {max_length} characters"
    # HTML-Escape für XSS-Schutz
    name = html.escape(name)
    return True, name, None


def validate_starting_player(starting_player) -> Tuple[bool, Optional[int], Optional[str]]:
    """Validiert starting_player."""
    if starting_player is None:
        return True, 0, None  # Default ist 0
    try:
        sp = int(starting_player)
        if sp not in (0, 1):
            return False, None, "starting_player must be 0 or 1"
        return True, sp, None
    except (ValueError, TypeError):
        return False, None, "starting_player must be an integer"
