"""Tests für Rate Limiter."""

import pytest

from gui.server.rate_limiter import RateLimiter, WebSocketRateLimiter


def test_zu_viele_requests_werden_abgelehnt():
    """Wenn zu viele Requests im Zeitfenster kommen, wird das Limit überschritten."""
    limiter = RateLimiter()
    ip = "192.168.1.1"
    
    for _ in range(5):
        assert limiter.is_allowed(ip, max_requests=5, window_seconds=60) is True
    
    assert limiter.is_allowed(ip, max_requests=5, window_seconds=60) is False


def test_websocket_rate_limit_greift_bei_vielen_nachrichten():
    """WebSocket: Zu viele Nachrichten von einem Spieler werden abgelehnt."""
    limiter = WebSocketRateLimiter()
    player_id = 0
    
    for _ in range(50):
        assert limiter.check_limit(player_id, max_messages=50, window=10.0) is True
    
    assert limiter.check_limit(player_id, max_messages=50, window=10.0) is False
