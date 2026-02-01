"""
Zentrale Registry für Tree-Größen/Statistiken.

Anforderung:
- Die Registry wird NICHT pro Training-Run geschrieben, sondern beim Tree-Build/Save.
- Dedupliziert: Wenn ein identischer Eintrag bereits existiert, wird NICHT erneut geschrieben.
- Speicherung als JSON (Liste von Einträgen), damit man später leicht auswerten kann.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List


def _repo_root() -> str:
    # .../src/utils/tree_registry.py -> .../<repo_root>
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_registry_path() -> str:
    out_dir = os.path.join(_repo_root(), "data", "trees")
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, "tree_registry.json")


def _load_registry(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        return []
    except Exception:
        # Keine harte Abhängigkeit der Trainingspipeline von der Registry-Datei
        return []


def _atomic_write_json(path: str, obj: Any) -> None:
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp_path, path)


def record_tree_stats(entry: Dict[str, Any]) -> bool:
    """
    Speichert `entry` in der Registry, falls dieser Eintrag noch nicht existiert.
    Returns:
      True  => neu hinzugefügt
      False => bereits vorhanden / konnte nicht geschrieben werden
    """
    path = get_registry_path()
    entries = _load_registry(path)

    # Dedupe über exakte Gleichheit (ohne Timestamp o.ä., damit gleiches nicht wiederholt wird)
    if entry in entries:
        return False

    entries.append(entry)
    try:
        _atomic_write_json(path, entries)
        return True
    except Exception:
        return False
