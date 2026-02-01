from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingConfig:
    game: str
    iterations: int
    algorithm: str = 'cfr'
    br_eval_schedule: Optional[str] = None
    alternating_updates: bool = True
    partial_pruning: bool = False
    no_suit_abstraction: bool = False
    dcfr_alpha: float = 1.5
    dcfr_beta: float = 0.0
    dcfr_gamma: float = 2.0
    squared_weight: bool = False
    early_stop_exploitability_mb: Optional[float] = None
