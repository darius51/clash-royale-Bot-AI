# state/representation.py
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class GameState:
    elixir_me: int
    elixir_enemy: Optional[int]  # poate rămâne None dacă nu-l estimăm
    cards_ready: List[int]
    time_left: Optional[float] = None

    def to_vector(self) -> list[int]:
        # -1 când nu știm elixirul adversarului
        return [int(self.elixir_me), int(self.elixir_enemy if self.elixir_enemy is not None else -1), int(len(self.cards_ready))]
