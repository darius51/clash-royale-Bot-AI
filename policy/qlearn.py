# policy/qlearn.py — Tabular Q-Learning cu persistență JSON

from __future__ import annotations
import json, os, math, random
from typing import Dict, List, Any, Tuple

class QLearner:
    def __init__(
        self,
        n_actions: int = 8,
        alpha: float = 0.18,
        gamma: float = 0.95,
        eps0: float = 0.35,
        eps_min: float = 0.05,
        eps_decay: float = 7e-4,   # scade epsilon în timp
        path: str = "data/qtable.json",
    ):
        self.n = int(n_actions)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.eps0 = float(eps0)
        self.eps_min = float(eps_min)
        self.eps_decay = float(eps_decay)
        self.path = path
        self.Q: Dict[str, List[float]] = {}
        self.load()

    # --------- intern ----------
    def _key(self, s: Any) -> str:
        if isinstance(s, (tuple, list)):
            return "T:" + "|".join(map(str, s))
        if isinstance(s, dict):
            return "D:" + "|".join(f"{k}={s[k]}" for k in sorted(s.keys()))
        return str(s)

    def _row(self, s: Any) -> List[float]:
        k = self._key(s)
        row = self.Q.get(k)
        if row is None:
            row = [0.0] * self.n
            self.Q[k] = row
        return row

    # --------- API ----------
    def act(self, state: Any, t_seconds: float | None = None) -> Tuple[int, float]:
        row = self._row(state)
        eps = self.eps0 if t_seconds is None else max(self.eps_min, self.eps0 * math.exp(-self.eps_decay * t_seconds))
        if random.random() < eps:
            a = random.randrange(self.n)
            return a, eps
        a = int(max(range(self.n), key=lambda i: row[i]))
        return a, eps

    def update(self, s: Any, a: int, r: float, s_next: Any) -> None:
        row = self._row(s)
        row_next = self._row(s_next)
        td_target = float(r) + self.gamma * max(row_next)
        row[a] += self.alpha * (td_target - row[a])

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.Q, f)

    def load(self) -> None:
        if os.path.isfile(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self.Q = json.load(f)
            except Exception:
                self.Q = {}
