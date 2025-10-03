# policy/bandit.py — ε-greedy contextual bandit, cu persistență JSON.

from __future__ import annotations
import json, os, random
from typing import Dict, List, Any

class EpsGreedyBandit:
    def __init__(self, n_actions: int = 8, alpha: float = 0.15, path: str = "data/policy_bandit.json"):
        self.n = int(n_actions)
        self.alpha = float(alpha)
        self.path = path
        self.Q: Dict[str, List[float]] = {}
        self.load()

    def _key(self, ctx: Any) -> str:
        # ctx e orice tuplu/listă/dict serializabil; îl transformăm într-un string stabil
        if isinstance(ctx, (list, tuple)):
            return "T:" + "|".join(map(str, ctx))
        if isinstance(ctx, dict):
            return "D:" + "|".join(f"{k}={ctx[k]}" for k in sorted(ctx.keys()))
        return str(ctx)

    def pick(self, ctx: Any, eps: float = 0.2) -> int:
        key = self._key(ctx)
        row = self.Q.get(key)
        if (row is None) or (random.random() < eps):
            return random.randrange(self.n)
        # exploatare
        return int(max(range(self.n), key=lambda i: row[i]))

    def update(self, ctx: Any, action: int, reward: float) -> None:
        key = self._key(ctx)
        row = self.Q.setdefault(key, [0.0]*self.n)
        a = int(action)
        # Q(s,a) <- Q + alpha*(r - Q)
        row[a] += self.alpha * (float(reward) - row[a])

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
