from config import settings


class RulePolicy:
    def act(self, state) -> tuple[str, dict]:
# Regula "prostuță" baseline: dacă avem >=4 elixir, joacă prima carte
        if state.elixir_me >= 4 and state.cards_ready:
            slot = state.cards_ready[0]
            return ("PLAY_CARD", {"slot": slot})
        return ("NOOP", {})