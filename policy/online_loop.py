# policy/online_loop.py — Autoplay non-stop cu debounce pe WIN/LOSE și OK (versiune corectată)
from __future__ import annotations
import time, json, os
from typing import Optional
import numpy as np
import cv2

from controllers.memu_adapter import MemuAdapter
from controllers.input_mapper import InputMapper, UILayout
from vision.live_view import LiveViewer
from policy.qlearn import QLearner
from vision.card_detector import detect_hand_cards
from vision.unit_tracker import UnitTracker
from config import LAYOUT_PATH

# -------- util: HP bars (reward) --------
def _ratio_color(frame_bgr: np.ndarray, x1: int, y1: int, x2: int, y2: int, color: str) -> float:
    roi = frame_bgr[max(0,y1):max(0,y2), max(0,x1):max(0,x2)]
    if roi.size == 0: return 0.0
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    if color == "red":
        lower1, upper1 = np.array([0,120,120]),  np.array([10,255,255])
        lower2, upper2 = np.array([170,120,120]), np.array([180,255,255])
        mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
    else:
        lower, upper = np.array([95,100,100]), np.array([130,255,255])
        mask = cv2.inRange(hsv, lower, upper)
    k = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    return float(np.count_nonzero(mask)) / float(mask.size)

# -------- stats win/lose --------
class Stats:
    def __init__(self, path: str = "data/match_stats.json"):
        self.path = path
        self.data = {"wins": 0, "losses": 0}
        self._load()
    def _load(self):
        if os.path.isfile(self.path):
            try: self.data = json.load(open(self.path, "r", encoding="utf-8"))
            except Exception: pass
    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        json.dump(self.data, open(self.path, "w", encoding="utf-8"))
    def add_win(self):   self.data["wins"]   = int(self.data.get("wins",0)) + 1;   self.save()
    def add_loss(self):  self.data["losses"] = int(self.data.get("losses",0)) + 1; self.save()

class OnlineLearner:
    def __init__(self, ctrl: Optional[MemuAdapter] = None):
        self.ctrl = ctrl or MemuAdapter(render_mode="directx")
        self.layout = UILayout.load(LAYOUT_PATH)
        self.map = InputMapper(self.ctrl, self.layout)
        self.viewer: Optional[LiveViewer] = None

        # Agent RL (persistă în data/qtable.json)
        self.agent = QLearner(n_actions=8, alpha=0.18, gamma=0.95,
                              eps0=0.35, eps_min=0.05, eps_decay=7e-4,
                              path="data/qtable.json")
        # Tracking unități (șabloane în data/templates/)
        self.u_tracker = UnitTracker(template_dir="data/templates",
                                     names=None, ttl=0.9, iou_match_thr=0.35, thresh=0.78)
        self.stats = Stats(path="data/match_stats.json")

        # evaluare reward întârziată după o acțiune
        self._pending_eval = None  # (due_ts, lane_str, prev_enemy_hp, prev_ally_hp, s, a)

        # debounce final meci
        self._end_label = None
        self._end_count = 0

    # ---------- viewer ----------
    def _ensure_viewer(self):
        if self.viewer is None:
            self.viewer = LiveViewer(title="AI View (pixel elixir + Q-learning)",
                                     fps=24, scale=1.15, show_layout=True)
            self.map.set_action_callback(self.viewer.notify_action)
            self.viewer.open()

    # ---------- features pentru state/reward ----------
    def _hp_metrics(self, frame):
        H, W = frame.shape[:2]
        lx1,ly1,lx2,ly2 = self.layout.rect("left_tower",  W, H)
        rx1,ry1,rx2,ry2 = self.layout.rect("right_tower", W, H)
        lex1,ley1,lex2,ley2 = self.layout.rect("left_enemy_tower",  W, H)
        rex1,rey1,rex2,rey2 = self.layout.rect("right_enemy_tower", W, H)
        return {
            "ally_left"  : _ratio_color(frame, lx1,ly1,lx2,ly2, "blue"),
            "ally_right" : _ratio_color(frame, rx1,ry1,rx2,ry2, "blue"),
            "enemy_left" : _ratio_color(frame, lex1,ley1,lex2,ley2, "red"),
            "enemy_right": _ratio_color(frame, rex1,rey1,rex2,rey2, "red"),
        }
    def _phase(self, t: float) -> int: return 0 if t < 60 else (1 if t < 120 else 2)
    def _e_bucket(self, e: int) -> int: return max(0, min(5, int(e)//2))
    def _balance_bucket(self, hp) -> int:
        dL = hp["enemy_left"]  - hp["ally_left"]
        dR = hp["enemy_right"] - hp["ally_right"]
        diff = dL - dR
        b = int(round(diff / 0.15))
        return max(-2, min(2, b)) + 2
    def _state(self, t_since_start: float, elixir_now: int, hp_metrics) -> tuple[int,int,int]:
        return ( self._phase(t_since_start),
                 self._e_bucket(elixir_now),
                 self._balance_bucket(hp_metrics) )
    def _action_to_slot_lane(self, a: int):
        a = int(a) % 8
        slot = (a % 4) + 1
        lane = "left" if a < 4 else "right"
        return slot, lane

    # ---------- overlays push ----------
    def _push_all(self, img):
        elixir = self.map.get_elixir_from_frame(img)
        hp     = self._hp_metrics(img)
        try:    card_dets = detect_hand_cards(img, self.layout)
        except Exception: card_dets = None
        try:    unit_dets = self.u_tracker.update(img, self.layout)
        except Exception: unit_dets = None
        self.viewer.push_arena(img, layout=self.layout, elixir=elixir, hp_metrics=hp,
                               card_dets=card_dets, unit_dets=unit_dets)

    # ---------- debounce final meci ----------
    def _reset_end_detector(self):
        self._end_label, self._end_count = None, 0

    def _check_match_end(self, img) -> Optional[str]:
        """
        Confirmă finalul doar dacă:
          - detectăm același rezultat ('ally'/'enemy') 3 cadre la rând
          - ȘI butonul OK este prezent (thr mai strict)
        """
        res = self.map.detect_winner_banner(img, thr=0.12)
        ok  = self.map.ok_button_present(img, thr=0.20)
        if res and ok:
            if self._end_label == res:
                self._end_count += 1
            else:
                self._end_label, self._end_count = res, 1
            if self._end_count >= 3:
                return res
        else:
            if self._end_count > 0:
                self._end_count -= 1
            if self._end_count == 0:
                self._end_label = None
        return None

    # ---------- lobby & queue (CU wake tap) ----------
    def _goto_battle_blocking(self) -> bool:
        """
        Re-queue până intră în arenă.
        Dacă BATTLE nu e vizibil, dă 'wake tap' periodic. (Numai aici!)
        """
        taps = 0
        consecutive_elixir_frames = 0
        last_wake = 0.0
        w,h = self.ctrl.get_device_resolution()
        if not w or not h:
            fr = self.ctrl.screenshot()
            h,w = fr.shape[:2]
        wake_xy = (int(w*0.5), int(h*0.85))

        while True:
            frame = self.ctrl.screenshot()
            self._ensure_viewer()
            self._push_all(frame)
            if not self.viewer.update():
                return False

            if self.map.battle_button_present(frame):
                print("[QUEUE] BATTLE vizibil -> tap")
                self.map.tap_battle(); taps += 1
                time.sleep(0.8)
                frame = self.ctrl.screenshot()

            # a intrat în arenă dacă elixir > 0 câteva cadre
            e = self.map.get_elixir_from_frame(frame)
            consecutive_elixir_frames = consecutive_elixir_frames + 1 if e > 0 else 0
            if consecutive_elixir_frames >= 3:
                print("[QUEUE] Arena detectată (elixir>0).")
                return True

            # altfel încearcă un 'wake tap' la interval
            now = time.time()
            if now - last_wake > 2.5:
                print("[QUEUE] BATTLE nu e vizibil -> wake tap")
                self.ctrl.tap(*wake_xy)
                last_wake = now
                time.sleep(0.4)

    def _wait_full_elixir_blocking(self) -> bool:
        """Așteaptă până elixir == 10 (pixel). ESC/Q oprește și întoarce False."""
        while True:
            frame = self.ctrl.screenshot()
            self._push_all(frame)
            if not self.viewer.update():
                return False
            if self.map.get_elixir_from_frame(frame) >= 10:
                print("[MATCH] Elixir 10/10 -> încep episodul.")
                return True

    # ---------- final de meci ----------
    def _finish_match_and_return_lobby(self) -> None:
        """
        La final:
          - contorizează win/lose (confirmat)
          - apasă OK DOAR când e stabil 2 cadre
          - așteaptă dispariția OK și apariția BATTLE (fiecare stabil de 2 cadre)
          - fără wake-tap aici (evităm să apăsăm peste BATTLE accidental)
        """
        # 1) confirmă încă o dată rezultatul și contorizează
        t_deadline = time.time() + 45.0
        while time.time() < t_deadline:
            img = self.ctrl.screenshot()
            res = self.map.detect_winner_banner(img, thr=0.12)
            if res and self.map.ok_button_present(img, thr=0.20):
                if res == "ally":
                    self.stats.add_win();  print("[END] WIN. Stats:", self.stats.data)
                else:
                    self.stats.add_loss(); print("[END] LOSE. Stats:", self.stats.data)
                break
            self._push_all(img)
            if not self.viewer.update(): return
            time.sleep(0.3)

        # 2) apasă OK când e stabil; apoi așteaptă să dispară
        ok_stable = 0
        ok_gone_stable = 0
        t_deadline = time.time() + 30.0
        while time.time() < t_deadline:
            img = self.ctrl.screenshot()
            if self.map.ok_button_present(img, thr=0.20):
                ok_stable += 1
                if ok_stable >= 2:
                    print("[END] OK -> tap")
                    self.map.tap_ok_button(tries=1)
                    time.sleep(0.9)
                    ok_stable = 0
                    ok_gone_stable = 0
            else:
                ok_stable = 0
                ok_gone_stable += 1
            if ok_gone_stable >= 2:
                break
            self._push_all(img)
            if not self.viewer.update(): return
            time.sleep(0.3)

        # 3) Așteaptă BATTLE stabil (nu dăm wake-tap aici!)
        battle_stable = 0
        t_deadline = time.time() + 20.0
        while time.time() < t_deadline:
            img = self.ctrl.screenshot()
            if self.map.battle_button_present(img):
                battle_stable += 1
                if battle_stable >= 2:
                    print("[END] Back to lobby.")
                    return
            else:
                battle_stable = 0
            self._push_all(img)
            if not self.viewer.update(): return
            time.sleep(0.4)

    # ---------- jocul efectiv al unui meci ----------
    def _play_one_match_episode(self) -> None:
        self._reset_end_detector()
        t0 = time.time()
        last_decision_t = 0.0

        frame = self.ctrl.screenshot()
        hp_prev = self._hp_metrics(frame)
        elixir_now = self.map.get_elixir_from_frame(frame)
        s = self._state(0.0, elixir_now, hp_prev)

        while True:
            img = self.ctrl.screenshot()
            self._push_all(img)
            if not self.viewer.update():
                return

            # verifică final de meci cu debounce + OK
            end = self._check_match_end(img)
            if end in ("ally","enemy"):
                self._finish_match_and_return_lobby()
                return

            # evaluare întârziată → update Q(s,a)
            if self._pending_eval and time.time() >= self._pending_eval[0]:
                _, lane_str, enemy_prev, ally_prev, s_eval, a_eval = self._pending_eval
                cur = self._hp_metrics(img)
                enemy_key = "enemy_left" if "left" in lane_str else "enemy_right"
                ally_key  = "ally_left"  if "left" in lane_str else "ally_right"
                de = max(0.0, enemy_prev - cur[enemy_key])
                da = max(0.0, cur[ally_key] - ally_prev)
                r = (de * 1.0) - (da * 0.5)

                e_next = self.map.get_elixir_from_frame(img)
                s_next = self._state(time.time()-t0, e_next, cur)
                self.agent.update(s_eval, a_eval, r, s_next)
                self.agent.save()

                hp_prev = cur
                s = s_next
                self._pending_eval = None
                print(f"[QL] s={s_eval} a={a_eval} r={r:.3f} -> s'={s_next}")

            # decide ~la 1.6s
            if time.time() - last_decision_t > 1.6:
                elixir_now = self.map.get_elixir_from_frame(img)
                s = self._state(time.time()-t0, elixir_now, hp_prev)
                a, eps_cur = self.agent.act(s, t_seconds=time.time()-t0)
                slot, lane = self._action_to_slot_lane(a)
                print(f"[ACT] s={s} eps={eps_cur:.3f} -> a={a} (card{slot},{lane})  elixir={elixir_now}")

                side_enemy_prev = hp_prev["enemy_left" if lane=="left" else "enemy_right"]
                side_ally_prev  = hp_prev["ally_left"  if lane=="left" else "ally_right"]

                self.map.play_card((slot, lane))
                self._pending_eval = (time.time() + 2.3, lane, side_enemy_prev, side_ally_prev, s, a)
                last_decision_t = time.time()

    # ---------- buclă infinită ----------
    def run_forever(self):
        """Queue → joacă → finalizează → back to lobby → repeat, până închizi (ESC/Q)."""
        self.ctrl.ensure_clash()
        self._ensure_viewer()
        while True:
            # queue (cu wake-tap doar aici)
            if not self._goto_battle_blocking():
                break
            # wait 10 elixir
            if not self._wait_full_elixir_blocking():
                break
            # episod meci
            self._play_one_match_episode()

    # compat
    def one_episode(self, eps: float = 0.2, seconds: Optional[float] = None) -> float:
        self.run_forever()
        return 0.0
