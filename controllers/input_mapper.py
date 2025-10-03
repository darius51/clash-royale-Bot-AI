# controllers/input_mapper.py — acțiuni + ELIXIR pixel-detect + BATTLE detect
# + WIN/LOSE detect + OK detect/tap (versiunea completă, compatibilă cu online_loop.run_forever)

from __future__ import annotations
import json, re
from dataclasses import dataclass
from typing import Tuple, Dict, Iterable, Any, Optional

import numpy as np
import cv2

from controllers.memu_adapter import MemuAdapter
from config import (
    LAYOUT_PATH,
    PIX_CANONICAL_W, PIX_CANONICAL_H,
    ELIXIR_PIXEL_COORDS, ELIXIR_PIXEL_RGB,
    ELIXIR_TOL, ELIXIR_SAMPLE_RADIUS,
)

# ---------------- UI layout ----------------
@dataclass
class ZoneRel:
    name: str
    x1: float; y1: float; x2: float; y2: float
    def rect_abs(self, w: int, h: int) -> Tuple[int,int,int,int]:
        return int(self.x1*w), int(self.y1*h), int(self.x2*w), int(self.y2*h)
    def center_abs(self, w: int, h: int) -> Tuple[int,int]:
        x1,y1,x2,y2 = self.rect_abs(w,h)
        return (x1+x2)//2, (y1+y2)//2

class UILayout:
    def __init__(self, zones: Dict[str, ZoneRel]):
        self.zones = zones
    @classmethod
    def load(cls, path: str) -> "UILayout":
        data = json.load(open(path, "r", encoding="utf-8"))
        zones = {it["name"]: ZoneRel(it["name"], it["x1"], it["y1"], it["x2"], it["y2"])
                 for it in data.get("items", [])}
        return cls(zones)
    def rect(self, name: str, w: int, h: int): return self.zones[name].rect_abs(w,h)
    def center(self, name: str, w: int, h: int): return self.zones[name].center_abs(w,h)
    def has(self, name: str) -> bool: return name in self.zones
    def names(self): return self.zones.keys()

# ---------------- ELIXIR pixel detect ----------------
def _px_equal(rgb: Tuple[int,int,int], target: Tuple[int,int,int], tol: int) -> bool:
    return (abs(int(rgb[0])-int(target[0])) <= tol and
            abs(int(rgb[1])-int(target[1])) <= tol and
            abs(int(rgb[2])-int(target[2])) <= tol)

def _read_rgb(frame: np.ndarray, x: int, y: int) -> Tuple[int,int,int]:
    b, g, r = frame[y, x]
    return (int(r), int(g), int(b))

def elixir_from_pixels(frame_bgr: np.ndarray,
                       coords=ELIXIR_PIXEL_COORDS,
                       target_rgb=ELIXIR_PIXEL_RGB,
                       tol: int = ELIXIR_TOL,
                       sample_radius: int = ELIXIR_SAMPLE_RADIUS,
                       canon: Tuple[int,int] = (PIX_CANONICAL_W, PIX_CANONICAL_H)) -> int:
    H, W = frame_bgr.shape[:2]
    cW, cH = canon
    for i in range(len(coords)-1, -1, -1):  # 10 -> 1
        cx, cy = coords[i]
        x = int(round(cx * W / float(cW)))
        y = int(round(cy * H / float(cH)))
        x = max(0, min(W-1, x))
        y = max(0, min(H-1, y))
        hit = False
        r = max(0, int(sample_radius))
        for dy in range(-r, r+1):
            yy = y + dy
            if yy < 0 or yy >= H: continue
            for dx in range(-r, r+1):
                xx = x + dx
                if xx < 0 or xx >= W: continue
                if _px_equal(_read_rgb(frame_bgr, xx, yy), target_rgb, tol):
                    hit = True; break
            if hit: break
        if hit:
            return i + 1
    return 0

# ---------------- Mapper ----------------
class InputMapper:
    """
    High-level actions pe layout relativ:
      - swipe card -> lane (fără play coords)
      - elixir via pixel-detect
      - BATTLE detect & tap
      - WIN/LOSE banner detect
      - OK detect & tap
    """
    def __init__(self, ctrl: MemuAdapter, layout: UILayout|None=None, action_cb=None):
        self.ctrl = ctrl
        self.layout = layout or UILayout.load(LAYOUT_PATH)
        self._action_cb = action_cb

        w,h = self.ctrl.get_device_resolution()
        if not w or not h:
            img = self.ctrl.screenshot(as_gray=True)
            h,w = img.shape[:2]
        self._W, self._H = int(w), int(h)

    # callback din viewer
    def set_action_callback(self, cb) -> None:
        self._action_cb = cb

    # ------------- helpers -------------
    def _xy(self, name: str) -> Tuple[int,int]:
        return self.layout.center(name, self._W, self._H)

    def _lane_name(self, lane: Any) -> str:
        if isinstance(lane, str):
            s = lane.strip().lower()
            if s in ("left_lane","left","l"):  return "left_lane"
            if s in ("right_lane","right","r"): return "right_lane"
        if isinstance(lane, bool): return "right_lane" if lane else "left_lane"
        if isinstance(lane, int):  return "right_lane" if lane == 1 else "left_lane"
        return "left_lane"

    def _lane_rect_fallback(self, side: str, W: int, H: int) -> Tuple[int,int,int,int]:
        x_center = None
        if side == "left":
            if self.layout.has("left_tower"):
                lx1,ly1,lx2,ly2 = self.layout.rect("left_tower", W, H)
                x_center = (lx1 + lx2)//2
            x1p, x2p = (0.20, 0.35) if x_center is None else ((x_center - int(0.09*W)), (x_center + int(0.09*W)))
        else:
            if self.layout.has("right_tower"):
                rx1,ry1,rx2,ry2 = self.layout.rect("right_tower", W, H)
                x_center = (rx1 + rx2)//2
            x1p, x2p = (0.65, 0.80) if x_center is None else ((x_center - int(0.09*W)), (x_center + int(0.09*W)))
        if isinstance(x1p, float):
            x1 = int(x1p * W); x2 = int(x2p * W)
        else:
            x1 = max(0, min(W-1, int(x1p)))
            x2 = max(0, min(W-1, int(x2p)))
        y1 = int(0.60 * H); y2 = int(0.90 * H)
        x1, x2 = max(0, min(x1, x2-1)), min(W-1, max(x1+1, x2))
        y1, y2 = max(0, min(y1, y2-1)), min(H-1, max(y1+1, y2))
        return x1, y1, x2, y2

    def _lane_rect(self, lane_name: str, W: int, H: int) -> Tuple[int,int,int,int]:
        if self.layout.has(lane_name):
            return self.layout.rect(lane_name, W, H)
        side = "left" if "left" in lane_name else "right"
        return self._lane_rect_fallback(side, W, H)

    # ------------- BATTLE detect -------------
    def _battle_mask_ratio(self, frame_bgr: np.ndarray) -> float:
        if not self.layout.has("battle_button"):
            return 0.0
        H, W = frame_bgr.shape[:2]
        x1,y1,x2,y2 = self.layout.rect("battle_button", W, H)
        roi = frame_bgr[max(0,y1):min(H,y2), max(0,x1):min(W,x2)]
        if roi.size == 0: return 0.0
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        ranges = [
            (np.array([15,140,140], np.uint8), np.array([35,255,255], np.uint8)),  # galben
            (np.array([8, 140,140], np.uint8), np.array([20,255,255], np.uint8)),  # oranj
        ]
        mask = np.zeros(roi.shape[:2], dtype=np.uint8)
        for lo,hi in ranges: mask |= cv2.inRange(hsv, lo, hi)
        k = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
        return float(np.count_nonzero(mask))/float(mask.size)

    def battle_button_present(self, frame_bgr: np.ndarray, min_ratio: float = 0.12) -> bool:
        try: return self._battle_mask_ratio(frame_bgr) >= float(min_ratio)
        except Exception: return False

    def tap_battle(self):
        if self.layout.has("battle_button"):
            x,y = self._xy("battle_button")
        else:
            x,y = int(self._W*0.5), int(self._H*0.88)
        self.ctrl.tap(x,y)

    # ------------- ELIXIR -------------
    def get_elixir_from_frame(self, frame_bgr: np.ndarray) -> int:
        return elixir_from_pixels(frame_bgr)

    def get_elixir(self) -> int:
        frame = self.ctrl.screenshot(as_gray=False)
        return elixir_from_pixels(frame)

    # ------------- WIN/LOSE banner detect -------------
    def _center_banner_roi(self, frame_bgr: np.ndarray):
        H, W = frame_bgr.shape[:2]
        x1, x2 = int(0.10*W), int(0.90*W)
        y1, y2 = int(0.18*H), int(0.55*H)
        roi = frame_bgr[y1:y2, x1:x2]
        return roi, (x1,y1,x2,y2)

    def detect_winner_banner(self, frame_bgr: np.ndarray, thr: float = 0.10) -> Optional[str]:
        """
        Returnează: 'ally' (win), 'enemy' (lose) sau None.
        Heuristic: dominanță albastru (ally) vs magenta/pink (enemy) în bannerul central.
        """
        roi, _ = self._center_banner_roi(frame_bgr)
        if roi.size == 0: return None
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # albastru
        mask_blue = cv2.inRange(hsv, (95,100,100), (130,255,255))
        # pink/magenta + un pic de roșu pentru siguranță
        mask_pink = cv2.inRange(hsv, (140,80,80), (179,255,255)) | cv2.inRange(hsv, (0,80,80), (10,255,255))
        k = np.ones((3,3), np.uint8)
        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, k, iterations=1)
        mask_pink = cv2.morphologyEx(mask_pink, cv2.MORPH_OPEN, k, iterations=1)
        rb = float(np.count_nonzero(mask_blue))/float(mask_blue.size)
        rp = float(np.count_nonzero(mask_pink))/float(mask_pink.size)
        if rb >= thr and rb > rp*1.2: return "ally"
        if rp >= thr and rp > rb*1.2: return "enemy"
        return None

    # ------------- OK button detect + tap -------------
    def ok_button_present(self, frame_bgr: np.ndarray, thr: float = 0.18) -> bool:
        H, W = frame_bgr.shape[:2]
        x1, x2 = int(0.35*W), int(0.65*W)
        y1, y2 = int(0.82*H), int(0.96*H)
        roi = frame_bgr[y1:y2, x1:x2]
        if roi.size == 0: return False
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask_blue = cv2.inRange(hsv, (95,100,100), (130,255,255))
        k = np.ones((3,3), np.uint8)
        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, k, iterations=1)
        ratio = float(np.count_nonzero(mask_blue))/float(mask_blue.size)
        return ratio >= thr

    def tap_ok_button(self, tries: int = 1):
        if self.layout.has("ok_button"):
            x,y = self._xy("ok_button")
        else:
            x,y = int(self._W*0.5), int(self._H*0.90)
        for _ in range(max(1, int(tries))):
            self.ctrl.tap(x,y)

    # ------------- acțiuni joc -------------
    def play_card_to_lane(self, slot_idx: int, lane: str, y_bias: float = 0.65) -> Tuple[int,int,int,int]:
        slot_name = f"card{slot_idx}"
        if not self.layout.has(slot_name):
            raise ValueError(f"Layout nu are {slot_name}")
        sx, sy = self._xy(slot_name)

        lane_name = lane if (isinstance(lane, str) and lane in ("left_lane", "right_lane")) else self._lane_name(lane)
        lx1, ly1, lx2, ly2 = self._lane_rect(lane_name, self._W, self._H)
        tx = (lx1 + lx2)//2
        ty = int(ly1*(1-y_bias) + ly2*y_bias)

        frame = self.ctrl.screenshot(as_gray=False)
        val = self.get_elixir_from_frame(frame)

        print(f"[ACTION] card{slot_idx} -> {lane_name} @ elixir={val}/10")
        if callable(self._action_cb):
            try: self._action_cb(slot_idx, lane_name, val)
            except Exception: pass

        self.ctrl.swipe(sx, sy, tx, ty, duration_ms=180)
        return sx, sy, tx, ty

    def play_left(self, slot_idx: int, y_bias: float = 0.65):
        return self.play_card_to_lane(slot_idx, "left_lane", y_bias=y_bias)

    def play_right(self, slot_idx: int, y_bias: float = 0.65):
        return self.play_card_to_lane(slot_idx, "right_lane", y_bias=y_bias)

    def _parse_action(self, action: Any) -> Tuple[Optional[int], Optional[str]]:
        if action is None: return (None, None)
        if isinstance(action, str):
            s = action.strip().lower()
            if s in ("","noop","none","skip"): return (None, None)
            m = re.findall(r"(\d)", s)
            slot = int(m[0]) if m else None
            lane = "left_lane" if ("left" in s or "l" in s) else ("right_lane" if ("right" in s or "r" in s) else None)
            if slot is not None and 1 <= slot <= 4: return (slot, lane or "left_lane")
            try: action = int(s)
            except Exception: return (None, None)
        if isinstance(action, int):
            if action < 0: return (None, None)
            if 0 <= action <= 7:
                lane = "left_lane" if action < 4 else "right_lane"
                return ((action % 4) + 1, lane)
            if 1 <= action <= 4: return (action, "left_lane")
            a = action % 8
            return ((a % 4) + 1, "left_lane" if a < 4 else "right_lane")
        if isinstance(action, (tuple,list)) and len(action) >= 2:
            slot, lane_raw = action[0], action[1]
            if isinstance(slot, str) and slot.isdigit(): slot = int(slot)
            if isinstance(slot, int):
                slot = slot if 1 <= slot <= 4 else (slot % 4) + 1
                return (slot, self._lane_name(lane_raw))
            return (None, None)
        if isinstance(action, dict):
            slot = action.get("slot"); lane_raw = action.get("lane","left")
            if isinstance(slot, str) and slot.isdigit(): slot = int(slot)
            if isinstance(slot, int):
                slot = slot if 1 <= slot <= 4 else (slot % 4) + 1
                return (slot, self._lane_name(lane_raw))
            return (None, None)
        return (None, None)

    def play_card(self, action: Any, ready: Optional[Iterable[bool]] = None, y_bias: float = 0.65) -> bool:
        slot, lane = self._parse_action(action)
        if slot is None or lane is None: return False
        if ready is not None:
            try:
                if len(ready) >= slot and not bool(ready[slot-1]): return False
            except Exception:
                pass
        self.play_card_to_lane(slot, lane, y_bias=y_bias)
        return True
