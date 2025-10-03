# controllers/memu_adapter.py
from __future__ import annotations
import os, subprocess, re, time, tempfile, shutil
from typing import Optional, Tuple, Dict

import numpy as np
import cv2

try:
    from config import ADB_PATH
except Exception:
    ADB_PATH = None
try:
    from config import ADB_DEVICE_ID
except Exception:
    ADB_DEVICE_ID = None
try:
    from config import CLASH_PKG
except Exception:
    CLASH_PKG = None

def _run(cmd: list[str], capture: bool = True, timeout: int = 30) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=capture, timeout=timeout, check=False)

def _resolve_adb_path(explicit: Optional[str]) -> str:
    if explicit:
        p = os.path.expandvars(explicit)
        if os.path.isfile(p): return p
    wh = shutil.which("adb")
    if wh and os.path.isfile(wh): return wh
    cands = []
    for env in ("ANDROID_HOME", "ANDROID_SDK_ROOT"):
        sdk = os.environ.get(env)
        if sdk:
            cands += [os.path.join(sdk, "platform-tools", "adb.exe"),
                      os.path.join(sdk, "platform-tools", "adb")]
    cands += [
        r"C:\Android\platform-tools\adb.exe",
        r"C:\Program Files\Android\Android Studio\platform-tools\adb.exe",
        r"C:\Program Files (x86)\Android\android-sdk\platform-tools\adb.exe",
        os.path.expandvars(r"C:\Users\%USERNAME%\AppData\Local\Android\Sdk\platform-tools\adb.exe"),
        r"C:\Program Files\Microvirt\MEmu\adb.exe",
        r"C:\Program Files\Microvirt\MEmuHyperv\adb.exe",
        r"C:\Program Files\Netease\MuMu\emulator\nemu\adb.exe",
    ]
    for p in cands:
        if os.path.isfile(p): return p
    return "adb"

def _ensure_server(adb_path: str) -> None:
    _run([adb_path, "kill-server"], capture=False)
    _run([adb_path, "start-server"], capture=False)

def _devices_map(adb_path: str) -> Dict[str, str]:
    proc = _run([adb_path, "devices"])
    out = proc.stdout.decode(errors="ignore").splitlines()
    mp: Dict[str, str] = {}
    for ln in out[1:]:
        ln = ln.strip()
        if not ln: continue
        if "\t" in ln:
            serial, state = ln.split("\t", 1)
            mp[serial.strip()] = state.strip()
    return mp

def _memu_autoconnect(adb_path: str) -> None:
    for p in [21503, 21513, 21523, 21533, 21543, 21553]:
        _run([adb_path, "connect", f"127.0.0.1:{p}"], capture=False)
        time.sleep(0.1)

def _ensure_connected_device(adb_path: str, device_id: str) -> bool:
    devs = _devices_map(adb_path)
    if devs.get(device_id) == "device": return True
    if ":" in device_id:
        _run([adb_path, "connect", device_id], capture=False)
        time.sleep(0.2)
        devs = _devices_map(adb_path)
        if devs.get(device_id) == "device": return True
    _memu_autoconnect(adb_path)
    devs = _devices_map(adb_path)
    return devs.get(device_id) == "device"

def _adb_base(adb_path: str, device_id: Optional[str]) -> list[str]:
    adb = [adb_path]
    if device_id: adb += ["-s", device_id]
    return adb

class MemuAdapter:
    """ ADB adapter: screenshot, tap, swipe, keyevent, rezoluție + utilitare Clash. """
    def __init__(self, device_id: Optional[str] = None, adb_path: Optional[str] = None,
                 render_mode: Optional[str] = None, **_ignored):
        self.adb_path = _resolve_adb_path(adb_path or ADB_PATH)
        try:
            v = _run([self.adb_path, "version"])
        except FileNotFoundError as e:
            raise RuntimeError("Nu am găsit adb. Setează ADB_PATH în config.py.") from e
        if v.returncode != 0:
            raise RuntimeError(f"Eșec '{self.adb_path} version'. Verifică instalarea/calea ADB.")
        _ensure_server(self.adb_path)

        self.device_id = device_id or ADB_DEVICE_ID
        if self.device_id: _ensure_connected_device(self.adb_path, self.device_id)
        if not self.device_id:
            _memu_autoconnect(self.adb_path)
            mp = _devices_map(self.adb_path)
            self.device_id = next((k for k, s in mp.items() if s == "device"), None)
        if not self.device_id or _devices_map(self.adb_path).get(self.device_id) != "device":
            listing = _run([self.adb_path, "devices", "-l"]).stdout.decode(errors="ignore")
            raise RuntimeError(f"Nu s-a găsit device ADB.\n[ADB] {self.adb_path}\n{listing}")

        self.render_mode = render_mode  # doar compat

    def _ensure_ready(self) -> None:
        if not _ensure_connected_device(self.adb_path, self.device_id):
            listing = _run([self.adb_path, "devices", "-l"]).stdout.decode(errors="ignore")
            raise RuntimeError(f"Device indisponibil: {self.device_id}\n{listing}")

    # --- App ---
    def is_device_ready(self) -> bool:
        self._ensure_ready()
        proc = _run(_adb_base(self.adb_path, self.device_id) + ["get-state"])
        return proc.stdout.decode(errors="ignore").strip() == "device"

    def start_app(self, pkg: Optional[str] = None) -> None:
        self._ensure_ready()
        pkg = pkg or CLASH_PKG
        if not pkg: return
        _run(_adb_base(self.adb_path, self.device_id) + [
            "shell","monkey","-p",pkg,"-c","android.intent.category.LAUNCHER","1"
        ], capture=True)  # capture=True ca să nu-ți inunde consola

    def is_app_foreground(self, pkg: Optional[str] = None) -> bool:
        self._ensure_ready()
        pkg = pkg or CLASH_PKG
        if not pkg: return False
        proc = _run(_adb_base(self.adb_path, self.device_id) + ["shell","dumpsys","activity","activities"])
        return (pkg in proc.stdout.decode(errors="ignore"))

    def ensure_clash(self, pkg: Optional[str] = None, max_tries: int = 3, wait_after: float = 2.0) -> bool:
        self._ensure_ready()
        pkg = pkg or CLASH_PKG or "com.supercell.clashroyale"
        if self.is_app_foreground(pkg): return True
        for _ in range(max_tries):
            self.start_app(pkg); time.sleep(wait_after)
            if self.is_app_foreground(pkg): return True
        return self.is_app_foreground(pkg)

    # --- Input ---
    def tap(self, x: int, y: int) -> None:
        self._ensure_ready()
        _run(_adb_base(self.adb_path, self.device_id) + ["shell","input","tap",str(int(x)),str(int(y))], capture=False)

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 200) -> None:
        self._ensure_ready()
        _run(_adb_base(self.adb_path, self.device_id) + ["shell","input","swipe",
            str(int(x1)),str(int(y1)),str(int(x2)),str(int(y2)),str(int(duration_ms))], capture=False)

    def keyevent(self, keycode: int) -> None:
        self._ensure_ready()
        _run(_adb_base(self.adb_path, self.device_id) + ["shell","input","keyevent",str(int(keycode))], capture=False)

    back = lambda self: self.keyevent(4)
    home = lambda self: self.keyevent(3)

    # --- Screen ---
    def screenshot(self, as_gray: bool = False) -> np.ndarray:
        self._ensure_ready()
        proc = _run(_adb_base(self.adb_path, self.device_id) + ["exec-out","screencap","-p"], capture=True, timeout=30)
        data = proc.stdout
        if not data:
            tmp_remote = "/sdcard/__tmp_screen.png"
            _run(_adb_base(self.adb_path, self.device_id) + ["shell","screencap","-p",tmp_remote], capture=False)
            with tempfile.TemporaryDirectory() as td:
                local = os.path.join(td,"screen.png")
                _run(_adb_base(self.adb_path, self.device_id) + ["pull",tmp_remote,local], capture=False)
                _run(_adb_base(self.adb_path, self.device_id) + ["shell","rm","-f",tmp_remote], capture=False)
                if not os.path.isfile(local):
                    listing = _run([self.adb_path,"devices","-l"]).stdout.decode(errors="ignore")
                    raise RuntimeError("ADB pull a eșuat; device căzut?\n"+listing)
                with open(local,"rb") as f: data = f.read()
        img = np.frombuffer(data, dtype=np.uint8)
        flag = cv2.IMREAD_GRAYSCALE if as_gray else cv2.IMREAD_COLOR
        img = cv2.imdecode(img, flag)
        if img is None: raise RuntimeError("cv2.imdecode a returnat None (screenshot corupt).")
        return img

    # aliasuri compat
    def screencap(self, as_gray: bool = False): return self.screenshot(as_gray=as_gray)
    def screenshot_np(self, gray: bool = False): return self.screenshot(as_gray=gray)
    def get_frame(self): return self.screenshot(as_gray=False)
    def get_gray(self): return self.screenshot(as_gray=True)
    def screenshot_bgr(self): return self.screenshot(as_gray=False)
    def screenshot_rgb(self):
        img = self.screenshot(as_gray=False)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def get_device_resolution(self) -> Tuple[int, int]:
        self._ensure_ready()
        proc = _run(_adb_base(self.adb_path, self.device_id) + ["shell","wm","size"])
        txt = proc.stdout.decode(errors="ignore")
        m = re.search(r"(Override|Physical)\s+size:\s*(\d+)\s*x\s*(\d+)", txt)
        if m: return (int(m.group(2)), int(m.group(3)))
        img = self.screenshot(as_gray=True); h, w = img.shape[:2]
        return (w, h)

    def close(self) -> None: pass
