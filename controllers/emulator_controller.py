import subprocess
from config import settings


class EmulatorController:
    """Fallback generic ADB controller (rămâne util pentru test)."""
    def __init__(self, serial: str | None = None):
        self.serial = serial or settings.DEVICE_SERIAL


    def _adb(self, *args: str) -> bytes:
        return subprocess.check_output(["adb", "-s", self.serial, *args])


    def tap(self, x: int, y: int) -> None:
        self._adb("shell", "input", "tap", str(x), str(y))


    def swipe(self, x1: int, y1: int, x2: int, y2: int, dur_ms: int | None=None):
        dur_ms = dur_ms or settings.SWIPE_DURATION_MS
        self._adb("shell", "input", "swipe", str(x1), str(y1), str(x2), str(y2), str(dur_ms))


    def screenshot_png(self) -> bytes:
        return self._adb("exec-out", "screencap", "-p")