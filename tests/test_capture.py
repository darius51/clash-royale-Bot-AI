from controllers.emulator_controller import EmulatorController
from vision.capture import ScreenCapture


def test_grab():
    ctrl = EmulatorController()
    cap = ScreenCapture(ctrl)
    img = cap.grab()
    assert img is not None and img.shape[0] > 0