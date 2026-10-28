#!/usr/bin/env python3
"""
Input controller — mouse and keyboard injection via X11 XTest extension.
Works through XWayland on Wayland. No external tools needed (uses ctypes + libX11/libXtst).

Supports: mouse move, click, double-click, drag, scroll, key press, key type.
"""

import ctypes
import ctypes.util
import time
import os

# X11 constants
Button1 = 1  # Left click
Button2 = 2  # Middle click
Button3 = 3  # Right click
Button4 = 4  # Scroll up
Button5 = 5  # Scroll down

# XTest event types
MotionNotify = 6
ButtonPress = 4
ButtonRelease = 5
KeyPress = 2
KeyRelease = 3


class X11Input:
    """Low-level X11 input injection via XTest."""

    def __init__(self, display: str = None):
        self.display_name = display or os.environ.get('DISPLAY', ':0')
        self._xlib = ctypes.CDLL('libX11.so.6')
        self._xtst = ctypes.CDLL('libXtst.so.6')

        # Set up function signatures
        self._setup_signatures()

        # Open display
        self._display = self._xlib.XOpenDisplay(
            self.display_name.encode('utf-8')
        )
        if not self._display:
            raise RuntimeError(f"Cannot open display: {self.display_name}")

        self._screen = self._xlib.XDefaultScreen(self._display)
        self._root = self._xlib.XRootWindow(self._display, self._screen)

        # Get screen dimensions
        self.screen_width = self._xlib.XDisplayWidth(self._display, self._screen)
        self.screen_height = self._xlib.XDisplayHeight(self._display, self._screen)

        # XTEST is required for input injection — fail loudly if it's missing,
        # otherwise every fake-input call below would fail silently
        opcode = ctypes.c_int()
        first_event = ctypes.c_int()
        first_error = ctypes.c_int()
        if not self._xlib.XQueryExtension(
            self._display, b"XTEST",
            ctypes.byref(opcode), ctypes.byref(first_event), ctypes.byref(first_error),
        ):
            self._xlib.XCloseDisplay(self._display)
            self._display = None
            raise RuntimeError(
                f"XTEST extension not available on display {self.display_name} — "
                "cannot inject mouse/keyboard input"
            )

    def _setup_signatures(self):
        """Define ctypes function signatures for X11/XTest."""
        x = self._xlib
        xt = self._xtst

        # X11 functions
        x.XOpenDisplay.restype = ctypes.c_void_p
        x.XOpenDisplay.argtypes = [ctypes.c_char_p]
        x.XCloseDisplay.argtypes = [ctypes.c_void_p]
        x.XDefaultScreen.restype = ctypes.c_int
        x.XDefaultScreen.argtypes = [ctypes.c_void_p]
        x.XRootWindow.restype = ctypes.c_ulong
        x.XRootWindow.argtypes = [ctypes.c_void_p, ctypes.c_int]
        x.XDisplayWidth.restype = ctypes.c_int
        x.XDisplayWidth.argtypes = [ctypes.c_void_p, ctypes.c_int]
        x.XDisplayHeight.restype = ctypes.c_int
        x.XDisplayHeight.argtypes = [ctypes.c_void_p, ctypes.c_int]
        x.XFlush.argtypes = [ctypes.c_void_p]
        x.XSync.argtypes = [ctypes.c_void_p, ctypes.c_int]
        x.XWarpPointer.argtypes = [
            ctypes.c_void_p, ctypes.c_ulong, ctypes.c_ulong,
            ctypes.c_int, ctypes.c_int, ctypes.c_uint, ctypes.c_uint,
            ctypes.c_int, ctypes.c_int
        ]
        x.XQueryPointer.restype = ctypes.c_int
        x.XQueryPointer.argtypes = [
            ctypes.c_void_p, ctypes.c_ulong, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p
        ]
        x.XQueryExtension.restype = ctypes.c_int
        x.XQueryExtension.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]

        # XTest functions
        xt.XTestFakeMotionEvent.restype = ctypes.c_int
        xt.XTestFakeMotionEvent.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_ulong
        ]
        xt.XTestFakeButtonEvent.restype = ctypes.c_int
        xt.XTestFakeButtonEvent.argtypes = [
            ctypes.c_void_p, ctypes.c_uint, ctypes.c_int, ctypes.c_ulong
        ]
        xt.XTestFakeKeyEvent.restype = ctypes.c_int
        xt.XTestFakeKeyEvent.argtypes = [
            ctypes.c_void_p, ctypes.c_uint, ctypes.c_int, ctypes.c_ulong
        ]

    def _flush(self):
        self._xlib.XFlush(self._display)
        self._xlib.XSync(self._display, 0)

    def _fake_motion(self, x: int, y: int):
        """XTestFakeMotionEvent — raises if the server rejects it."""
        if not self._xtst.XTestFakeMotionEvent(self._display, self._screen, x, y, 0):
            raise RuntimeError(
                f"XTestFakeMotionEvent to ({x},{y}) failed — input injection "
                "is not working on this display"
            )

    def _fake_button(self, button: int, press: bool):
        """XTestFakeButtonEvent — raises if the server rejects it."""
        if not self._xtst.XTestFakeButtonEvent(
            self._display, button, 1 if press else 0, 0
        ):
            raise RuntimeError(
                f"XTestFakeButtonEvent (button {button}, "
                f"{'press' if press else 'release'}) failed — input injection "
                "is not working on this display"
            )

    def _fake_key(self, keycode: int, press: bool):
        """XTestFakeKeyEvent — raises if the server rejects it."""
        if not self._xtst.XTestFakeKeyEvent(
            self._display, keycode, 1 if press else 0, 0
        ):
            raise RuntimeError(
                f"XTestFakeKeyEvent (keycode {keycode}) failed — input injection "
                "is not working on this display"
            )

    # ── Mouse ──────────────────────────────────────────────────────

    def move(self, x: int, y: int, smooth: bool = False, steps: int = 15):
        """Move mouse cursor to (x, y). If smooth, animate the movement."""
        if smooth:
            # Get current position
            root_ret = ctypes.c_ulong()
            child_ret = ctypes.c_ulong()
            root_x = ctypes.c_int()
            root_y = ctypes.c_int()
            win_x = ctypes.c_int()
            win_y = ctypes.c_int()
            mask = ctypes.c_uint()

            self._xlib.XQueryPointer(
                self._display, self._root,
                ctypes.byref(root_ret), ctypes.byref(child_ret),
                ctypes.byref(root_x), ctypes.byref(root_y),
                ctypes.byref(win_x), ctypes.byref(win_y),
                ctypes.byref(mask)
            )
            start_x, start_y = root_x.value, root_y.value

            for i in range(1, steps + 1):
                t = i / steps
                # Ease-in-out
                t = t * t * (3 - 2 * t)
                cx = int(start_x + (x - start_x) * t)
                cy = int(start_y + (y - start_y) * t)
                self._fake_motion(cx, cy)
                self._flush()
                time.sleep(0.01)
        else:
            self._fake_motion(x, y)
            self._flush()

    def click(self, x: int = None, y: int = None, button: int = Button1,
              double: bool = False):
        """Click at (x, y) or current position. button: 1=left, 2=middle, 3=right."""
        if x is not None and y is not None:
            self.move(x, y)
        self._fake_button(button, True)
        self._flush()
        time.sleep(0.02)
        self._fake_button(button, False)
        self._flush()
        if double:
            time.sleep(0.05)
            self._fake_button(button, True)
            self._flush()
            time.sleep(0.02)
            self._fake_button(button, False)
            self._flush()

    def drag(self, start_x: int, start_y: int, end_x: int, end_y: int,
             button: int = Button1, duration: float = 0.8, steps: int = 30):
        """Drag from (start_x, start_y) to (end_x, end_y).

        Uses smooth movement to the start position and deliberate timing
        (longer pauses) so card games register the grab, drag, and drop.
        """
        # Smooth move to start so the game registers the hover
        self.move(start_x, start_y, smooth=True, steps=10)
        time.sleep(0.3)  # let the game show hover highlight
        # Press button
        self._fake_button(button, True)
        self._flush()
        time.sleep(0.15)  # let the game register the grab
        # Move in steps with ease-in-out
        for i in range(1, steps + 1):
            t = i / steps
            t = t * t * (3 - 2 * t)  # ease-in-out
            cx = int(start_x + (end_x - start_x) * t)
            cy = int(start_y + (end_y - start_y) * t)
            self._fake_motion(cx, cy)
            self._flush()
            time.sleep(duration / steps)
        # Release button — pause so the game registers the drop
        time.sleep(0.15)
        self._fake_button(button, False)
        self._flush()
        time.sleep(0.1)
        # Verify the pointer actually moved — catches silent XTest failure
        px, py = self.get_pointer_position()
        if abs(px - end_x) > 5 or abs(py - end_y) > 5:
            raise RuntimeError(
                f"Pointer did not reach ({end_x},{end_y}) — it is at ({px},{py}). "
                "XTest input injection is not working on this display."
            )

    def scroll(self, clicks: int, direction: str = 'up'):
        """Scroll. direction: 'up' or 'down'. Positive clicks = scroll up."""
        button = Button4 if direction == 'up' else Button5
        for _ in range(abs(clicks)):
            self._fake_button(button, True)
            self._flush()
            time.sleep(0.02)
            self._fake_button(button, False)
            self._flush()
            time.sleep(0.02)

    # ── Keyboard ───────────────────────────────────────────────────

    def _keysym(self, key: str) -> int:
        """Convert key name to X11 keysym."""
        keysyms = {
            'enter': 0xff0d, 'return': 0xff0d,
            'space': 0x0020,
            'tab': 0xff09,
            'escape': 0xff1b, 'esc': 0xff1b,
            'backspace': 0xff08,
            'delete': 0xffff,
            'up': 0xff52, 'down': 0xff54,
            'left': 0xff51, 'right': 0xff53,
            'home': 0xff50, 'end': 0xff57,
            'pageup': 0xff55, 'pagedown': 0xff56,
            'shift': 0xffe1, 'ctrl': 0xffe3, 'alt': 0xffe9,
            'f1': 0xffbe, 'f2': 0xffbf, 'f3': 0xffc0, 'f4': 0xffc1,
            'f5': 0xffc2, 'f6': 0xffc3, 'f7': 0xffc4, 'f8': 0xffc5,
        }
        if key.lower() in keysyms:
            return keysyms[key.lower()]
        # Single character — use its Unicode code point
        if len(key) == 1:
            return ord(key)
        raise ValueError(f"Unknown key: {key}")

    def key(self, key: str, hold: float = 0.05):
        """Press and release a key."""
        keysym = self._keysym(key)
        keycode = self._xlib.XKeysymToKeycode(self._display, keysym)
        self._fake_key(keycode, True)
        self._flush()
        time.sleep(hold)
        self._fake_key(keycode, False)
        self._flush()

    def type_text(self, text: str, interval: float = 0.03):
        """Type a string of text."""
        for char in text:
            keysym = ord(char) if len(char) == 1 else self._keysym(char)
            keycode = self._xlib.XKeysymToKeycode(self._display, keysym)
            self._fake_key(keycode, True)
            self._flush()
            time.sleep(0.01)
            self._fake_key(keycode, False)
            self._flush()
            time.sleep(interval)

    # ── Window info ────────────────────────────────────────────────

    def get_pointer_position(self) -> tuple[int, int]:
        """Get current mouse position."""
        root_ret = ctypes.c_ulong()
        child_ret = ctypes.c_ulong()
        root_x = ctypes.c_int()
        root_y = ctypes.c_int()
        win_x = ctypes.c_int()
        win_y = ctypes.c_int()
        mask = ctypes.c_uint()

        self._xlib.XQueryPointer(
            self._display, self._root,
            ctypes.byref(root_ret), ctypes.byref(child_ret),
            ctypes.byref(root_x), ctypes.byref(root_y),
            ctypes.byref(win_x), ctypes.byref(win_y),
            ctypes.byref(mask)
        )
        return root_x.value, root_y.value

    def close(self):
        """Close the X11 display connection."""
        if self._display:
            self._xlib.XCloseDisplay(self._display)
            self._display = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# High-level action helpers

def click_at(x: int, y: int, double: bool = False):
    """Quick click at screen coordinates."""
    with X11Input() as inp:
        inp.click(x, y, double=double)


def drag_from_to(start_x: int, start_y: int, end_x: int, end_y: int,
                 duration: float = 0.5):
    """Quick drag between two screen coordinates."""
    with X11Input() as inp:
        inp.drag(start_x, start_y, end_x, end_y, duration=duration)


def move_mouse(x: int, y: int):
    """Quick mouse move."""
    with X11Input() as inp:
        inp.move(x, y)


if __name__ == '__main__':
    # Self-test: move mouse in a small square pattern
    import sys
    print(f"Screen: {X11Input().screen_width}x{X11Input().screen_height}")
    print("Testing mouse movement (watch your cursor)...")
    with X11Input() as inp:
        cx, cy = inp.screen_width // 2, inp.screen_height // 2
        inp.move(cx, cy, smooth=True)
        time.sleep(0.3)
        inp.move(cx + 100, cy, smooth=True)
        time.sleep(0.3)
        inp.move(cx + 100, cy + 100, smooth=True)
        time.sleep(0.3)
        inp.move(cx, cy + 100, smooth=True)
        time.sleep(0.3)
        inp.move(cx, cy, smooth=True)
        print("Mouse test complete. Cursor returned to center.")
