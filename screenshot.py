#!/usr/bin/env python3
"""
Screenshot capture — full screen (xdg-desktop-portal) or single window (X11 XGetImage).
Works on Wayland via XWayland. No extra packages needed.
"""

import ctypes
import dbus
import dbus.mainloop.glib
from gi.repository import GLib
import os
import sys
import tempfile


# ── Full-screen capture (xdg-desktop-portal) ───────────────────────

def take_screenshot(output_path: str = None) -> str:
    """
    Take a full-screen screenshot using the xdg-desktop-portal Screenshot interface.
    Returns the file path of the saved screenshot.
    """
    dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
    session_bus = dbus.SessionBus()

    portal_obj = session_bus.get_object(
        'org.freedesktop.portal.Desktop',
        '/org/freedesktop/portal/desktop'
    )
    portal_iface = dbus.Interface(
        portal_obj,
        'org.freedesktop.portal.Screenshot'
    )

    result = {'uri': None, 'done': False, 'error': None}

    def response_cb(response_code, results, path=None):
        if response_code == 0:
            result['uri'] = results.get('uri', '')
        else:
            result['error'] = f"Portal returned error code: {response_code}"
        result['done'] = True

    session_bus.add_signal_receiver(
        response_cb,
        signal_name='Response',
        dbus_interface='org.freedesktop.portal.Request'
    )

    token = f"screenshot_{os.getpid()}_{int(GLib.get_monotonic_time())}"
    options = {'handle_token': token, 'interactive': False}

    portal_iface.Screenshot('', options)

    loop = GLib.MainLoop()
    GLib.timeout_add_seconds(15, loop.quit)
    loop.run()

    if result['error']:
        raise RuntimeError(result['error'])
    if not result['uri']:
        raise RuntimeError("No screenshot URI received from portal")

    src_path = result['uri'].replace('file://', '')

    if output_path:
        import shutil
        shutil.copy(src_path, output_path)
        try:
            os.unlink(src_path)
        except OSError:
            pass
        return output_path

    return src_path


# ── Window-specific capture (X11 XGetImage via XWayland) ───────────

class X11WindowCapture:
    """Capture a specific window using X11 XGetImage. Works via XWayland."""

    def __init__(self, display: str = None):
        self.display_name = display or os.environ.get('DISPLAY', ':0')
        self._xlib = ctypes.CDLL('libX11.so.6')
        self._setup_signatures()
        self._display = self._xlib.XOpenDisplay(self.display_name.encode('utf-8'))
        if not self._display:
            raise RuntimeError(f"Cannot open display: {self.display_name}")
        self._screen = self._xlib.XDefaultScreen(self._display)
        self._root = self._xlib.XRootWindow(self._display, self._screen)

    def _setup_signatures(self):
        x = self._xlib
        x.XOpenDisplay.restype = ctypes.c_void_p
        x.XOpenDisplay.argtypes = [ctypes.c_char_p]
        x.XCloseDisplay.argtypes = [ctypes.c_void_p]
        x.XDefaultScreen.restype = ctypes.c_int
        x.XDefaultScreen.argtypes = [ctypes.c_void_p]
        x.XRootWindow.restype = ctypes.c_ulong
        x.XRootWindow.argtypes = [ctypes.c_void_p, ctypes.c_int]
        x.XQueryTree.restype = ctypes.c_int
        x.XQueryTree.argtypes = [
            ctypes.c_void_p, ctypes.c_ulong,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p
        ]
        x.XFetchName.restype = ctypes.c_int
        x.XFetchName.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_void_p]
        x.XFree.argtypes = [ctypes.c_void_p]
        x.XGetWindowAttributes.restype = ctypes.c_int
        x.XGetWindowAttributes.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_void_p]
        x.XTranslateCoordinates.restype = ctypes.c_int
        x.XTranslateCoordinates.argtypes = [
            ctypes.c_void_p, ctypes.c_ulong, ctypes.c_ulong,
            ctypes.c_int, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p
        ]
        x.XGetImage.restype = ctypes.c_void_p
        x.XGetImage.argtypes = [
            ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int, ctypes.c_int,
            ctypes.c_uint, ctypes.c_uint, ctypes.c_ulong, ctypes.c_int
        ]
        x.XDestroyImage.argtypes = [ctypes.c_void_p]
        x.XDefaultVisual.restype = ctypes.c_void_p
        x.XDefaultVisual.argtypes = [ctypes.c_void_p, ctypes.c_int]
        x.XDefaultDepth.restype = ctypes.c_int
        x.XDefaultDepth.argtypes = [ctypes.c_void_p, ctypes.c_int]
        x.XRaiseWindow.restype = ctypes.c_int
        x.XRaiseWindow.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        x.XSetInputFocus.restype = ctypes.c_int
        x.XSetInputFocus.argtypes = [
            ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong
        ]
        x.XFlush.argtypes = [ctypes.c_void_p]
        x.XSync.argtypes = [ctypes.c_void_p, ctypes.c_int]

    def _get_window_name(self, window: int) -> str:
        """Get the title/name of a window."""
        name_ptr = ctypes.c_char_p()
        status = self._xlib.XFetchName(self._display, window, ctypes.byref(name_ptr))
        if status and name_ptr.value:
            name = name_ptr.value.decode('utf-8', errors='replace')
            self._xlib.XFree(name_ptr)
            return name
        return ""

    def _get_window_geometry(self, window: int) -> dict:
        """Get window position (relative to root) and size."""
        class XWindowAttributes(ctypes.Structure):
            _fields_ = [
                ("x", ctypes.c_int), ("y", ctypes.c_int),
                ("width", ctypes.c_int), ("height", ctypes.c_int),
                ("border_width", ctypes.c_int), ("depth", ctypes.c_int),
                ("visual", ctypes.c_void_p), ("root", ctypes.c_ulong),
                ("class", ctypes.c_int), ("bit_gravity", ctypes.c_int),
                ("win_gravity", ctypes.c_int), ("backing_store", ctypes.c_int),
                ("backing_planes", ctypes.c_ulong), ("backing_pixel", ctypes.c_ulong),
                ("save_under", ctypes.c_int), ("colormap", ctypes.c_ulong),
                ("map_installed", ctypes.c_int), ("map_state", ctypes.c_int),
                ("all_event_masks", ctypes.c_ulong), ("your_event_mask", ctypes.c_ulong),
                ("do_not_propagate_mask", ctypes.c_ulong), ("override_redirect", ctypes.c_int),
                ("screen", ctypes.c_void_p),
            ]

        attrs = XWindowAttributes()
        status = self._xlib.XGetWindowAttributes(self._display, window, ctypes.byref(attrs))
        if not status:
            return None

        # Translate to root coordinates
        root_x = ctypes.c_int()
        root_y = ctypes.c_int()
        child = ctypes.c_ulong()
        self._xlib.XTranslateCoordinates(
            self._display, window, self._root,
            0, 0, ctypes.byref(root_x), ctypes.byref(root_y), ctypes.byref(child)
        )

        return {
            'x': root_x.value,
            'y': root_y.value,
            'width': attrs.width,
            'height': attrs.height,
            'map_state': attrs.map_state,  # 2 = viewable
        }

    def list_windows(self) -> list[dict]:
        """List all top-level windows with their names and geometry."""
        root_return = ctypes.c_ulong()
        parent_return = ctypes.c_ulong()
        children_return = ctypes.POINTER(ctypes.c_ulong)()
        nchildren_return = ctypes.c_uint()

        status = self._xlib.XQueryTree(
            self._display, self._root,
            ctypes.byref(root_return), ctypes.byref(parent_return),
            ctypes.byref(children_return), ctypes.byref(nchildren_return)
        )
        if not status:
            return []

        windows = []
        nchildren = nchildren_return.value
        for i in range(nchildren):
            win = children_return[i]
            name = self._get_window_name(win)
            geom = self._get_window_geometry(win)
            if geom and geom['map_state'] == 2 and geom['width'] > 100 and geom['height'] > 100:
                windows.append({
                    'id': win,
                    'name': name,
                    'x': geom['x'], 'y': geom['y'],
                    'width': geom['width'], 'height': geom['height'],
                })

        if nchildren > 0:
            self._xlib.XFree(children_return)
        return windows

    def find_window(self, title_substring: str) -> dict | None:
        """Find a window whose title contains the given substring (case-insensitive)."""
        windows = self.list_windows()
        title_lower = title_substring.lower()
        for w in windows:
            if title_lower in w['name'].lower():
                return w
        return None

    def raise_window(self, window_id: int):
        """
        Raise the window to the top of the stack and give it keyboard focus,
        so injected input (XTest) reaches it instead of whatever window is
        overlapping it. Raises if the window is not viewable.
        """
        geom = self._get_window_geometry(window_id)
        if not geom or geom.get('map_state') != 2:  # 2 = viewable
            raise RuntimeError(
                "Game window is not visible (minimized or closed) — restore it first."
            )
        self._xlib.XRaiseWindow(self._display, window_id)
        self._xlib.XSetInputFocus(self._display, window_id, 1, 0)  # revert_to=PointerRoot
        self._xlib.XFlush(self._display)
        self._xlib.XSync(self._display, 0)

    def capture_window(self, window_id: int, output_path: str) -> str:
        """
        Capture a specific window directly using X11 XGetImage.
        Handles byte order and row padding correctly.
        """
        geom = self._get_window_geometry(window_id)
        if not geom:
            raise RuntimeError(f"Cannot get geometry for window {window_id}")

        w, h = geom['width'], geom['height']
        print(f"  Window: {w}x{h} at ({geom['x']},{geom['y']})", file=sys.stderr)

        # XGetImage: ZPixmap = 2, capture the window's own pixels
        image_ptr = self._xlib.XGetImage(
            self._display, window_id,
            0, 0, w, h,
            0xFFFFFFFF,  # all planes
            2,  # ZPixmap
        )
        if not image_ptr:
            raise RuntimeError(f"XGetImage failed for window {window_id}")

        # XImage struct layout for x86_64
        class XImage(ctypes.Structure):
            _fields_ = [
                ("width", ctypes.c_int), ("height", ctypes.c_int),
                ("xoffset", ctypes.c_int), ("format", ctypes.c_int),
                ("data", ctypes.c_void_p), ("byte_order", ctypes.c_int),
                ("bitmap_unit", ctypes.c_int), ("bitmap_bit_order", ctypes.c_int),
                ("bitmap_pad", ctypes.c_int), ("depth", ctypes.c_int),
                ("bytes_per_line", ctypes.c_int), ("bits_per_pixel", ctypes.c_int),
                ("red_mask", ctypes.c_ulong), ("green_mask", ctypes.c_ulong),
                ("blue_mask", ctypes.c_ulong),
            ]

        img = ctypes.cast(image_ptr, ctypes.POINTER(XImage)).contents
        bpp = img.bits_per_pixel
        bpl = img.bytes_per_line
        print(f"  XImage: {img.width}x{img.height}, {bpp}bpp, {bpl} bytes/line, depth={img.depth}", file=sys.stderr)

        # Copy raw pixel data
        raw = ctypes.string_at(img.data, bpl * img.height)

        from PIL import Image

        if bpp == 32:
            # 32bpp: typically BGRX on little-endian x86_64 X11
            # bytes_per_line may include padding — handle row by row
            pil_img = Image.new('RGB', (w, h))
            pixels = pil_img.load()
            for row in range(h):
                row_offset = row * bpl
                for col in range(w):
                    offset = row_offset + col * 4
                    b = raw[offset]
                    g = raw[offset + 1]
                    r = raw[offset + 2]
                    pixels[col, row] = (r, g, b)
        elif bpp == 24:
            # 24bpp: BGR, 3 bytes per pixel, may have row padding
            pil_img = Image.new('RGB', (w, h))
            pixels = pil_img.load()
            for row in range(h):
                row_offset = row * bpl
                for col in range(w):
                    offset = row_offset + col * 3
                    b = raw[offset]
                    g = raw[offset + 1]
                    r = raw[offset + 2]
                    pixels[col, row] = (r, g, b)
        else:
            # Fallback: try PIL raw decoder
            bytes_pp = bpp // 8
            if bytes_pp * w == bpl:
                # No padding
                if bpp == 32:
                    pil_img = Image.frombytes('RGB', (w, h), raw, 'raw', 'BGRX')
                else:
                    pil_img = Image.frombytes('RGB', (w, h), raw, 'raw', 'BGR')
            else:
                raise RuntimeError(f"Unsupported bpp={bpp}, bpl={bpl}, w={w}")

        pil_img.save(output_path, 'PNG')
        print(f"  Saved: {pil_img.width}x{pil_img.height}", file=sys.stderr)

        self._xlib.XDestroyImage(image_ptr)
        return output_path

    def capture_by_title(self, title_substring: str, output_path: str) -> str:
        """Find a window by title and capture it."""
        win = self.find_window(title_substring)
        if not win:
            raise RuntimeError(f"No window found matching '{title_substring}'")
        return self.capture_window(win['id'], output_path)

    def close(self):
        if self._display:
            self._xlib.XCloseDisplay(self._display)
            self._display = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def take_window_screenshot(title_substring: str, output_path: str = None) -> str:
    """
    Capture a specific window by title substring.
    Falls back to full-screen if window not found.
    Returns the file path of the saved screenshot.
    """
    if output_path is None:
        output_path = os.path.join(
            tempfile.gettempdir(), f'window_shot_{int(__import__("time").time())}.png'
        )

    print(f"  Looking for window matching '{title_substring}'...", file=sys.stderr)
    try:
        with X11WindowCapture() as cap:
            win = cap.find_window(title_substring)
            if not win:
                print(f"  Window '{title_substring}' not found. Available windows:", file=sys.stderr)
                for w in cap.list_windows():
                    print(f"    \"{w['name']}\" {w['width']}x{w['height']}", file=sys.stderr)
                raise RuntimeError(f"Window '{title_substring}' not found")
            print(f"  Found: \"{win['name']}\" {win['width']}x{win['height']} at ({win['x']},{win['y']})", file=sys.stderr)
            return cap.capture_window(win['id'], output_path)
    except Exception as e:
        print(f"  Window capture failed: {e}", file=sys.stderr)
        print(f"  Falling back to full screen...", file=sys.stderr)
        return take_screenshot(output_path)


if __name__ == '__main__':
    # CLI: list windows or capture by title
    if len(sys.argv) > 1 and sys.argv[1] == '--list':
        with X11WindowCapture() as cap:
            windows = cap.list_windows()
            print(f"Found {len(windows)} windows:")
            for w in windows:
                print(f"  [{w['id']}] \"{w['name']}\" {w['width']}x{w['height']} at ({w['x']},{w['y']})")
    elif len(sys.argv) > 1:
        # Capture window by title
        title = sys.argv[1]
        output = sys.argv[2] if len(sys.argv) > 2 else '/tmp/window_screenshot.png'
        path = take_window_screenshot(title, output)
        print(path)
    else:
        # Full screen
        output = '/tmp/screenshot.png'
        path = take_screenshot(output)
        print(path)
