#!/usr/bin/env python3
"""
Card Game Advisor — native GTK4 desktop UI.

Window selector, screenshot preview, AI move advice, a prompt library
(titled prompts saved to prompts.json), and auto-move execution.
No website, no browser — just a desktop app.

Usage:
    python3 gui.py
    ./play.sh --gui
"""

import os
import sys
import tempfile
import threading
import time

import gi
gi.require_version('Gtk', '4.0')
gi.require_version('Gdk', '4.0')
gi.require_version('GdkPixbuf', '2.0')
from gi.repository import Gtk, Gdk, GdkPixbuf, GLib, Gio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from screenshot import X11WindowCapture
from mistral_client import ask_best_move_parsed, ensure_prompts_file, save_prompts
from move_parser import (Action, execute_actions, offset_actions,
                        extract_boxes, drag_to_clicks)
from move_detector import detect_move, build_system_prompt, build_user_prompt
from overlay import draw_move_overlay, move_changed_board
from input_controller import X11Input


class AdvisorWindow(Gtk.ApplicationWindow):
    def __init__(self, app, api_key: str):
        super().__init__(application=app, title="Card Game Advisor")
        self.api_key = api_key
        self.set_default_size(1000, 760)

        self._capture = None  # X11WindowCapture instance
        self._windows = []    # list of window dicts
        self._busy = False
        self._playing = False       # auto-play loop is running
        self._stop_requested = False  # signal to stop the auto-play loop

        self._prompts: list[dict] = []          # [{'title': str, 'text': str}, ...]
        self._active_prompt: str | None = None
        self._last_move = None  # {'win', 'actions', 'advice', 'move_text', 'detection'}

        self._build_ui()
        self._load_prompts()
        self._refresh_windows()

    def _build_ui(self):
        # ── Main layout ──
        main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        main_box.set_margin_top(12)
        main_box.set_margin_bottom(12)
        main_box.set_margin_start(12)
        main_box.set_margin_end(12)
        self.set_child(main_box)

        # ── Top bar: window selector ──
        top_bar = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        main_box.append(top_bar)

        top_bar.append(Gtk.Label(label="Window:"))

        self.window_model = Gio.ListStore(item_type=Gtk.StringObject)
        self.window_dropdown = Gtk.DropDown(model=self.window_model)
        self.window_dropdown.set_hexpand(True)
        top_bar.append(self.window_dropdown)

        refresh_btn = Gtk.Button(label="Refresh")
        refresh_btn.connect("clicked", self._on_refresh_windows)
        top_bar.append(refresh_btn)

        # ── Action buttons ──
        btn_bar = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        main_box.append(btn_bar)

        self.capture_btn = Gtk.Button(label="Capture & Advise")
        self.capture_btn.connect("clicked", self._on_capture)
        self.capture_btn.set_hexpand(True)
        self.capture_btn.get_style_context().add_class("suggested-action")
        btn_bar.append(self.capture_btn)

        self.dry_run_btn = Gtk.Button(label="Dry Run")
        self.dry_run_btn.connect("clicked", self._on_dry_run)
        btn_bar.append(self.dry_run_btn)

        self.execute_btn = Gtk.Button(label="Execute Move")
        self.execute_btn.connect("clicked", self._on_execute)
        self.execute_btn.get_style_context().add_class("suggested-action")
        btn_bar.append(self.execute_btn)

        self.reexecute_btn = Gtk.Button(label="Re-execute Last Move")
        self.reexecute_btn.connect("clicked", self._on_reexecute)
        self.reexecute_btn.set_tooltip_text("No move received yet")
        btn_bar.append(self.reexecute_btn)

        self.play_btn = Gtk.Button(label="Play")
        self.play_btn.connect("clicked", self._on_play)
        self.play_btn.get_style_context().add_class("suggested-action")
        self.play_btn.set_tooltip_text("Auto-play in a loop (5s between moves)")
        btn_bar.append(self.play_btn)

        self.stop_btn = Gtk.Button(label="Stop")
        self.stop_btn.connect("clicked", self._on_stop)
        self.stop_btn.get_style_context().add_class("destructive-action")
        self.stop_btn.set_sensitive(False)
        btn_bar.append(self.stop_btn)

        # ── Nudge: fine aim correction applied to every executed move ──
        nudge_bar = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        main_box.append(nudge_bar)

        nudge_bar.append(Gtk.Label(label="Nudge (px):"))

        self.nudge_x_spin = Gtk.SpinButton.new_with_range(-200, 200, 1)
        self.nudge_x_spin.set_value(0)
        self.nudge_x_spin.set_tooltip_text(
            "Horizontal correction added to every executed move (screen pixels)"
        )
        nudge_bar.append(self.nudge_x_spin)

        self.nudge_y_spin = Gtk.SpinButton.new_with_range(-200, 200, 1)
        self.nudge_y_spin.set_value(0)
        self.nudge_y_spin.set_tooltip_text(
            "Vertical correction added to every executed move (screen pixels)"
        )
        nudge_bar.append(self.nudge_y_spin)

        nudge_reset = Gtk.Button(label="Reset")
        nudge_reset.connect("clicked", self._on_nudge_reset)
        nudge_bar.append(nudge_reset)

        # ── Reasoning level: None / Max ──
        nudge_bar.append(Gtk.Label(label="  Reasoning:"))
        self.reasoning_model = Gio.ListStore(item_type=Gtk.StringObject)
        self.reasoning_model.append(Gtk.StringObject.new("None (fast)"))
        self.reasoning_model.append(Gtk.StringObject.new("Max (think)"))
        self.reasoning_dropdown = Gtk.DropDown(model=self.reasoning_model)
        self.reasoning_dropdown.set_selected(0)
        self.reasoning_dropdown.set_tooltip_text(
            "AI reasoning level for move advice: None = fast, Max = extended thinking"
        )
        nudge_bar.append(self.reasoning_dropdown)

        # ── Prompt bar: switch / create / save prompts ──
        prompt_bar = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        main_box.append(prompt_bar)

        prompt_bar.append(Gtk.Label(label="Prompt:"))

        self.prompt_model = Gio.ListStore(item_type=Gtk.StringObject)
        self.prompt_dropdown = Gtk.DropDown(model=self.prompt_model)
        self.prompt_dropdown.set_hexpand(True)
        self.prompt_dropdown.connect("notify::selected", self._on_prompt_selected)
        prompt_bar.append(self.prompt_dropdown)

        self.prompt_title_entry = Gtk.Entry(placeholder_text="new prompt title")
        self.prompt_title_entry.set_width_chars(18)
        prompt_bar.append(self.prompt_title_entry)

        new_btn = Gtk.Button(label="New")
        new_btn.connect("clicked", self._on_new_prompt)
        prompt_bar.append(new_btn)

        save_btn = Gtk.Button(label="Save")
        save_btn.connect("clicked", self._on_save_prompt)
        prompt_bar.append(save_btn)

        # ── Prompt editor (collapsible) ──
        editor_expander = Gtk.Expander(label="Edit prompt")
        main_box.append(editor_expander)
        editor_scroll = Gtk.ScrolledWindow()
        editor_scroll.set_min_content_height(100)
        editor_scroll.set_max_content_height(220)
        editor_expander.set_child(editor_scroll)

        self.prompt_view = Gtk.TextView()
        self.prompt_view.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)
        self.prompt_view.set_monospace(True)
        self.prompt_view.set_top_margin(8)
        self.prompt_view.set_bottom_margin(8)
        self.prompt_view.set_left_margin(8)
        self.prompt_view.set_right_margin(8)
        editor_scroll.set_child(self.prompt_view)

        # ── Status label ──
        self.status_label = Gtk.Label(label="Select a window and click Capture & Advise.")
        self.status_label.set_xalign(0)
        self.status_label.get_style_context().add_class("dim-label")
        main_box.append(self.status_label)

        # ── Paned: screenshot | advice ──
        paned = Gtk.Paned(orientation=Gtk.Orientation.HORIZONTAL, wide_handle=True)
        paned.set_position(450)
        paned.set_vexpand(True)
        main_box.append(paned)

        # Left: screenshot preview
        left_frame = Gtk.Frame()
        paned.set_start_child(left_frame)
        left_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        left_frame.set_child(left_box)

        left_box.append(Gtk.Label(label="Screenshot", margin_top=4))
        self.screenshot_view = Gtk.Picture()
        self.screenshot_view.set_content_fit(Gtk.ContentFit.CONTAIN)
        self.screenshot_view.set_vexpand(True)
        self.screenshot_view.set_hexpand(True)
        left_box.append(self.screenshot_view)

        # Right: advice display
        right_frame = Gtk.Frame()
        paned.set_end_child(right_frame)
        right_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        right_frame.set_child(right_box)

        right_box.append(Gtk.Label(label="Advice", margin_top=4))
        advice_scroll = Gtk.ScrolledWindow()
        advice_scroll.set_vexpand(True)
        advice_scroll.set_hexpand(True)
        right_box.append(advice_scroll)

        self.advice_view = Gtk.TextView()
        self.advice_view.set_editable(False)
        self.advice_view.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)
        self.advice_view.set_monospace(True)
        self.advice_view.set_top_margin(8)
        self.advice_view.set_bottom_margin(8)
        self.advice_view.set_left_margin(8)
        self.advice_view.set_right_margin(8)
        advice_scroll.set_child(self.advice_view)

        # ── History ──
        history_frame = Gtk.Frame()
        history_frame.set_vexpand(False)
        main_box.append(history_frame)
        history_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        history_frame.set_child(history_box)

        history_box.append(Gtk.Label(label="History", margin_top=4))
        history_scroll = Gtk.ScrolledWindow()
        history_scroll.set_min_content_height(80)
        history_scroll.set_max_content_height(150)
        history_box.append(history_scroll)

        self.history_list = Gtk.ListBox()
        self.history_list.set_selection_mode(Gtk.SelectionMode.NONE)
        history_scroll.set_child(self.history_list)

    # ── Window listing ──

    def _refresh_windows(self, *_args):
        """Scan for windows and populate the dropdown."""
        self.status_label.set_label("Scanning windows...")
        self.window_model.remove_all()
        self._windows = []

        try:
            if self._capture is None:
                self._capture = X11WindowCapture()
            windows = self._capture.list_windows()
        except Exception as e:
            self.status_label.set_label(f"Error listing windows: {e}")
            return

        # Filter out our own window and tiny utility windows
        my_title = "Card Game Advisor"
        for w in windows:
            name = w["name"] or "(unnamed)"
            if name == my_title:
                continue
            self._windows.append(w)
            label = f"{name}  ({w['width']}x{w['height']})"
            self.window_model.append(Gtk.StringObject.new(label))

        if self._windows:
            self.window_dropdown.set_selected(0)
            self.status_label.set_label(f"Found {len(self._windows)} window(s). Select one and click Capture & Advise.")
        else:
            self.status_label.set_label("No windows found. Click Refresh to try again.")

    def _on_refresh_windows(self, *_args):
        self._refresh_windows()

    # ── Prompt management ──

    def _load_prompts(self):
        """Load the prompt library (creating defaults on first run) into the UI."""
        self._prompts, active = ensure_prompts_file()
        self._rebuild_prompt_dropdown()
        idx = 0
        if active:
            for i, p in enumerate(self._prompts):
                if p["title"] == active:
                    idx = i
                    break
        self.prompt_dropdown.set_selected(idx)
        if self._prompts:
            self._active_prompt = self._prompts[idx]["title"]

    def _rebuild_prompt_dropdown(self):
        self.prompt_model.remove_all()
        for p in self._prompts:
            self.prompt_model.append(Gtk.StringObject.new(p["title"]))

    def _on_prompt_selected(self, *_args):
        idx = self.prompt_dropdown.get_selected()
        if idx == Gtk.INVALID_LIST_POSITION or idx >= len(self._prompts):
            return
        self._active_prompt = self._prompts[idx]["title"]
        buf = self.prompt_view.get_buffer()
        buf.set_text(self._prompts[idx]["text"])

    def _current_prompt_text(self) -> str:
        buf = self.prompt_view.get_buffer()
        return buf.get_text(buf.get_start_iter(), buf.get_end_iter(), True)

    def _on_new_prompt(self, *_args):
        title = self.prompt_title_entry.get_text().strip()
        if not title:
            n = len(self._prompts) + 1
            title = f"Prompt {n}"
            while any(p["title"] == title for p in self._prompts):
                n += 1
                title = f"Prompt {n}"
        if any(p["title"] == title for p in self._prompts):
            self.status_label.set_label(f"Prompt '{title}' already exists.")
            return
        # New prompts start as a copy of the current text — tweak from a working base
        self._prompts.append({"title": title, "text": self._current_prompt_text()})
        self._rebuild_prompt_dropdown()
        self.prompt_dropdown.set_selected(len(self._prompts) - 1)
        self._active_prompt = title
        save_prompts(self._prompts, title)
        self.prompt_title_entry.set_text("")
        self.status_label.set_label(f"Saved new prompt '{title}'.")

    def _on_save_prompt(self, *_args):
        idx = self.prompt_dropdown.get_selected()
        if idx == Gtk.INVALID_LIST_POSITION or idx >= len(self._prompts):
            self.status_label.set_label("Select or create a prompt first.")
            return
        self._prompts[idx]["text"] = self._current_prompt_text()
        self._active_prompt = self._prompts[idx]["title"]
        save_prompts(self._prompts, self._active_prompt)
        self.status_label.set_label(f"Saved prompt '{self._active_prompt}'.")

    # ── Nudge (aim correction) ──

    def _current_nudge(self) -> tuple[int, int]:
        return (self.nudge_x_spin.get_value_as_int(),
                self.nudge_y_spin.get_value_as_int())

    def _current_reasoning(self) -> str:
        """Return 'none' or 'max' based on the reasoning dropdown selection."""
        idx = self.reasoning_dropdown.get_selected()
        return "max" if idx == 1 else "none"

    def _detect_game(self) -> str:
        """Detect the game from the active prompt title.

        Returns 'chess' if the prompt title contains 'chess', 'solitaire'
        if it contains 'solitaire' or 'klondike', otherwise 'chess' (default).
        """
        title = (self._active_prompt or "").lower()
        if "solitaire" in title or "klondike" in title:
            return "solitaire"
        return "chess"

    def _on_nudge_reset(self, *_args):
        self.nudge_x_spin.set_value(0)
        self.nudge_y_spin.set_value(0)

    # ── Capture & Advise / Dry Run / Execute ──

    def _on_capture(self, *_args):
        self._start_capture("advise")

    def _on_dry_run(self, *_args):
        self._start_capture("dry_run")

    def _on_execute(self, *_args):
        self._start_capture("execute")

    def _start_capture(self, mode: str):
        if self._busy:
            return

        idx = self.window_dropdown.get_selected()
        if idx == Gtk.INVALID_LIST_POSITION or idx >= len(self._windows):
            self.status_label.set_label("Please select a window first.")
            return

        prompt_text = self._current_prompt_text()
        if not prompt_text.strip():
            self.status_label.set_label("Prompt is empty — write a prompt first.")
            return

        win = self._windows[idx]
        self._busy = True
        for btn in (self.capture_btn, self.dry_run_btn, self.execute_btn,
                    self.reexecute_btn, self.play_btn):
            btn.set_sensitive(False)
        mode_label = {"advise": "advice", "dry_run": "dry run", "execute": "move"}[mode]
        self.status_label.set_label(f"Capturing '{win['name']}' ({mode_label})...")

        # Run capture + API call in a background thread
        thread = threading.Thread(
            target=self._worker,
            args=(win, prompt_text, mode, self._current_nudge(),
                  self._current_reasoning()),
            daemon=True,
        )
        thread.start()

    # ── Re-execute last move ──

    def _on_reexecute(self, *_args):
        if self._busy:
            return
        if not self._last_move:
            self.status_label.set_label("No move received yet — capture one first.")
            return
        self._busy = True
        for btn in (self.capture_btn, self.dry_run_btn, self.execute_btn,
                    self.reexecute_btn, self.play_btn):
            btn.set_sensitive(False)
        self.status_label.set_label("Re-executing last move...")
        thread = threading.Thread(
            target=self._reexecute_worker, args=(self._current_nudge(),),
            daemon=True,
        )
        thread.start()

    # ── Play / Stop (auto-play loop) ──

    def _on_play(self, *_args):
        if self._busy or self._playing:
            return

        idx = self.window_dropdown.get_selected()
        if idx == Gtk.INVALID_LIST_POSITION or idx >= len(self._windows):
            self.status_label.set_label("Please select a window first.")
            return

        prompt_text = self._current_prompt_text()
        if not prompt_text.strip():
            self.status_label.set_label("Prompt is empty — write a prompt first.")
            return

        win = self._windows[idx]
        self._playing = True
        self._stop_requested = False
        self._busy = True
        for btn in (self.capture_btn, self.dry_run_btn, self.execute_btn,
                    self.reexecute_btn, self.play_btn):
            btn.set_sensitive(False)
        self.stop_btn.set_sensitive(True)
        self.status_label.set_label(f"Auto-playing '{win['name']}' — press Stop to halt.")

        thread = threading.Thread(
            target=self._play_loop_worker,
            args=(win, prompt_text, self._current_nudge(),
                  self._current_reasoning()),
            daemon=True,
        )
        thread.start()

    def _on_stop(self, *_args):
        if not self._playing:
            return
        self._stop_requested = True
        self.status_label.set_label("Stopping after the current move...")
        self.stop_btn.set_sensitive(False)

    def _play_loop_worker(self, win: dict, prompt_text: str,
                          nudge: tuple[int, int] = (0, 0),
                          reasoning: str = "none"):
        """Auto-play loop: capture → AI → detect → execute → wait 5s → repeat.

        Runs until _stop_requested is set or an error occurs.
        """
        move_count = 0
        try:
            while not self._stop_requested:
                move_count += 1
                GLib.idle_add(
                    self.status_label.set_label,
                    f"Move #{move_count} — capturing '{win['name']}'...",
                )
                try:
                    self._worker(win, prompt_text, "execute", nudge, reasoning)
                except Exception as e:
                    GLib.idle_add(self._show_error, f"Move #{move_count} error: {e}")
                    break

                if self._stop_requested:
                    break

                # Wait 5 seconds, checking for stop every 0.5s
                GLib.idle_add(
                    self.status_label.set_label,
                    f"Move #{move_count} done. Next move in 5s... (press Stop to halt)",
                )
                for _ in range(10):
                    if self._stop_requested:
                        break
                    time.sleep(0.5)

            GLib.idle_add(
                self.status_label.set_label,
                f"Auto-play stopped after {move_count} move(s).",
            )
        finally:
            self._playing = False
            self._stop_requested = False
            GLib.idle_add(self._set_idle)

    def _reexecute_worker(self, nudge: tuple[int, int] = (0, 0)):
        """Re-run the last received move — no new screenshot, no API call."""
        move = self._last_move
        win = move["win"]
        try:
            if self._capture is None:
                self._capture = X11WindowCapture()
            # Fresh window position — the window may have moved since capture
            off_x, off_y = win.get("x", 0), win.get("y", 0)
            geom = self._capture._get_window_geometry(win["id"])
            if geom:
                off_x, off_y = geom["x"], geom["y"]

            # Stored actions are window-relative — re-translate to screen coords
            actions = offset_actions(move["actions"], off_x + nudge[0], off_y + nudge[1])
            GLib.idle_add(
                self.status_label.set_label,
                "Re-executing in 1.5s — keep your mouse still...",
            )
            time.sleep(1.5)
            # Capture the current board as the "before" image for verification
            pre_path = self._capture_window_image(win)
            exec_log, post_path = self._execute_and_verify(
                win, geom or win, pre_path, actions, move["actions"],
                move["advice"], off_x + nudge[0], off_y + nudge[1],
                move.get("game", "chess"),
            )
            exec_log.insert(0, f"Window '{win['name']}' at screen ({off_x},{off_y})")
            # Show the post-move screenshot with the overlay kept on it
            try:
                post_overlay = os.path.join(
                    tempfile.gettempdir(),
                    f"card_advisor_gui_post_overlay_{int(time.time())}.png",
                )
                draw_move_overlay(post_path, post_overlay, move["actions"], None)
                GLib.idle_add(self._show_screenshot, post_overlay)
            except Exception:
                GLib.idle_add(self._show_screenshot, post_path)
            GLib.idle_add(self._show_reexecuted, move, actions, exec_log)
        except Exception as e:
            GLib.idle_add(self._show_error, str(e))
        finally:
            GLib.idle_add(self._set_idle)

    def _set_reexecute_tooltip(self, text: str):
        self.reexecute_btn.set_tooltip_text(f"Re-execute: {text}")
        return GLib.SOURCE_REMOVE

    def _show_reexecuted(self, move: dict, actions: list, exec_log: list[str]):
        """Show the result of re-executing the last move."""
        text = "── Re-executed last move ──\n" + "\n".join(
            a.describe() for a in actions
        )
        text += "\n── Execution ──\n" + "\n".join(exec_log)
        buf = self.advice_view.get_buffer()
        buf.set_text(text)

        timestamp = time.strftime("%H:%M:%S")
        last_line = move["advice"].strip().split("\n")[-1].strip()
        self._add_history_row(f"<b>{timestamp}</b> — Re-executed", last_line)
        self.status_label.set_label("Last move re-executed.")
        return GLib.SOURCE_REMOVE

    def _worker(self, win: dict, prompt_text: str, mode: str,
                nudge: tuple[int, int] = (0, 0),
                reasoning: str = "none"):
        """
        Background thread: two-step auto-play flow.
        Step 1 — capture screenshot → AI → best move text (e.g. "Nf3").
        Step 2 — move_detector → source/destination pixel coordinates →
                 drag via X11 input.
        mode: 'advise' (advice only), 'dry_run' (plan, no clicks),
              'execute' (perform the move).
        nudge: (dx, dy) screen-pixel correction applied to every move.
        """
        try:
            # 1. Capture screenshot
            tmp_path = os.path.join(
                tempfile.gettempdir(), f"card_advisor_gui_{int(time.time())}.png"
            )
            GLib.idle_add(self.status_label.set_label, f"Capturing '{win['name']}'...")
            self._capture.capture_window(win["id"], tmp_path)
            # Re-read the window position NOW — it may have moved since the
            # window list was scanned, and image coordinates are window-relative
            geom = self._capture._get_window_geometry(win["id"])
            if geom:
                off_x, off_y = geom["x"], geom["y"]
            else:
                off_x, off_y = win.get("x", 0), win.get("y", 0)
            GLib.idle_add(self.status_label.set_label, "Screenshot captured. Asking Mistral...")

            # 2. Show screenshot preview
            GLib.idle_add(self._show_screenshot, tmp_path)

            # 3. Step 1 — Ask Mistral for the best move + extract move text
            game = self._detect_game()
            advice, move_text = ask_best_move_parsed(
                self.api_key, tmp_path,
                game=game,
                prompt=prompt_text,
                reasoning=reasoning,
            )

            # 4. Detect pixel coordinates for the move
            GLib.idle_add(
                self.status_label.set_label,
                f"Move: {move_text!r} — detecting on board...",
            )

            actions: list[Action] = []
            detection = None
            detector_debug = ""
            if move_text and mode in ("dry_run", "execute"):
                det = detect_move(self.api_key, tmp_path, move_text, game=game)
                detection = det.describe()

                # Build a debug view of what was sent to move_detector
                from mistral_client import image_size as _img_size
                iw, ih = _img_size(tmp_path)
                sys_prompt = build_system_prompt(game)
                user_prompt = build_user_prompt(move_text, game, iw, ih)
                detector_debug = (
                    f"── move_detector request ──\n"
                    f"image: {tmp_path} ({iw}x{ih})\n"
                    f"move: {move_text!r}  game: {game}\n"
                    f"\n── system prompt ──\n{sys_prompt}\n"
                    f"\n── user prompt ──\n{user_prompt}\n"
                    f"\n── raw response ──\n{det.raw}\n"
                    f"\n── parsed ──\n{det.describe()}\n"
                )

                if det.source_point and det.dest_point:
                    sx, sy = det.source_point
                    dx, dy = det.dest_point
                    # For Solitaire: grab from the lower portion of the source
                    # card — the top may be covered by face-down cards in the
                    # column. Use 80% of box height from the top (near bottom).
                    if game == "solitaire" and det.source_box:
                        box_top = det.source_box[1] / 1000 * ih
                        box_h = (det.source_box[3] - det.source_box[1]) / 1000 * ih
                        sy = int(box_top + box_h * 0.8)
                    actions.append(Action(
                        type='drag', x1=sx, y1=sy, x2=dx, y2=dy,
                        raw=f"move_detector: {det.description or det.from_square}",
                    ))

            if actions:
                self._last_move = {
                    "win": win, "actions": actions, "advice": advice,
                    "move_text": move_text, "detection": detection, "game": game,
                }
                screen_actions = offset_actions(
                    actions, off_x + nudge[0], off_y + nudge[1]
                )
                GLib.idle_add(
                    self._set_reexecute_tooltip,
                    "; ".join(a.describe() for a in actions),
                )
                # Draw the planned move on the screenshot preview
                try:
                    overlay_path = os.path.join(
                        tempfile.gettempdir(),
                        f"card_advisor_gui_overlay_{int(time.time())}.png",
                    )
                    draw_move_overlay(tmp_path, overlay_path, actions, None)
                    GLib.idle_add(self._show_screenshot, overlay_path)
                except Exception:
                    pass
            else:
                screen_actions = []

            # 5. Execute (or dry-run)
            exec_log: list[str] = []
            if screen_actions and mode == "execute":
                GLib.idle_add(
                    self.status_label.set_label,
                    "Executing in 1.5s — keep your mouse still...",
                )
                time.sleep(1.5)
                exec_log, post_path = self._execute_and_verify(
                    win, geom or win, tmp_path, screen_actions, actions, advice,
                    off_x + nudge[0], off_y + nudge[1], game,
                )
                # Show the post-move screenshot with the overlay kept on it
                try:
                    post_overlay = os.path.join(
                        tempfile.gettempdir(),
                        f"card_advisor_gui_post_overlay_{int(time.time())}.png",
                    )
                    draw_move_overlay(post_path, post_overlay, actions, None)
                    GLib.idle_add(self._show_screenshot, post_overlay)
                except Exception:
                    GLib.idle_add(self._show_screenshot, post_path)
            elif screen_actions:
                exec_log = execute_actions(screen_actions, dry_run=True)
            if screen_actions:
                exec_log.insert(
                    0, f"Window '{win['name']}' at screen ({off_x},{off_y})"
                )

            # 6. Update UI with advice
            GLib.idle_add(self._show_advice, advice, tmp_path, screen_actions, exec_log, mode, detector_debug)

        except Exception as e:
            GLib.idle_add(self._show_error, str(e))
        finally:
            # Don't reset UI state when running inside the auto-play loop —
            # the loop worker manages button/idle state itself.
            if not self._playing:
                GLib.idle_add(self._set_idle)

    def _set_idle(self):
        self._busy = False
        self._playing = False
        self._stop_requested = False
        for btn in (self.capture_btn, self.dry_run_btn, self.execute_btn,
                    self.reexecute_btn, self.play_btn):
            btn.set_sensitive(True)
        self.stop_btn.set_sensitive(False)
        return GLib.SOURCE_REMOVE

    def _capture_window_image(self, win: dict) -> str:
        """Capture the game window to a temp file and return the path."""
        path = os.path.join(
            tempfile.gettempdir(), f"card_advisor_gui_post_{int(time.time())}.png"
        )
        self._capture.capture_window(win["id"], path)
        return path

    def _move_pointer_away(self, geom: dict):
        """
        Move the pointer out of the game window, so the post-move capture
        shows no hover highlight (keeps the before/after diff reliable).
        """
        try:
            with X11Input() as inp:
                sw, sh = inp.screen_width, inp.screen_height
                spots = (
                    (geom["x"] - 30, geom["y"] + 10),
                    (geom["x"] + geom["width"] + 30, geom["y"] + 10),
                    (geom["x"] + 10, geom["y"] - 30),
                    (geom["x"] + geom["width"] // 2,
                     geom["y"] + geom["height"] + 30),
                )
                for px, py in spots:
                    if 0 <= px < sw and 0 <= py < sh:
                        inp.move(px, py)
                        return
                inp.move(2, 2)
        except Exception:
            pass

    def _execute_and_verify(self, win: dict, geom: dict, pre_path: str,
                            screen_actions: list, actions: list, advice: str,
                            dx: int, dy: int, game: str = "chess") -> tuple[list[str], str]:
        """
        Execute the move, then verify the board actually changed around the
        move (before/after screenshot comparison). For chess, if the drag did
        not register, retry once with click-click. For Solitaire, always use
        drag — never fall back to clicks. Returns (exec_log, post_screenshot_path).
        """
        log: list[str] = []
        # Raise the game window so the injected input reaches it — the
        # advisor window may overlap it and would eat the clicks
        self._capture.raise_window(win["id"])
        log += execute_actions(screen_actions, dry_run=False)

        if not any(a.type == "drag" for a in actions):
            # nothing to verify for click-only moves — just report the result
            self._move_pointer_away(geom)
            time.sleep(0.5)
            return log, self._capture_window_image(win)

        boxes = extract_boxes(advice)
        GLib.idle_add(self.status_label.set_label, "Verifying the move landed...")
        self._move_pointer_away(geom)
        time.sleep(0.5)
        post_path = self._capture_window_image(win)

        if move_changed_board(pre_path, post_path, actions, boxes):
            log.append("Move verified.")
            return log, post_path

        # Solitaire: retry drag with adjusted grab point (60% of box height
        # instead of 80%) before giving up — the first grab point may have
        # been on a face-down card
        if game == "solitaire":
            log.append("Board unchanged — retrying drag with adjusted grab point...")
            retry_actions = []
            for a in actions:
                if a.type == "drag" and a.x1 is not None:
                    # Shift grab point up: from 80% to 60% of original offset
                    retry_actions.append(Action(
                        type='drag', x1=a.x1, y1=a.y1 - 15,
                        x2=a.x2, y2=a.y2, raw=a.raw + " (retry)",
                    ))
                else:
                    retry_actions.append(a)
            retry_screen = offset_actions(retry_actions, dx, dy)
            log += execute_actions(retry_screen, dry_run=False)
            self._move_pointer_away(geom)
            time.sleep(0.5)
            post_path = self._capture_window_image(win)
            if move_changed_board(pre_path, post_path, actions, boxes):
                log.append("Move verified (adjusted grab point worked).")
            else:
                log.append("Board still unchanged — drag did not register.")
            return log, post_path

        log.append("Board unchanged — retrying with click-click...")
        retry = offset_actions(drag_to_clicks(actions), dx, dy)
        log += execute_actions(retry, dry_run=False)
        self._move_pointer_away(geom)
        time.sleep(0.5)
        post_path = self._capture_window_image(win)

        if move_changed_board(pre_path, post_path, actions, boxes):
            log.append("Move verified (click-click worked).")
        else:
            log.append(
                "Board still unchanged — the move did not register in the game."
            )
        return log, post_path

    def _show_screenshot(self, path: str):
        """Display the screenshot in the preview pane."""
        try:
            pixbuf = GdkPixbuf.Pixbuf.new_from_file(path)
            texture = Gdk.Texture.new_for_pixbuf(pixbuf)
            self.screenshot_view.set_paintable(texture)
        except Exception as e:
            self.status_label.set_label(f"Error loading screenshot: {e}")
        return GLib.SOURCE_REMOVE

    def _add_history_row(self, header_markup: str, preview_text: str):
        """Prepend a row to the history list."""
        row = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        row.set_margin_top(4)
        row.set_margin_bottom(4)
        row.set_margin_start(8)
        row.set_margin_end(8)

        header = Gtk.Label()
        header.set_markup(header_markup)
        header.set_xalign(0)
        row.append(header)

        preview = Gtk.Label(label=preview_text[:120])
        preview.set_xalign(0)
        preview.get_style_context().add_class("dim-label")
        row.append(preview)

        self.history_list.prepend(row)

    def _show_advice(self, advice: str, image_path: str, actions=None,
                     exec_log=None, mode: str = "advise",
                     detector_debug: str = ""):
        """Display advice text (plus planned actions / execution log) and history."""
        text = advice
        if detector_debug:
            text += "\n\n" + detector_debug
        if actions:
            text += "\n\n── Planned actions ──\n" + "\n".join(
                a.describe() for a in actions
            )
            if mode == "dry_run":
                text += "\n(dry run — nothing was executed)"
            elif exec_log:
                text += "\n── Execution ──\n" + "\n".join(exec_log)
        buf = self.advice_view.get_buffer()
        buf.set_text(text)

        win_name = "?"
        idx = self.window_dropdown.get_selected()
        if idx != Gtk.INVALID_LIST_POSITION and idx < len(self._windows):
            win_name = self._windows[idx]["name"]

        timestamp = time.strftime("%H:%M:%S")
        # Show just the last line (the one-sentence move) in history
        last_line = advice.strip().split("\n")[-1].strip()
        self._add_history_row(f"<b>{timestamp}</b> — {win_name}", last_line)
        if mode == "execute":
            self.status_label.set_label("Move executed.")
        elif mode == "dry_run":
            self.status_label.set_label("Dry run complete — nothing was executed.")
        else:
            self.status_label.set_label("Done.")
        return GLib.SOURCE_REMOVE

    def _show_error(self, message: str):
        self.status_label.set_label(f"Error: {message}")
        buf = self.advice_view.get_buffer()
        buf.set_text(f"Error: {message}")
        return GLib.SOURCE_REMOVE


class AdvisorApp(Gtk.Application):
    def __init__(self, api_key: str):
        super().__init__(application_id="com.cardgame.advisor")
        self.api_key = api_key

    def do_activate(self):
        win = AdvisorWindow(self, self.api_key)
        win.present()


def main():
    api_key = os.environ.get("MISTRAL_API_KEY", "")
    if not api_key:
        print("Error: Set MISTRAL_API_KEY environment variable.")
        print("  export MISTRAL_API_KEY='your-key-here'")
        sys.exit(1)

    app = AdvisorApp(api_key)
    sys.exit(app.run(sys.argv))


if __name__ == "__main__":
    main()
