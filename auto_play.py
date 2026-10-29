#!/usr/bin/env python3
"""
Auto-Play — screenshot → AI move → move_detector → perform.

Two-step flow:
  1. Screenshot → Mistral Large 4 → best move (text)
  2. move_detector → resolve the move into source/destination pixel
     coordinates → drag via X11 mouse input.

Use --dry-run to preview without clicking.

Usage:
    python3 auto_play.py --dry-run         # SAFE: show planned actions, no clicks
    python3 auto_play.py                  # Screenshot + AI plays one move
    python3 auto_play.py --prompt "My Game"  # Select a prompt by title
    python3 auto_play.py --loop 10        # Play 10 moves in a row (with history)
    python3 auto_play.py --delay 2        # Wait 2s between moves (loop mode)
"""

import argparse
import os
import sys
import time
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from screenshot import take_screenshot, take_window_screenshot, X11WindowCapture
from mistral_client import ask_best_move_parsed, image_size
from move_parser import Action, execute_actions
from move_detector import detect_move, build_system_prompt, build_user_prompt
from prompt_loader import get_prompt_by_title
from input_controller import X11Input


def _window_offset(window_title: str) -> tuple[int, int]:
    """Screen coordinates of a window's top-left corner (for window captures)."""
    try:
        with X11WindowCapture() as cap:
            win = cap.find_window(window_title)
    except Exception:
        return 0, 0
    if win:
        return win['x'], win['y']
    return 0, 0


def _raise_window(window_title: str):
    """Raise the game window so injected input reaches it."""
    with X11WindowCapture() as cap:
        win = cap.find_window(window_title)
        if win:
            cap.raise_window(win['id'])


def play_one_move(
    api_key: str,
    prompt_title: str = None,
    window_title: str = None,
    save_screenshot: str = None,
    verbose: bool = True,
    history: list[str] = None,
    move_number: int = 1,
    debug: bool = False,
    dry_run: bool = False,
    reasoning: str = "low",
) -> dict:
    """
    Two-step auto-play:
      Step 1 — screenshot → AI → best move text.
      Step 2 — move_detector → source/destination pixel coordinates →
               drag via X11 input injection.

    prompt_title: title of a prompt in prompts.json. If None, the active
                  prompt is used.
    history: list of previous moves for loop context.
    move_number: current move number (1-based) for history tracking.
    dry_run: show the planned actions without moving the mouse.
    """
    # Look up the prompt config from prompts.json
    prompt_cfg = get_prompt_by_title(prompt_title) if prompt_title else None
    if prompt_cfg is None:
        from prompt_loader import get_active_prompt
        prompt_cfg = get_active_prompt()
    if prompt_cfg is None:
        raise RuntimeError("No prompt found in prompts.json")
    prompt_text = prompt_cfg.get("text", "")
    game_key = prompt_cfg.get("game", "default")
    behavior = prompt_cfg.get("behavior", {})

    # ── Step 0: Screenshot (with 5s delay so user can switch windows) ──
    if verbose:
        target = f"window '{window_title}'" if window_title else "full screen"
        print(f"[1/4] Switch to your game — capturing {target} in 5 seconds...")
        for i in range(5, 0, -1):
            print(f"      {i}...")
            time.sleep(1)
    tmp_path = save_screenshot or os.path.join(
        tempfile.gettempdir(), f'auto_play_{int(time.time())}.png'
    )
    if window_title:
        image_path = take_window_screenshot(window_title, tmp_path)
    else:
        image_path = take_screenshot(tmp_path)

    if debug:
        import shutil
        shutil.copy(image_path, '/tmp/api_image.png')
        print(f"  [DEBUG] Image saved to /tmp/api_image.png for comparison", file=sys.stderr)
    if verbose:
        print(f"      Saved: {image_path}")

    # ── Step 1: Ask the AI for the best move + extract move text ──
    if verbose:
        history_note = f" ({len(history)} previous moves)" if history else ""
        mode_note = " [DRY RUN]" if dry_run else ""
        print(f"[2/4] Asking Mistral for the best move{mode_note}{history_note}...")
    response, move_text = ask_best_move_parsed(
        api_key, image_path,
        prompt=prompt_text, behavior=behavior, history=history,
        reasoning=reasoning,
    )

    if verbose:
        print(f"\n{'='*60}")
    print(response)
    if verbose:
        print(f"{'='*60}\n")
        print(f"[3/4] Extracted move: {move_text!r}")
    if not move_text:
        if verbose:
            print("[4/4] No move extracted from AI response.")
        return {
            'response': response,
            'move': '',
            'actions': [],
            'log': [],
            'screenshot': tmp_path if save_screenshot else None,
            'history_entry': f"Move {move_number}: (no move)",
        }

    # ── Step 2: Send the move to move_detector for pixel coordinates ──
    if verbose:
        print(f"[3/4] Detecting move '{move_text}' on the board...")
        iw, ih = image_size(image_path)
        print(f"      move_detector request:")
        print(f"        image: {image_path} ({iw}x{ih})")
        print(f"        move: {move_text!r}  game_key: {game_key}")
        print(f"        system prompt ({len(build_system_prompt(game_key))} chars):")
        for line in build_system_prompt(game_key).splitlines():
            print(f"          {line}")
        print(f"        user prompt:")
        for line in build_user_prompt(move_text, game_key, iw, ih).splitlines():
            print(f"          {line}")
    det = detect_move(api_key, image_path, move_text,
                      game_key=game_key, behavior=behavior)
    if verbose:
        print(f"      {det.describe()}")
        print(f"      raw response:")
        for line in det.raw.splitlines():
            print(f"        {line}")

    # ── Step 3: Build action and execute ──
    actions: list[Action] = []
    if det.source_point and det.dest_point:
        sx, sy = det.source_point
        dx, dy = det.dest_point
        # Adjust grab point: grab at behavior['grab_y'] fraction of the
        # source box height (e.g. 0.8 = near the bottom of the card).
        grab_y = behavior.get("grab_y")
        if grab_y is not None and det.source_box:
            from mistral_client import image_size as _img_size
            _iw, _ih = _img_size(image_path)
            if _ih:
                box_top = det.source_box[1] / 1000 * _ih
                box_h = (det.source_box[3] - det.source_box[1]) / 1000 * _ih
                sy = int(box_top + box_h * grab_y)
        # Window captures are window-relative — translate to screen coordinates
        if window_title:
            off_x, off_y = _window_offset(window_title)
            sx += off_x
            sy += off_y
            dx += off_x
            dy += off_y
        # Single-click action (e.g. reveal/flag a cell)
        if behavior.get("action") == "click":
            rc_word = behavior.get("right_click_word", "")
            is_right = rc_word and rc_word in move_text.lower()
            actions.append(Action(
                type='right_click' if is_right else 'click',
                x1=sx, y1=sy,
                raw=f"move_detector: {det.description or move_text} ({sx},{sy})",
            ))
        else:
            actions.append(Action(
                type='drag', x1=sx, y1=sy, x2=dx, y2=dy,
                raw=f"move_detector: {det.from_square} -> {det.to_square} "
                    f"({sx},{sy}) -> ({dx},{dy})",
            ))

    log: list[str] = []
    if actions:
        if not dry_run and window_title:
            _raise_window(window_title)
        log = execute_actions(actions, dry_run=dry_run)
        if verbose:
            verb = "Dry run" if dry_run else "Executing"
            print(f"[4/4] {verb} {len(actions)} action(s):")
            for line in log:
                print(f"  {line}")
    elif verbose:
        print("[4/4] No pixel coordinates from move_detector — cannot execute.")

    # History entry
    history_entry = f"Move {move_number}: {move_text}"

    # Cleanup temp screenshot if not saving
    if not save_screenshot and os.path.exists(image_path):
        try:
            os.unlink(image_path)
        except OSError:
            pass

    return {
        'response': response,
        'move': move_text,
        'detection': det.describe() if det else None,
        'actions': [a.describe() for a in actions],
        'log': log,
        'screenshot': tmp_path if save_screenshot else None,
        'history_entry': history_entry,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Screenshot → AI → auto-play the best move'
    )
    parser.add_argument(
        '--api-key', '-k',
        default=os.environ.get('MISTRAL_API_KEY', ''),
        help='Mistral API key (or set MISTRAL_API_KEY env var)'
    )
    parser.add_argument(
        '--window', '-w',
        default=None,
        help='Capture only this window (by title substring). '
             'Full screen if not specified.'
    )
    parser.add_argument(
        '--prompt', '-p',
        default=None,
        help='Select a prompt by title from prompts.json (default: active prompt)'
    )
    parser.add_argument(
        '--loop', '-l',
        type=int,
        default=1,
        help='Number of screenshots to take in a row (default: 1)'
    )
    parser.add_argument(
        '--delay', '-d',
        type=float,
        default=1.0,
        help='Seconds to wait between screenshots in loop mode (default: 1.0)'
    )
    parser.add_argument(
        '--save', '-s',
        default=None,
        help='Save screenshots to this path (use {n} for move number)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Save the exact image sent to the API to /tmp/api_image.png for comparison'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show the planned actions without executing them (SAFE)'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Minimal output'
    )
    parser.add_argument(
        '--reasoning', '-r',
        choices=['none', 'low', 'max'],
        default='low',
        help='AI reasoning level for move advice: none (fastest), low (default) '
             'or max (think) (default: low)'
    )

    args = parser.parse_args()

    if not args.api_key:
        print("Error: No Mistral API key. Set MISTRAL_API_KEY or use --api-key.")
        print("Get a key at: https://console.mistral.ai/api-keys/")
        sys.exit(1)

    verbose = not args.quiet

    move_history: list[str] = []

    for move_num in range(1, args.loop + 1):
        if args.loop > 1:
            if verbose:
                print(f"\n{'#'*60}")
                print(f"# MOVE {move_num}/{args.loop}")
                print(f"{'#'*60}\n")
            else:
                print(f"Move {move_num}/{args.loop}...")

        save_path = None
        if args.save:
            save_path = args.save.replace('{n}', str(move_num))

        try:
            result = play_one_move(
                api_key=args.api_key,
                prompt_title=args.prompt,
                window_title=args.window,
                save_screenshot=save_path,
                verbose=verbose,

                history=move_history if args.loop > 1 else None,
                move_number=move_num,
                debug=args.debug,
                dry_run=args.dry_run,
                reasoning=args.reasoning,
            )

            # Record this move in history for the next iteration
            if 'history_entry' in result:
                move_history.append(result['history_entry'])

        except KeyboardInterrupt:
            print("\nInterrupted by user.")
            break
        except Exception as e:
            print(f"\nError: {e}")
            if args.loop > 1:
                break
            sys.exit(1)

        if move_num < args.loop:
            if verbose:
                print(f"\nWaiting {args.delay}s before next advice...")
            time.sleep(args.delay)

    if verbose:
        print("\nDone.")


if __name__ == '__main__':
    main()
