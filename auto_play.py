#!/usr/bin/env python3
"""
Card Game Auto-Play — screenshot → AI → perform the best move.

Takes a screenshot of your card game, asks Mistral Large 4 for the best
move, parses the ACTION block from the response, and performs the move
via X11 mouse input. Use --dry-run to preview without clicking.

Usage:
    python3 auto_play.py --dry-run         # SAFE: show planned actions, no clicks
    python3 auto_play.py                  # Screenshot + AI plays one move
    python3 auto_play.py --game "Poker"   # Specify the game
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
from mistral_client import ask_best_move, image_size
from move_parser import execute_actions, offset_actions, parse_move_actions
from input_controller import X11Input


def get_screen_size() -> tuple[int, int]:
    """Get screen dimensions via X11."""
    with X11Input() as inp:
        return inp.screen_width, inp.screen_height


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
    game_hint: str = None,
    window_title: str = None,
    save_screenshot: str = None,
    verbose: bool = True,
    history: list[str] = None,
    move_number: int = 1,
    debug: bool = False,
    dry_run: bool = False,
) -> dict:
    """
    Take a screenshot, ask the AI for the best move, then perform it.

    The AI response is parsed for an ACTION block (pixel coordinates) and
    executed via X11 input injection, unless dry_run is set.

    history: list of previous moves for loop context.
    move_number: current move number (1-based) for history tracking.
    dry_run: show the planned actions without moving the mouse.
    """
    # 1. Screenshot (with 5s delay so user can switch to the game window)
    if verbose:
        target = f"window '{window_title}'" if window_title else "full screen"
        print(f"[1/3] Switch to your game — capturing {target} in 5 seconds...")
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

    # Debug: save a copy of the exact image sent to the API
    if debug:
        import shutil
        shutil.copy(image_path, '/tmp/api_image.png')
        print(f"  [DEBUG] Image saved to /tmp/api_image.png for comparison", file=sys.stderr)
    if verbose:
        print(f"      Saved: {image_path}")

    # 2. Screen size (for prompt context)
    screen_w, screen_h = get_screen_size()
    if verbose:
        print(f"      Screen: {screen_w}x{screen_h}")

    # 3. AI advice + executable actions
    if verbose:
        history_note = f" ({len(history)} previous moves)" if history else ""
        mode_note = " [DRY RUN]" if dry_run else ""
        print(f"[2/3] Asking Mistral for the best move{mode_note}{history_note}...")
    response = ask_best_move(
        api_key, image_path,
        game_hint=game_hint, history=history,
        want_actions=True,
    )

    # Print the advice
    if verbose:
        print(f"\n{'='*60}")
    print(response)
    if verbose:
        print(f"{'='*60}\n")

    # 4. Parse actions (DRAG from CENTERS pixel centers, boxes fallback) and execute
    img_w, img_h = image_size(image_path)
    actions = parse_move_actions(response, img_w, img_h)
    if actions and window_title:
        # Window captures are window-relative — translate to screen coordinates
        off_x, off_y = _window_offset(window_title)
        actions = offset_actions(actions, off_x, off_y)

    log: list[str] = []
    if actions:
        if not dry_run and window_title:
            # Raise the game window so the injected input reaches it —
            # another window (e.g. the terminal) may be overlapping it
            _raise_window(window_title)
        log = execute_actions(actions, dry_run=dry_run)
        if verbose:
            verb = "Dry run" if dry_run else "Executing"
            print(f"[3/3] {verb} {len(actions)} action(s):")
            for line in log:
                print(f"  {line}")
    elif verbose:
        print("[3/3] No actions parsed from AI response.")

    # History entry: the move sentence (last line of the response)
    last_line = response.strip().splitlines()[-1].strip() if response.strip() else ""
    history_entry = f"Move {move_number}: {last_line[:100]}"

    # Cleanup temp screenshot if not saving
    if not save_screenshot and os.path.exists(image_path):
        try:
            os.unlink(image_path)
        except OSError:
            pass

    return {
        'response': response,
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
        help='Capture only this window (by title substring, e.g. "Solitaire"). '
             'Full screen if not specified.'
    )
    parser.add_argument(
        '--game', '-g',
        default=None,
        help='Tell the AI which game you are playing'
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
                game_hint=args.game,
                window_title=args.window,
                save_screenshot=save_path,
                verbose=verbose,
                
                history=move_history if args.loop > 1 else None,
                move_number=move_num,
                debug=args.debug,
                dry_run=args.dry_run,
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
