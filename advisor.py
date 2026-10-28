#!/usr/bin/env python3
"""
Card Game Advisor — screenshot + AI advice.

Usage:
    ./play.sh                          # Screenshot + best move advice
    ./play.sh --window "Solitaire"     # Capture only this window
    ./play.sh --game "Klondike"        # Tell the AI which game
    ./play.sh --image <path>           # Use existing image
    ./play.sh --save                   # Save the screenshot
"""

import argparse
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from screenshot import take_screenshot, take_window_screenshot
from mistral_client import ask_best_move


def main():
    parser = argparse.ArgumentParser(description='Screenshot + AI advice')
    parser.add_argument('--window', '-w', default=None,
                        help='Capture only this window (by title, e.g. "Solitaire")')
    parser.add_argument('--game', '-g', default=None,
                        help='Tell the AI which game you are playing')
    parser.add_argument('--image', '-i', default=None,
                        help='Use an existing image instead of taking a screenshot')
    parser.add_argument('--api-key', '-k', default=os.environ.get('MISTRAL_API_KEY', ''),
                        help='Mistral API key (or set MISTRAL_API_KEY)')
    parser.add_argument('--save', '-s', action='store_true',
                        help='Save the screenshot')
    parser.add_argument('--output', '-o', default='/tmp/card_game_screenshot.png',
                        help='Where to save (with --save)')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Only print the advice')

    args = parser.parse_args()

    if not args.api_key:
        print("Error: No Mistral API key. Set MISTRAL_API_KEY.")
        sys.exit(1)

    # Get the image
    if args.image:
        image_path = args.image
        if not os.path.exists(image_path):
            print(f"Error: Image not found: {image_path}")
            sys.exit(1)
        if not args.quiet:
            print(f"Using image: {image_path}")
    else:
        if not args.quiet:
            target = f"window '{args.window}'" if args.window else "full screen"
            print(f"Switch to your game — capturing {target} in 5 seconds...")
            for i in range(5, 0, -1):
                print(f"  {i}...")
                time.sleep(1)
        tmp_path = args.output if args.save else os.path.join(
            tempfile.gettempdir(), f'card_advisor_{int(time.time())}.png'
        )
        try:
            if args.window:
                image_path = take_window_screenshot(args.window, tmp_path)
            else:
                image_path = take_screenshot(tmp_path)
        except Exception as e:
            print(f"Error taking screenshot: {e}")
            sys.exit(1)
        if not args.quiet:
            print(f"Screenshot saved: {image_path}")

    if not args.quiet:
        print("\nAsking Mistral...\n")

    try:
        advice = ask_best_move(args.api_key, image_path, game_hint=args.game)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(advice)

    # Cleanup
    if not args.image and not args.save:
        try:
            os.unlink(image_path)
        except OSError:
            pass


if __name__ == '__main__':
    main()
