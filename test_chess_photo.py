#!/usr/bin/env python3
"""
Live Mistral API test — the real chess photo.

Two-step flow: sends chess_start.png to Mistral for the best move (text),
then uses move_detector to resolve it into source/destination pixel
coordinates, and draws the move over the screenshot.

Run:   python3 test_chess_photo.py            (with MISTRAL_API_KEY exported)
       MISTRAL_API_KEY=sk-... python3 test_chess_photo.py
Other photo:  CHESS_PHOTO=/path/to/photo.png python3 test_chess_photo.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PIL import Image

from move_parser import Action
from move_detector import detect_move
from mistral_client import CHESS_PROMPT, ask_best_move_parsed
from overlay import MOVE_COLOR, draw_move_overlay

HERE = os.path.dirname(os.path.abspath(__file__))
PHOTO = os.environ.get('CHESS_PHOTO', os.path.join(HERE, 'chess_start_large.png'))


def has_color_near(px, cx, cy, color, r=4):
    return any(px[cx + dx, cy + dy] == color
               for dx in range(-r, r + 1) for dy in range(-r, r + 1))


api_key = os.environ.get('MISTRAL_API_KEY', '')
if not api_key:
    print('Error: Set MISTRAL_API_KEY first.')
    sys.exit(1)

print(f'Photo: {PHOTO}')

# Step 1: ask for the best move (text only)
print('Step 1: Asking Mistral for the best move...')
response, move_text = ask_best_move_parsed(api_key, PHOTO, game='chess', prompt=CHESS_PROMPT)
print('--- raw API response ---')
print(response)
print('--- end of raw response ---')
print()

print(f'Extracted move: {move_text!r}')
assert move_text, 'no move extracted from the response'

# Step 2: detect the move on the board for pixel coordinates
print('Step 2: Detecting move on the board...')
det = detect_move(api_key, PHOTO, move_text, game='chess')
print(det.describe())
assert det.source_point and det.dest_point, 'move_detector returned no pixel coordinates'

# Build a drag action and draw the overlay
sx, sy = det.source_point
dx, dy = det.dest_point
actions = [Action(type='drag', x1=sx, y1=sy, x2=dx, y2=dy,
                  raw=f'{det.from_square} -> {det.to_square}')]
for a in actions:
    print('action:', a.describe())

out = draw_move_overlay(
    PHOTO, os.path.join(tempfile.gettempdir(), 'chess_photo_overlay_live.png'),
    actions, None)
drag = next((a for a in actions
             if a.type == 'drag' and a.x1 is not None), None)
if drag is not None:
    px = Image.open(out).convert('RGB').load()
    assert has_color_near(px, (drag.x1 + drag.x2) // 2,
                          (drag.y1 + drag.y2) // 2, MOVE_COLOR), \
        'the drag arrow was not drawn on the overlay'
print('overlay with the move drawn on the photo ->', out)

print()
print('LIVE API TEST PASSED')
