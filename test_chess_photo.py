#!/usr/bin/env python3
"""
Live Mistral API test — the real chess photo.

Sends chess_start.png (Stockfish, starting position, White to move) to the
Mistral API with the chess prompt and the actions output format, prints the
raw reply, parses the actions, and draws the move (source/destination boxes
+ drag arrow) over the screenshot.

Run:   python3 test_chess_photo.py            (with MISTRAL_API_KEY exported)
       MISTRAL_API_KEY=sk-... python3 test_chess_photo.py
Other photo:  CHESS_PHOTO=/path/to/photo.png python3 test_chess_photo.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PIL import Image

from move_parser import extract_boxes, extract_centers, parse_move_actions
from mistral_client import CHESS_PROMPT, ask_best_move, image_size
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
print('Sending to the Mistral API (chess prompt, actions format)...')
print()
response = ask_best_move(api_key, PHOTO, want_actions=True, prompt=CHESS_PROMPT)

print('--- raw API response ---')
print(response)
print('--- end of raw response ---')
print()

img_w, img_h = image_size(PHOTO)
actions = parse_move_actions(response, img_w, img_h)
assert actions, 'the API response parsed to no actions'
for a in actions:
    print('parsed action:', a.describe())

centers = extract_centers(response)
if centers:
    print('centers:', centers)
boxes = extract_boxes(response)
if boxes:
    print('boxes:', boxes)

# draw the move (source/destination boxes + drag arrow) over the screenshot
out = draw_move_overlay(
    PHOTO, os.path.join(tempfile.gettempdir(), 'chess_photo_overlay_live.png'),
    actions, boxes)
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
