#!/usr/bin/env python3
"""
Tests for move_detector.detect_move.

Two layers:
  * OFFLINE (always run, no API key, no network): unit tests for the parsing
    helpers (extract_results / extract_boxes / extract_squares) and an
    end-to-end test of detect_move with the HTTP layer stubbed out.
  * LIVE (only when MISTRAL_API_KEY is set): sends the real chess photo
    (chess_start_large.png, Stockfish starting position, White to move) and
    the move "e4" to the Mistral API and checks it resolves to e2 -> e4.

Run:   python3 test_move_detector.py
       MISTRAL_API_KEY=sk-... python3 test_move_detector.py   # also runs LIVE
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import move_detector as md

HERE = os.path.dirname(os.path.abspath(__file__))
PHOTO = os.environ.get('CHESS_PHOTO', os.path.join(HERE, 'chess_start_large.png'))

PASS = 0
FAIL = 0


def check(name, cond, detail=''):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f'  ok   {name}')
    else:
        FAIL += 1
        print(f'  FAIL {name}  {detail}')


def has_color_near(px, cx, cy, color, r=4):
    """True if any pixel within radius r of (cx, cy) equals color exactly."""
    return any(px[cx + dx, cy + dy] == color
               for dx in range(-r, r + 1) for dy in range(-r, r + 1))


# ── canned model replies ────────────────────────────────────────────────────

REPLY_E4 = """\
The move e4 is a pawn push. On the starting position the only pawn that can
reach e4 is the pawn on e2, advancing two squares.

BOXES:
{"box_2d": [531, 687, 594, 750], "label": "source"}
{"box_2d": [531, 500, 594, 563], "label": "destination"}
SQUARES:
from: e2
to: e4
"""

# Cookbook style: both dicts on a single line, no whitespace between them.
REPLY_ADJACENT = (
    'The knight on g1 can reach f3. '
    '{"box_2d": [760, 687, 823, 750], "label": "source"}'
    '{"box_2d": [666, 593, 729, 656], "label": "destination"} '
    'from: g1\nto: f3'
)

REPLY_PROSE_FALLBACK = 'I would move the pawn from e2 to e4. No boxes.'

REPLY_NONE = 'I cannot locate that move on the board. BOXES: NONE\nSQUARES: NONE'


# ── offline tests ───────────────────────────────────────────────────────────

def test_extract_results():
    print('[offline] extract_results')
    res = md.extract_results(REPLY_E4)
    check('finds 2 dicts', len(res) == 2, f'got {len(res)}')
    check('first has box_2d', 'box_2d' in res[0], str(res[0]))
    check('first label source', res[0].get('label') == 'source', str(res[0]))
    check('second label destination', res[1].get('label') == 'destination',
          str(res[1]))
    check('adjacent-line dicts parse',
          len(md.extract_results(REPLY_ADJACENT)) == 2,
          str(md.extract_results(REPLY_ADJACENT)))


def test_extract_boxes():
    print('[offline] extract_boxes')
    boxes = md.extract_boxes(REPLY_E4)
    check('returns dict', boxes is not None)
    check('source box', boxes['source'] == (531.0, 687.0, 594.0, 750.0),
          str(boxes))
    check('dest box', boxes['dest'] == (531.0, 500.0, 594.0, 563.0), str(boxes))
    boxes2 = md.extract_boxes(REPLY_ADJACENT)
    check('adjacent source', boxes2['source'] == (760.0, 687.0, 823.0, 750.0),
          str(boxes2))
    check('adjacent dest', boxes2['dest'] == (666.0, 593.0, 729.0, 656.0),
          str(boxes2))
    check('NONE -> None', md.extract_boxes(REPLY_NONE) is None)


def test_extract_squares():
    print('[offline] extract_squares')
    check('from/to lines', md.extract_squares(REPLY_E4) == ('e2', 'e4'),
          str(md.extract_squares(REPLY_E4)))
    check('adjacent from/to', md.extract_squares(REPLY_ADJACENT) == ('g1', 'f3'),
          str(md.extract_squares(REPLY_ADJACENT)))
    check('prose fallback', md.extract_squares(REPLY_PROSE_FALLBACK) == ('e2', 'e4'),
          str(md.extract_squares(REPLY_PROSE_FALLBACK)))
    check('NONE -> None', md.extract_squares(REPLY_NONE) is None)


def test_box_center_px():
    print('[offline] _box_center_px')
    # normalized box centered at (562.5, 718.5) on a 1000x1000 image.
    # round() uses banker's rounding (half-to-even): 562.5 -> 562, 718.5 -> 718.
    c = md._box_center_px((531, 687, 594, 750), 1000, 1000)
    check('center', c == (562, 718), str(c))
    # clamps to [0, 999]
    c2 = md._box_center_px((-50, -50, 1200, 1200), 1000, 1000)
    check('clamp', c2 == (500, 500), str(c2))


def test_build_payload():
    print('[offline] _build_payload (greedy sampling requires top_p=1)')
    p = md._build_payload(PHOTO, 'detect e4', 'system-prompt', md.MODEL, 0.0)
    check('top_p is 1', p.get('top_p') == 1, str(p.get('top_p')))
    check('temperature 0 (greedy)', p.get('temperature') == 0.0,
          str(p.get('temperature')))
    check('reasoning_effort none (default)', p.get('reasoning_effort') == 'none',
          str(p.get('reasoning_effort')))
    p2 = md._build_payload(PHOTO, 'x', 's', md.MODEL, 0.0, reasoning_effort='high')
    check('reasoning_effort configurable', p2.get('reasoning_effort') == 'high',
          str(p2.get('reasoning_effort')))
    check('model set', p.get('model') == md.MODEL, str(p.get('model')))
    msgs = p['messages']
    check('system+user messages',
          len(msgs) == 2 and msgs[0]['role'] == 'system'
          and msgs[1]['role'] == 'user',
          str([m['role'] for m in msgs]))
    content = msgs[1]['content']
    types = [c['type'] for c in content]
    check('user content text+image_url', types == ['text', 'image_url'],
          str(types))
    check('image is a data uri',
          content[1]['image_url']['url'].startswith('data:image/'),
          content[1]['image_url']['url'][:30])
    check('text prompt embedded', 'detect e4' in content[0]['text'])
    check('system prompt embedded', msgs[0]['content'] == 'system-prompt')


def test_build_user_prompt():
    print('[offline] build_user_prompt')
    p = md.build_user_prompt('e4', 'chess', 1306, 866)
    check('contains move', '"e4"' in p, p)
    check('contains game', 'chess' in p, p)
    check('contains image size', '1306x866' in p, p)
    p2 = md.build_user_prompt('Nf3')
    check('no size when omitted', 'pixels' not in p2, p2)
    check('default game chess', 'chess' in p2, p2)


def test_detect_move_end_to_end_stub():
    print('[offline] detect_move end-to-end (HTTP stubbed)')
    # Stub the network call so we test prompt -> "API" -> parse with no key.
    captured = {}

    def fake_chat(api_key, image_path, user_prompt, system_prompt, model,
                  temperature=0.0, timeout=600):
        captured['api_key'] = api_key
        captured['image_path'] = image_path
        captured['user_prompt'] = user_prompt
        captured['system_prompt'] = system_prompt
        captured['model'] = model
        return REPLY_E4

    orig = md._chat_completion
    md._chat_completion = fake_chat
    try:
        res = md.detect_move('test-key', PHOTO, 'e4')
    finally:
        md._chat_completion = orig

    check('from_square e2', res.from_square == 'e2', str(res.from_square))
    check('to_square e4', res.to_square == 'e4', str(res.to_square))
    check('from_to tuple', res.from_to == ('e2', 'e4'), str(res.from_to))
    check('source_box set', res.source_box == (531.0, 687.0, 594.0, 750.0),
          str(res.source_box))
    check('dest_box set', res.dest_box == (531.0, 500.0, 594.0, 563.0),
          str(res.dest_box))
    check('image_size read', res.image_size is not None, str(res.image_size))
    w, h = res.image_size
    check('source_point in bounds',
          0 <= res.source_point[0] <= w and 0 <= res.source_point[1] <= h,
          str(res.source_point))
    check('dest_point in bounds',
          0 <= res.dest_point[0] <= w and 0 <= res.dest_point[1] <= h,
          str(res.dest_point))
    # e2 is below e4 on the board -> source center has a LARGER y than dest.
    check('source below dest (y)', res.source_point[1] > res.dest_point[1],
          f'src={res.source_point} dest={res.dest_point}')
    check('raw preserved', res.raw == REPLY_E4)
    check('model passed through', captured['model'] == md.MODEL)
    check('system prompt mentions box_2d',
          'box_2d' in captured['system_prompt'])
    check('user prompt mentions move', '"e4"' in captured['user_prompt'])
    check('user prompt has image size', f'{w}x{h}' in captured['user_prompt'],
          captured['user_prompt'])
    check('describe()', 'e2 -> e4' in res.describe(), res.describe())


def test_detect_move_validation():
    print('[offline] detect_move validation')
    try:
        md.detect_move('', PHOTO, 'e4')
        check('empty key raises', False)
    except ValueError:
        check('empty key raises', True)
    try:
        md.detect_move('k', PHOTO, '   ')
        check('empty move raises', False)
    except ValueError:
        check('empty move raises', True)


def test_detect_move_from_to_stub():
    print('[offline] detect_move_from_to (HTTP stubbed)')
    orig = md._chat_completion
    md._chat_completion = lambda *a, **k: REPLY_E4
    try:
        ft = md.detect_move_from_to('k', PHOTO, 'e4')
    finally:
        md._chat_completion = orig
    check('returns (e2, e4)', ft == ('e2', 'e4'), str(ft))


def test_draw_detected_move():
    print('[offline] draw_detected_move')
    from PIL import Image
    w, h = md._image_size(PHOTO)
    det = md.parse_detection(REPLY_E4, 'e4', 'chess', w, h)
    out = '/tmp/test_move_detector_overlay.png'
    ret = md.draw_detected_move(PHOTO, det, out)
    check('returns out_path', ret == out, ret)
    img = Image.open(out).convert('RGB')
    check('same size as input', img.size == (w, h), str(img.size))
    px = img.load()
    sp, dp = det.source_point, det.dest_point
    mid = ((sp[0] + dp[0]) // 2, (sp[1] + dp[1]) // 2)
    check('arrow color at midpoint',
          has_color_near(px, mid[0], mid[1], md.MOVE_COLOR))
    check('arrow color at source dot',
          has_color_near(px, sp[0], sp[1], md.MOVE_COLOR))
    check('arrow color at dest dot',
          has_color_near(px, dp[0], dp[1], md.MOVE_COLOR))
    sb = md._box_to_px(det.source_box, w, h)
    check('source box outline (green)',
          has_color_near(px, (sb[0] + sb[2]) // 2, sb[1], md.SRC_COLOR))
    db = md._box_to_px(det.dest_box, w, h)
    check('dest box outline (orange)',
          has_color_near(px, (db[0] + db[2]) // 2, db[1], md.DST_COLOR))


def test_detect_and_draw_stub():
    print('[offline] detect_and_draw (HTTP stubbed)')
    from PIL import Image
    orig = md._chat_completion
    md._chat_completion = lambda *a, **k: REPLY_E4
    out = '/tmp/test_move_detector_and_draw.png'
    try:
        det, path = md.detect_and_draw('k', PHOTO, 'e4', out)
    finally:
        md._chat_completion = orig
    check('returns det e2->e4', det.from_to == ('e2', 'e4'), str(det.from_to))
    check('returns path', path == out, path)
    check('det.overlay_path set', det.overlay_path == out, str(det.overlay_path))
    img = Image.open(out).convert('RGB')
    px = img.load()
    sp, dp = det.source_point, det.dest_point
    mid = ((sp[0] + dp[0]) // 2, (sp[1] + dp[1]) // 2)
    check('annotated image has arrow',
          has_color_near(px, mid[0], mid[1], md.MOVE_COLOR))


# ── live test (key-gated) ───────────────────────────────────────────────────

def test_live():
    print('[live] detect_move + draw on the real chess photo')
    api_key = os.environ.get('MISTRAL_API_KEY', '')
    if not api_key:
        print('  SKIP  MISTRAL_API_KEY not set (offline tests still validate the code)')
        return
    from PIL import Image
    print(f'  photo: {PHOTO}')
    out = '/tmp/move_detector_live_overlay.png'
    print('  sending "e4" to the Mistral API...')
    res, path = md.detect_and_draw(api_key, PHOTO, 'e4', out)
    print('  ' + res.describe())
    print(f'  annotated photo -> {path}')
    print('  --- raw API response ---')
    print(res.raw)
    print('  --- end of raw response ---')
    check('live from_square e2', res.from_square == 'e2', str(res.from_square))
    check('live to_square e4', res.to_square == 'e4', str(res.to_square))
    check('live boxes present', res.source_box is not None and
          res.dest_box is not None)
    check('live source below dest',
          res.source_point and res.dest_point and
          res.source_point[1] > res.dest_point[1])
    # e2 and e4 share the e-file, so a correct detection has nearly equal x.
    check('live same file (x close)',
          res.source_point and res.dest_point and
          abs(res.source_point[0] - res.dest_point[0]) <= 0.10 * res.image_size[0],
          f'src={res.source_point} dest={res.dest_point}')
    # verify the annotated picture actually shows the move
    img = Image.open(path).convert('RGB')
    check('live overlay same size', img.size == tuple(res.image_size),
          str(img.size))
    px = img.load()
    sp, dp = res.source_point, res.dest_point
    if sp and dp and res.source_box and res.dest_box:
        mid = ((sp[0] + dp[0]) // 2, (sp[1] + dp[1]) // 2)
        check('live overlay arrow at midpoint',
              has_color_near(px, mid[0], mid[1], md.MOVE_COLOR))
        sb = md._box_to_px(res.source_box, *res.image_size)
        check('live overlay source box (green)',
              has_color_near(px, (sb[0] + sb[2]) // 2, sb[1], md.SRC_COLOR))
        db = md._box_to_px(res.dest_box, *res.image_size)
        check('live overlay dest box (orange)',
              has_color_near(px, (db[0] + db[2]) // 2, db[1], md.DST_COLOR))
    else:
        check('live overlay arrow at midpoint', False, 'no points to draw')


def main():
    test_extract_results()
    test_extract_boxes()
    test_extract_squares()
    test_box_center_px()
    test_build_payload()
    test_build_user_prompt()
    test_detect_move_end_to_end_stub()
    test_detect_move_validation()
    test_detect_move_from_to_stub()
    test_draw_detected_move()
    test_detect_and_draw_stub()
    test_live()
    print()
    print(f'PASSED {PASS}  FAILED {FAIL}')
    sys.exit(1 if FAIL else 0)


if __name__ == '__main__':
    main()
