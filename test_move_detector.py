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
    image_b64, mime = md._encode_image(PHOTO)
    msgs = [md._msg_system('system-prompt'),
            md._msg_user_image('detect e4', image_b64, mime)]
    p = md._build_payload(msgs, md.MODEL, 0.0)
    check('top_p is 1', p.get('top_p') == 1, str(p.get('top_p')))
    check('temperature 0 (greedy)', p.get('temperature') == 0.0,
          str(p.get('temperature')))
    check('reasoning_effort none (default)', p.get('reasoning_effort') == 'none',
          str(p.get('reasoning_effort')))
    check('model set', p.get('model') == md.MODEL, str(p.get('model')))
    check('messages passed through', p['messages'] is msgs)
    check('system+user messages',
          len(p['messages']) == 2 and p['messages'][0]['role'] == 'system'
          and p['messages'][1]['role'] == 'user',
          str([m['role'] for m in p['messages']]))
    content = p['messages'][1]['content']
    types = [c['type'] for c in content]
    check('user content text+image_url', types == ['text', 'image_url'],
          str(types))
    check('image is a data uri',
          content[1]['image_url']['url'].startswith('data:image/'),
          content[1]['image_url']['url'][:30])
    check('text prompt embedded', 'detect e4' in content[0]['text'])
    check('system prompt embedded', p['messages'][0]['content'] == 'system-prompt')
    p2 = md._build_payload(msgs, md.MODEL, 0.0, reasoning_effort='high')
    check('reasoning_effort configurable', p2.get('reasoning_effort') == 'high',
          str(p2.get('reasoning_effort')))


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

    def fake_chat(api_key, messages, model, temperature=0.0,
                  reasoning_effort='none', timeout=600):
        captured['messages'] = messages
        captured['model'] = model
        captured['api_key'] = api_key
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
    # Boxes are overridden by the detected board grid (exact), not the model's.
    check('source_box from grid', res.source_box == (385, 665, 435, 740),
          str(res.source_box))
    check('dest_box from grid', res.dest_box == (385, 515, 435, 590),
          str(res.dest_box))
    check('source_point from grid', res.source_point == (536, 608),
          str(res.source_point))
    check('dest_point from grid', res.dest_point == (536, 478),
          str(res.dest_point))
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
    msgs = captured['messages']
    check('system prompt mentions box_2d', 'box_2d' in msgs[0]['content'])
    user_content = msgs[1]['content']
    user_text = user_content[0]['text']
    check('user prompt mentions move', '"e4"' in user_text, user_text)
    check('user prompt has image size', f'{w}x{h}' in user_text, user_text)
    check('user content has image_url',
          any(c.get('type') == 'image_url' for c in user_content))
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


def test_parse_san():
    print('[offline] _parse_san')
    check('Nf3 piece N', md._parse_san('Nf3')['piece'] == 'N')
    check('Nf3 to f3', md._parse_san('Nf3')['to'] == 'f3')
    check('Nc3 piece N', md._parse_san('Nc3')['piece'] == 'N')
    check('Nc3 to c3', md._parse_san('Nc3')['to'] == 'c3')
    check('e4 piece P', md._parse_san('e4')['piece'] == 'P')
    check('e4 to e4', md._parse_san('e4')['to'] == 'e4')
    check('exd5 capture', md._parse_san('exd5')['capture'] is True)
    check('exd5 to d5', md._parse_san('exd5')['to'] == 'd5')
    check('O-O castle', md._parse_san('O-O')['castle'] is True)
    check('O-O piece K', md._parse_san('O-O')['piece'] == 'K')
    check('e8=Q promo', md._parse_san('e8=Q')['promo'] == 'Q')
    check('Qh4 piece Q', md._parse_san('Qh4')['piece'] == 'Q')


def test_geometry_ok():
    print('[offline] _geometry_ok (move shape per piece)')
    check('N g1->f3 ok', md._geometry_ok('N', 'g1', 'f3') is True)
    check('N b1->c3 ok', md._geometry_ok('N', 'b1', 'c3') is True)
    check('N g1->c3 ILLEGAL', md._geometry_ok('N', 'g1', 'c3') is False)
    check('N g1->g1 ILLEGAL', md._geometry_ok('N', 'g1', 'g1') is False)
    check('B f1->c4 ok', md._geometry_ok('B', 'f1', 'c4') is True)
    check('B f1->f4 ILLEGAL', md._geometry_ok('B', 'f1', 'f4') is False)
    check('R a1->a8 ok', md._geometry_ok('R', 'a1', 'a8') is True)
    check('R a1->b8 ILLEGAL', md._geometry_ok('R', 'a1', 'b8') is False)
    check('Q d1->d8 ok', md._geometry_ok('Q', 'd1', 'd8') is True)
    check('Q d1->d4 ok (straight)', md._geometry_ok('Q', 'd1', 'd4') is True)
    check('Q d1->g4 ok (diagonal)', md._geometry_ok('Q', 'd1', 'g4') is True)
    check('Q d1->e3 ILLEGAL (knight shape)',
          md._geometry_ok('Q', 'd1', 'e3') is False)
    check('K e1->e2 ok', md._geometry_ok('K', 'e1', 'e2') is True)
    check('K e1->e3 ILLEGAL', md._geometry_ok('K', 'e1', 'e3') is False)
    check('Pawn needs board -> None', md._geometry_ok('P', 'e2', 'e4') is None)


def test_chess_corrective_prompt():
    print('[offline] _chess_corrective_prompt')
    w, h = md._image_size(PHOTO)
    wrong = md.parse_detection(
        '{"box_2d": [487,740,536,813], "label":"source"} '
        '{"box_2d": [386,667,435,740], "label":"destination"} from: g1 to: c3',
        'Nc3', 'chess', w, h)
    msg = md._chess_corrective_prompt('Nc3', wrong)
    check('retry for illegal knight move', msg is not None)
    check('mentions NOT a legal', 'NOT a legal' in msg, msg)
    check('mentions knight', 'knight' in msg, msg)
    right = md.parse_detection(
        '{"box_2d": [237,740,288,813], "label":"source"} '
        '{"box_2d": [289,519,338,582], "label":"destination"} from: b1 to: c3',
        'Nc3', 'chess', w, h)
    check('no retry for legal knight move',
          md._chess_corrective_prompt('Nc3', right) is None)
    pawn = md.parse_detection(REPLY_E4, 'e4', 'chess', w, h)
    check('no retry for pawn move',
          md._chess_corrective_prompt('e4', pawn) is None)


def test_detect_move_chess_retry_stub():
    print('[offline] detect_move chess self-correction retry (HTTP stubbed)')
    WRONG = ('{"box_2d": [487,740,536,813], "label":"source"} '
             '{"box_2d": [386,667,435,740], "label":"destination"} '
             'from: g1 to: c3')
    RIGHT = ('{"box_2d": [237,740,288,813], "label":"source"} '
             '{"box_2d": [289,519,338,582], "label":"destination"} '
             'from: b1 to: c3')
    calls = []

    def fake_chat(api_key, messages, model, temperature=0.0,
                  reasoning_effort='none', timeout=600):
        calls.append(messages)
        return WRONG if len(calls) == 1 else RIGHT

    orig = md._chat_completion
    md._chat_completion = fake_chat
    try:
        det = md.detect_move('k', PHOTO, 'Nc3')
    finally:
        md._chat_completion = orig
    check('retry happened (2 API calls)', len(calls) == 2, f'{len(calls)} calls')
    check('final from b1', det.from_square == 'b1', str(det.from_square))
    check('final to c3', det.to_square == 'c3', str(det.to_square))
    last_user = calls[-1][-1]
    check('follow-up is a user msg', last_user['role'] == 'user')
    check('corrective mentions illegal move',
          'NOT a legal' in last_user['content'], last_user['content'])
    check('assistant msg carries the wrong answer',
          calls[-1][-2]['role'] == 'assistant'
          and 'g1' in calls[-1][-2]['content'])


def test_detect_chess_board():
    print('[offline] _detect_chess_board (grid + square centers)')
    board = md._detect_chess_board(PHOTO)
    check('board detected', board is not None, str(board))
    if board:
        left, top, sw, sh = board
        check('board left ~243', abs(left - 243) <= 4, str(left))
        check('board top ~186', abs(top - 186) <= 6, str(top))
        check('sq_w ~65', abs(sw - 65) <= 3, str(sw))
        check('sq_h ~65', abs(sh - 65) <= 3, str(sh))
        centers = {
            'b1': (340, 674), 'c3': (406, 544), 'e2': (536, 608),
            'e4': (536, 478), 'f3': (601, 544), 'g1': (666, 674),
            'h3': (731, 544), 'a3': (276, 544),
        }
        for sq, exp in centers.items():
            c = md._px_box_center(md._square_px_box(board, sq))
            check(f'center {sq}',
                  abs(c[0] - exp[0]) <= 5 and abs(c[1] - exp[1]) <= 5,
                  f'{c} vs {exp}')
    check('board cached', md._detect_chess_board(PHOTO) is board)


def test_detect_move_box_override_stub():
    print('[offline] detect_move overrides model boxes with grid boxes (stubbed)')
    orig = md._chat_completion
    md._chat_completion = lambda *a, **k: REPLY_E4  # model boxes for e2/e4
    try:
        det = md.detect_move('k', PHOTO, 'e4')
    finally:
        md._chat_completion = orig
    check('from/to from model', det.from_to == ('e2', 'e4'), str(det.from_to))
    # boxes come from the detected grid, not the model's REPLY_E4 boxes
    check('source_box is grid box', det.source_box == (385, 665, 435, 740),
          str(det.source_box))
    check('dest_box is grid box', det.dest_box == (385, 515, 435, 590),
          str(det.dest_box))
    check('source_point is grid center', det.source_point == (536, 608),
          str(det.source_point))
    check('dest_point is grid center', det.dest_point == (536, 478),
          str(det.dest_point))
    check('model box not used',
          det.source_box != (531.0, 687.0, 594.0, 750.0), str(det.source_box))


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


def test_live_knight():
    print('[live] knight moves (source not in the notation — needs the board)')
    api_key = os.environ.get('MISTRAL_API_KEY', '')
    if not api_key:
        print('  SKIP  MISTRAL_API_KEY not set (offline tests still validate the code)')
        return
    # Start position, White to move. For "Nf3"/"Nc3" the notation gives only
    # the destination — the model must read the board to pick the right knight.
    cases = [
        ('Nf3', 'g1', 'f3'),
        ('Nc3', 'b1', 'c3'),
        ('Nh3', 'g1', 'h3'),
        ('Na3', 'b1', 'a3'),
    ]
    orig_chat = md._chat_completion
    for mv, exp_from, exp_to in cases:
        calls = []

        def counting_chat(*a, _calls=calls, **k):
            _calls.append(1)
            return orig_chat(*a, **k)

        md._chat_completion = counting_chat
        try:
            res = md.detect_move(api_key, PHOTO, mv)
        finally:
            md._chat_completion = orig_chat
        print(f'  {mv}: {res.describe()}  (api calls: {len(calls)})')
        print(f'    raw: {res.raw!r}')
        check(f'live {mv} from {exp_from}', res.from_square == exp_from,
              f'got {res.from_square!r} expected {exp_from!r}')
        check(f'live {mv} to {exp_to}', res.to_square == exp_to,
              f'got {res.to_square!r} expected {exp_to!r}')
        check(f'live {mv} boxes present',
              res.source_box is not None and res.dest_box is not None)


def test_live_nc3_diag():
    print('[live] Nc3 diagnostic (repeat, squares + boxes vs ground truth)')
    api_key = os.environ.get('MISTRAL_API_KEY', '')
    if not api_key:
        print('  SKIP  MISTRAL_API_KEY not set')
        return
    # Ground truth (from the detected board grid on chess_start_large.png):
    # b1 center (340, 674), c3 center (406, 544).
    GT = {'b1': (340, 674), 'c3': (406, 544)}
    for i in range(2):
        res = md.detect_move(api_key, PHOTO, 'Nc3')
        sp, dp = res.source_point, res.dest_point
        sp_ok = sp and abs(sp[0]-GT['b1'][0]) <= 10 and abs(sp[1]-GT['b1'][1]) <= 10
        dp_ok = dp and abs(dp[0]-GT['c3'][0]) <= 10 and abs(dp[1]-GT['c3'][1]) <= 10
        print(f'  run {i+1}: {res.describe()}')
        print(f'         from/to={res.from_to} src_pt={sp} dest_pt={dp} '
              f'src_box={res.source_box} dest_box={res.dest_box}')
        print(f'         raw={res.raw!r}')
        check(f'run{i+1} from b1', res.from_square == 'b1', str(res.from_square))
        check(f'run{i+1} to c3', res.to_square == 'c3', str(res.to_square))
        check(f'run{i+1} source box on b1', bool(sp_ok), f'src_pt={sp}')
        check(f'run{i+1} dest box on c3', bool(dp_ok), f'dest_pt={dp}')


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
    test_parse_san()
    test_geometry_ok()
    test_chess_corrective_prompt()
    test_detect_move_chess_retry_stub()
    test_detect_chess_board()
    test_detect_move_box_override_stub()
    test_live()
    test_live_knight()
    test_live_nc3_diag()
    print()
    print(f'PASSED {PASS}  FAILED {FAIL}')
    sys.exit(1 if FAIL else 0)


if __name__ == '__main__':
    main()
