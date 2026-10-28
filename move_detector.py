#!/usr/bin/env python3
"""
Move detector — uses the Mistral endpoint to detect a game move on a board
image and resolve it into a from/to (source/destination) pair.

Given a screenshot of a board game and a move written in natural / algebraic
notation (chess example: "e4", "Nf3", "exd5", "O-O"), this module asks
Mistral Large 4 (vision) to locate the two squares involved in the move and
returns them both as algebraic squares (from/to) and as normalized bounding
boxes (pixel centers included), following the object-detection recipe from the
Mistral cookbook:

    mistral/image_understanding/bbox_getstarted

The cookbook's recommendations implemented here:
  * a SYSTEM prompt stating the objective, the box_2d task and a response
    format with a concrete example;
  * free generation (no structured outputs) — the model replies in prose and
    ends with the standardized dictionaries;
  * results extracted afterwards with the cookbook regex
    r'\\{(?=[^{}]*\\b(?:box_2d|point_2d)\\b)(?=[^{}]*\\blabel\\b)[^{}]*\\}'
    + json.loads;
  * one type only — box_2d (not mixed with point_2d).

Output format (per the cookbook): {"box_2d": [xmin, ymin, xmax, ymax],
"label": string}, coordinates normalized 0-999 (0 = left/top edge of the
image, 999 = right/bottom edge).

This module is self-contained: it does NOT import mistral_client,
move_parser or overlay. It talks to the Mistral HTTP endpoint with the
standard library (urllib) only, so it has no third-party dependency. PIL is
used to read the image size and to draw the move overlay; it is optional for
detection (required only when drawing).

It can also SHOW the move on the picture: pass overlay_path=... to
detect_move (or use detect_and_draw / the CLI) and it draws the source box,
destination box and a drag arrow over the screenshot and saves the annotated
image.

Run as a script (detects the move and writes an annotated picture):
    python3 move_detector.py <image_path> <move> [api_key] [overlay_out.png]
    MISTRAL_API_KEY=sk-... python3 move_detector.py chess_start_large.png e4
"""

from __future__ import annotations

import base64
import bisect
import json
import math
import os
import re
import sys
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional

MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"
MODEL = "mistral-large-4"

# Overlay colours (same scheme as the rest of the project): green source box,
# orange destination box, yellow drag arrow.
SRC_COLOR = (0, 220, 0)
DST_COLOR = (255, 80, 0)
MOVE_COLOR = (255, 235, 0)

# Cookbook extraction regex: match a { ... } block that contains both a
# box_2d (or point_2d) key and a label key, with no nested braces.
_RESULT_DICT = re.compile(
    r'\{(?=[^{}]*\b(?:box_2d|point_2d)\b)(?=[^{}]*\blabel\b)[^{}]*\}'
)

# A chess square in algebraic notation: file a-h, rank 1-8.
_SQUARE = r'[a-hA-H][1-8]'


# ── System prompt (cookbook "Recommended Usage": objective + task + format) ──

def build_system_prompt(game: str = "chess") -> str:
    """System prompt: detection objective + box_2d task + strict format.

    Follows the cookbook's recommended system prompt (objective, the box_2d
    task, a response format with a concrete example, free generation) but is
    deliberately forceful: the model must END with the two standardized
    dictionaries and must not ramble about coordinate math.
    """
    return (
        "You are an object-detection vision model for a board-game-playing "
        f"agent. The game is {game}. "
        "Objective: given a screenshot and a move, locate the two squares "
        "involved in that move — the SOURCE square (the piece being moved) "
        "and the DESTINATION square (where it is moved to). "
        "Task — box_2d only: report the bounding box of each square as 4 "
        "coordinates [xmin, ymin, xmax, ymax], normalized 0-999 (0 = left/top "
        "edge of the image, 999 = right/bottom edge), in the WHOLE image. "
        "Bracket each square tightly. "
        "Do NOT reason step-by-step about coordinates and do NOT write a "
        "long analysis. You MUST end your reply with exactly two "
        "dictionaries (one per line) followed by two square lines, in this "
        "exact format: "
        '{"box_2d": [xmin, ymin, xmax, ymax], "label": "source"} '
        '{"box_2d": [xmin, ymin, xmax, ymax], "label": "destination"} '
        "from: <square> "
        "to: <square> "
        "Squares use algebraic notation (chess: file a-h + rank 1-8, e.g. "
        "e2). Example for the chess move \"e4\": "
        '{"box_2d": [531, 687, 594, 750], "label": "source"} '
        '{"box_2d": [531, 500, 594, 563], "label": "destination"} '
        "from: e2 "
        "to: e4 "
        "The move notation may give only the piece type and the destination "
        "(e.g. \"Nf3\" gives no source square, and several knights may exist). "
        "In that case READ THE BOARD: find every piece of that type, and "
        "determine which one can LEGALLY move to the destination square (a "
        "knight moves in an L: 2 squares one way + 1 square perpendicular; a "
        "pawn moves forward; a bishop diagonally; etc.). That piece's square "
        "is the SOURCE. Always check that source -> destination is a legal "
        "move for that piece before reporting — do NOT guess and do NOT "
        "default to a square. "
        "If the move cannot be located on the board, output 'BOXES: NONE' "
        "and 'SQUARES: NONE'."
    )


def build_user_prompt(move: str, game: str = "chess",
                      img_w: Optional[int] = None,
                      img_h: Optional[int] = None) -> str:
    """User prompt: the specific move to detect on the attached screenshot."""
    size = (f" The screenshot is {img_w}x{img_h} pixels."
            if img_w and img_h else "")
    return (
        f"The image shows a game of {game}.{size} "
        f'The player wants to play the move "{move}". Detect the source '
        f"square (the piece being moved) and the destination square (the "
        f"square it moves to). If the notation does not specify the source "
        f"square, use the board to find the piece that can legally make the "
        f"move. Report their bounding boxes (labels 'source' and "
        f"'destination') and the from/to squares."
    )


# ── Result type ─────────────────────────────────────────────────────────────

@dataclass
class DetectedMove:
    """A detected game move resolved into a from/to pair."""
    move: str                       # the input move text, e.g. "e4"
    game: str                       # e.g. "chess"
    from_square: Optional[str]      # algebraic source, e.g. "e2"
    to_square: Optional[str]        # algebraic destination, e.g. "e4"
    source_box: Optional[tuple]     # normalized (xmin, ymin, xmax, ymax) 0-999
    dest_box: Optional[tuple]       # normalized (xmin, ymin, xmax, ymax) 0-999
    source_point: Optional[tuple]   # pixel (x, y) center of the source box
    dest_point: Optional[tuple]     # pixel (x, y) center of the destination box
    image_size: Optional[tuple]     # (width, height) of the image in pixels
    raw: str                        # raw model reply
    overlay_path: Optional[str] = None  # annotated picture, if one was drawn

    @property
    def from_to(self) -> tuple:
        return (self.from_square, self.to_square)

    def describe(self) -> str:
        ft = f"{self.from_square} -> {self.to_square}" if self.from_square else "?"
        sp = f"{self.source_point}" if self.source_point else "?"
        dp = f"{self.dest_point}" if self.dest_point else "?"
        s = f"move {self.move!r}: {ft} | source px {sp} | dest px {dp}"
        if self.overlay_path:
            s += f" | photo {self.overlay_path}"
        return s


# ── Image helpers (stdlib + optional PIL) ───────────────────────────────────

_MIME = {'.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
         '.gif': 'image/gif', '.webp': 'image/webp', '.bmp': 'image/bmp'}


def _encode_image(image_path: str) -> tuple[str, str]:
    """Encode an image to base64. Returns (base64, mime_type)."""
    with open(image_path, 'rb') as f:
        data = base64.b64encode(f.read()).decode('utf-8')
    ext = os.path.splitext(image_path)[1].lower()
    return data, _MIME.get(ext, 'image/png')


def _image_size(image_path: str) -> tuple[Optional[int], Optional[int]]:
    """Return (width, height) of the image, or (None, None) if unreadable."""
    try:
        from PIL import Image
        with Image.open(image_path) as img:
            return img.width, img.height
    except Exception:
        return None, None


# ── Mistral endpoint call (stdlib urllib) ───────────────────────────────────

def _extract_text(message: dict) -> str:
    """Extract text from an API response message (string or list of parts)."""
    content = message.get('content', '')
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                if part.get('type') == 'text':
                    parts.append(part.get('text', ''))
                elif 'text' in part:
                    parts.append(part['text'])
            elif isinstance(part, str):
                parts.append(part)
        return '\n'.join(parts)
    return str(content)


# ── Chat message helpers ────────────────────────────────────────────────────

def _msg_system(text: str) -> dict:
    return {"role": "system", "content": text}


def _msg_user_text(text: str) -> dict:
    return {"role": "user", "content": text}


def _msg_assistant(text: str) -> dict:
    return {"role": "assistant", "content": text}


def _msg_user_image(text: str, image_b64: str, mime_type: str) -> dict:
    return {"role": "user", "content": [
        {"type": "text", "text": text},
        {"type": "image_url",
         "image_url": {"url": f"data:{mime_type};base64,{image_b64}"}},
    ]}


def _build_payload(messages: list, model: str, temperature: float,
                   reasoning_effort: str = "none") -> dict:
    """Build the chat-completions request payload.

    temperature defaults to 0 (greedy sampling); the Mistral API rejects
    greedy sampling unless top_p is 1, so top_p is always set to 1.
    reasoning_effort defaults to "none" (no chain-of-thought), matching the
    deterministic, token-light behaviour used elsewhere in this project.
    """
    return {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": 1,
        "reasoning_effort": reasoning_effort,
        "max_tokens": 4096,
    }


def _chat_completion(api_key: str, messages: list, model: str,
                     temperature: float = 0.0,
                     reasoning_effort: str = "none", timeout: int = 600) -> str:
    """Send a messages list to the Mistral endpoint. Returns the reply text.

    The image (if any) must already be embedded in a user message via
    _msg_user_image. Supports multi-turn (history) for self-correction retries.
    """
    payload = _build_payload(messages, model, temperature, reasoning_effort)

    req = urllib.request.Request(
        MISTRAL_API_URL,
        data=json.dumps(payload).encode('utf-8'),
        headers={
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}',
        },
        method='POST',
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        raise RuntimeError(
            f"API error ({e.code}): {e.read().decode('utf-8')[:500]}")

    choices = result.get('choices', [])
    if not choices:
        raise RuntimeError(f"No choices: {json.dumps(result)[:500]}")

    text = _extract_text(choices[0]['message'])
    if not text.strip():
        raise RuntimeError(
            f"Empty response (finish_reason="
            f"{choices[0].get('finish_reason', '?')})")
    return text


# ── Parsing (cookbook extract_results + from/to squares) ────────────────────

def extract_results(text_content: str) -> list[dict]:
    """Extract the standardized result dictionaries from the model reply.

    Uses the cookbook regex and json.loads. Returns a list of dicts, each
    with a 'box_2d' (or 'point_2d') key and a 'label' key.
    """
    results = []
    for chunk in _RESULT_DICT.findall(text_content):
        try:
            results.append(json.loads(chunk))
        except json.JSONDecodeError:
            continue
    return results


def extract_boxes(text_content: str) -> Optional[dict]:
    """Extract the source/destination bounding boxes (normalized 0-999).

    Returns {'source': (xmin, ymin, xmax, ymax), 'dest': (...)} or None.
    """
    if re.search(r'BOXES\s*:\s*NONE', text_content, re.IGNORECASE):
        return None
    boxes: dict = {}
    for item in extract_results(text_content):
        label = str(item.get('label', '')).lower()
        box = item.get('box_2d')
        if box is None or len(box) != 4:
            continue
        coords = tuple(float(v) for v in box)
        if label.startswith('source') or label.startswith('from'):
            boxes['source'] = coords
        elif label.startswith('dest') or label.startswith('to'):
            boxes['dest'] = coords
    if 'source' in boxes and 'dest' in boxes:
        return boxes
    return None


def extract_squares(text_content: str) -> Optional[tuple[str, str]]:
    """Extract the from/to squares (algebraic notation, e.g. ('e2', 'e4')).

    Looks for 'from: <sq>' / 'to: <sq>' lines first, then falls back to prose
    patterns like 'e2 -> e4', 'e2 to e4', 'from e2 to e4'. Returns None if not
    found or if the model reported NONE.
    """
    if re.search(r'SQUARES\s*:\s*NONE', text_content, re.IGNORECASE):
        return None
    frm = to = None
    m = re.search(r'\b(?:from|source)\s*:\s*(' + _SQUARE + r')\b',
                  text_content, re.IGNORECASE)
    if m:
        frm = m.group(1).lower()
    m = re.search(r'\b(?:to|dest(?:ination)?)\s*:\s*(' + _SQUARE + r')\b',
                  text_content, re.IGNORECASE)
    if m:
        to = m.group(1).lower()
    if frm and to:
        return frm, to
    m = re.search(r'\bfrom\s+(' + _SQUARE + r')\s+to\s+(' + _SQUARE + r')\b',
                  text_content, re.IGNORECASE)
    if m:
        return m.group(1).lower(), m.group(2).lower()
    m = re.search(r'\b(' + _SQUARE + r')\s*(?:->|→|—|–|-)\s*'
                  r'(' + _SQUARE + r')\b', text_content, re.IGNORECASE)
    if m:
        return m.group(1).lower(), m.group(2).lower()
    return None


def _box_center_px(box: tuple, w: int, h: int) -> tuple[int, int]:
    """Convert a normalized (0-999) box to its pixel center (x, y)."""
    x1, y1, x2, y2 = (max(0.0, min(999.0, float(v))) for v in box)
    return (round((x1 + x2) / 2 / 1000 * w), round((y1 + y2) / 2 / 1000 * h))


def parse_detection(raw: str, move: str, game: str,
                    img_w: Optional[int], img_h: Optional[int]) -> DetectedMove:
    """Parse a model reply into a DetectedMove."""
    boxes = extract_boxes(raw)
    squares = extract_squares(raw)
    source_box = boxes['source'] if boxes else None
    dest_box = boxes['dest'] if boxes else None
    source_point = dest_point = None
    if source_box and img_w and img_h:
        source_point = _box_center_px(source_box, img_w, img_h)
    if dest_box and img_w and img_h:
        dest_point = _box_center_px(dest_box, img_w, img_h)
    return DetectedMove(
        move=move,
        game=game,
        from_square=squares[0] if squares else None,
        to_square=squares[1] if squares else None,
        source_box=source_box,
        dest_box=dest_box,
        source_point=source_point,
        dest_point=dest_point,
        image_size=(img_w, img_h) if img_w and img_h else None,
        raw=raw,
    )


# ── Visualization (draw the detected move on the picture) ───────────────────

def _box_to_px(box: tuple, w: int, h: int) -> tuple[int, int, int, int]:
    """Convert a normalized (0-999) box to a pixel rectangle (x1, y1, x2, y2)."""
    x1, y1, x2, y2 = (max(0.0, min(999.0, float(v))) for v in box)
    return (round(x1 / 1000 * w), round(y1 / 1000 * h),
            round(x2 / 1000 * w), round(y2 / 1000 * h))


def _draw_arrow(draw, x1: int, y1: int, x2: int, y2: int,
                color: tuple, width: int) -> None:
    """Line from (x1, y1) to (x2, y2) with an arrowhead at the end."""
    draw.line([(x1, y1), (x2, y2)], fill=color, width=width)
    angle = math.atan2(y2 - y1, x2 - x1)
    head = 18
    for da in (math.radians(150), math.radians(-150)):
        draw.line([(x2, y2),
                   (x2 + head * math.cos(angle + da),
                    y2 + head * math.sin(angle + da))],
                  fill=color, width=width)


def draw_detected_move(image_path: str, det: DetectedMove,
                       out_path: str) -> str:
    """
    Draw a detected move on the board picture and save it to out_path.

    Renders the source box (green), the destination box (orange), a drag
    arrow with arrowhead from the source center to the destination center
    (yellow), endpoint dots, and from/to labels. Self-contained (PIL only);
    does not use the repo's overlay module.

    Returns out_path.
    """
    from PIL import Image, ImageDraw

    img = Image.open(image_path).convert('RGB')
    w, h = img.size
    draw = ImageDraw.Draw(img)
    lw = max(2, w // 400)

    if det.source_box:
        draw.rectangle(_box_to_px(det.source_box, w, h),
                       outline=SRC_COLOR, width=lw + 1)
    if det.dest_box:
        draw.rectangle(_box_to_px(det.dest_box, w, h),
                       outline=DST_COLOR, width=lw + 1)

    if det.source_point and det.dest_point:
        sx, sy = det.source_point
        dx, dy = det.dest_point
        _draw_arrow(draw, sx, sy, dx, dy, MOVE_COLOR, lw)
        r = 6
        draw.ellipse([sx - r, sy - r, sx + r, sy + r], fill=MOVE_COLOR)
        draw.ellipse([dx - r, dy - r, dx + r, dy + r], fill=MOVE_COLOR)

    # labels: from/to squares above the boxes, move summary at top-left
    if det.source_box and det.from_square:
        bx1, by1, _, _ = _box_to_px(det.source_box, w, h)
        draw.text((bx1, max(0, by1 - 14)), f"from {det.from_square}",
                  fill=(255, 255, 255))
    if det.dest_box and det.to_square:
        bx1, by1, _, _ = _box_to_px(det.dest_box, w, h)
        draw.text((bx1, max(0, by1 - 14)), f"to {det.to_square}",
                  fill=(255, 255, 255))
    if det.from_square and det.to_square:
        draw.text((8, 8),
                  f"{det.game}: {det.move}  {det.from_square} -> "
                  f"{det.to_square}", fill=(255, 255, 255))

    img.save(out_path)
    return out_path


# ── Chess self-correction (deterministic legality check + retry) ─────────────
#
# A move like "Nf3" does not name the source square — several knights may
# exist, so the model must read the board to pick the one that can legally
# reach the destination. Models sometimes guess the wrong source (e.g. report
# g1 for "Nc3", but g1 -> c3 is not a legal knight move). To make detection
# reliable we validate the reported source -> destination against the move's
# geometry (no board needed) and, if it is illegal, ask the model to re-examine
# the board and try again.

_PIECE_NAMES = {'N': 'knight', 'B': 'bishop', 'R': 'rook',
                'Q': 'queen', 'K': 'king'}


def _parse_san(move: str) -> dict:
    """Parse a chess move in SAN-like notation into its components.

    Returns {'piece': 'N'|'B'|'R'|'Q'|'K'|'P', 'capture': bool, 'to': 'e4',
    'castle': bool, 'promo': str|None}. Only geometry-relevant parts are
    extracted; this is NOT a full chess rules engine.
    """
    s = move.strip()
    if s in ('O-O', '0-0', 'O-O-O', '0-0-0'):
        return {'piece': 'K', 'capture': False, 'to': None,
                'castle': True, 'promo': None}
    promo = None
    if '=' in s:
        s, promo = s.split('=', 1)
        promo = promo[:1] or None
    piece = 'P'
    if s and s[0] in 'NBRQK':
        piece = s[0]
        s = s[1:]
    capture = 'x' in s
    if capture:
        s = s.replace('x', '')
    to = s[-2:].lower() if len(s) >= 2 else None
    return {'piece': piece, 'capture': capture, 'to': to,
            'castle': False, 'promo': promo}


def _sq_rc(sq: str) -> tuple[int, int]:
    """Square -> (file, rank), both 0-based: a1 -> (0, 0), h8 -> (7, 7)."""
    return (ord(sq[0].lower()) - ord('a'), int(sq[1]) - 1)


def _geometry_ok(piece: str, frm: str, to: str) -> Optional[bool]:
    """Is frm -> to a legal move SHAPE for the piece? (No board needed.)

    Returns True/False, or None if it cannot be decided without the board
    (pawns need colour/occupancy; castling is handled separately).
    """
    if not frm or not to or len(frm) != 2 or len(to) != 2:
        return None
    fc, fr = _sq_rc(frm)
    tc, tr = _sq_rc(to)
    df, dr = tc - fc, tr - fr
    adf, adr = abs(df), abs(dr)
    if piece == 'N':
        return (adr, adf) in ((1, 2), (2, 1))
    if piece == 'B':
        return adr == adf and adr > 0
    if piece == 'R':
        return (adr == 0) != (adf == 0)
    if piece == 'Q':
        return (adr == adf and adr > 0) or ((adr == 0) != (adf == 0))
    if piece == 'K':
        return max(adr, adf) == 1
    return None  # pawn / unknown — needs the board


def _chess_corrective_prompt(move: str, det: DetectedMove) -> Optional[str]:
    """If the model's from/to is not a legal move for the piece, build a
    corrective follow-up message; otherwise return None (no retry needed)."""
    info = _parse_san(move)
    piece = info['piece']
    if piece not in _PIECE_NAMES:
        return None
    pname = _PIECE_NAMES[piece]
    to = det.to_square or info['to']
    if det.from_square is None:
        return (
            f'For the move "{move}" you did not report a source square. '
            f'Re-examine the board: find the {pname} that can legally reach '
            f'{to} and report ITS square as the source. Reply in the same '
            f'format (two box_2d dictionaries, then from:/to: lines).'
        )
    if _geometry_ok(piece, det.from_square, det.to_square) is False:
        return (
            f'Your previous answer said source {det.from_square} and '
            f'destination {det.to_square} for "{move}", but '
            f'{det.from_square} -> {det.to_square} is NOT a legal {pname} '
            f'move. Re-examine the board: find the {pname} that can legally '
            f'reach {det.to_square} and report ITS square as the source. '
            f'Reply in the same format (two box_2d dictionaries, then '
            f'from:/to: lines).'
        )
    return None


# ── Board detection (deterministic square pixel boxes for chess) ────────────
#
# The model reliably resolves the from/to SQUARES (with the legality retry),
# but it often mis-localizes a square's pixel box (e.g. it placed the Nc3
# destination box on c2 instead of c3). So for chess we detect the 8x8 board
# grid straight from the image (its regular alternating-square edges) and
# compute the square pixel boxes ourselves, overriding the model's boxes.

_BOARD_CACHE: dict = {}


def _find_grid_lines(strength: list) -> Optional[list]:
    """Find 9 evenly-spaced strong peaks (the boundaries of 8 squares).

    RANSAC-style: search for the spacing/offset whose 9 expected positions
    (start + k*spacing, k=0..8) match the most peaks (>=7). Returns the 9
    boundary positions, or None.
    """
    maxs = max(strength) if strength else 0.0
    if maxs <= 0:
        return None
    thr = maxs * 0.15
    peaks = [i for i in range(1, len(strength) - 1)
             if strength[i] >= thr and strength[i] >= strength[i - 1]
             and strength[i] > strength[i + 1]]
    if len(peaks) < 7:
        return None
    peaks.sort()
    best = None  # ((matches, strength_sum), spacing, matched_positions)
    for s in range(15, 260):
        tol = max(3, int(s * 0.12))
        for p0 in peaks:
            matched = []
            for k in range(9):
                target = p0 + k * s
                idx = bisect.bisect_left(peaks, target)
                cand = None
                for ii in (idx - 1, idx):
                    if 0 <= ii < len(peaks) and abs(peaks[ii] - target) <= tol:
                        cand = peaks[ii]
                        break
                if cand is not None:
                    matched.append(cand)
            if len(matched) >= 7:
                key = (len(matched), sum(strength[p] for p in matched))
                if best is None or key > best[0]:
                    best = (key, s, matched)
    if best is None:
        return None
    _, s, matched = best
    bounds = sorted(matched)
    while len(bounds) < 9:
        bounds = [bounds[0] - s] + bounds + [bounds[-1] + s]
    return bounds[:9]


def _detect_chess_board(image_path: str) -> Optional[tuple]:
    """Detect the 8x8 chess board grid. Returns (left, top, sq_w, sq_h) in
    pixels, or None if no regular grid is found. Cached per image path."""
    key = os.path.abspath(image_path)
    if key in _BOARD_CACHE:
        return _BOARD_CACHE[key]
    result = None
    try:
        from PIL import Image
        img = Image.open(image_path).convert('L')
        W, H = img.size
        gray = list(img.getdata())  # ints 0-255, length W*H

        gx = [0.0] * (W - 1)  # vertical-edge strength per column (file lines)
        for y in range(H):
            base = y * W
            row = gray[base:base + W]
            for x in range(W - 1):
                gx[x] += abs(row[x + 1] - row[x])
        gy = [0.0] * (H - 1)  # horizontal-edge strength per row (rank lines)
        for y in range(H - 1):
            base = y * W
            nxt = base + W
            for x in range(W):
                gy[y] += abs(gray[nxt + x] - gray[base + x])

        bx = _find_grid_lines(gx)
        by = _find_grid_lines(gy)
        if bx and by:
            left, right = bx[0], bx[-1]
            top, bottom = by[0], by[-1]
            sq_w = (right - left) / 8.0
            sq_h = (bottom - top) / 8.0
            # sanity: a board is large and roughly square
            if 20 <= sq_w <= 400 and 20 <= sq_h <= 400 and \
                    abs(sq_w - sq_h) <= 0.25 * max(sq_w, sq_h):
                result = (left, top, sq_w, sq_h)
    except Exception:
        result = None
    _BOARD_CACHE[key] = result
    return result


def _square_px_box(board: tuple, sq: str) -> tuple:
    """Pixel rectangle (x1, y1, x2, y2) of a square from the board grid."""
    left, top, sw, sh = board
    f = ord(sq[0].lower()) - ord('a')
    r = int(sq[1]) - 1
    x = left + f * sw
    y = top + (7 - r) * sh
    return (x, y, x + sw, y + sh)


def _px_to_norm_box(box_px: tuple, w: int, h: int) -> tuple:
    """Convert a pixel rectangle to a normalized (0-999) box."""
    x1, y1, x2, y2 = box_px
    return (round(x1 / w * 1000), round(y1 / h * 1000),
            round(x2 / w * 1000), round(y2 / h * 1000))


def _px_box_center(box_px: tuple) -> tuple:
    return (round((box_px[0] + box_px[2]) / 2),
            round((box_px[1] + box_px[3]) / 2))


# ── Public API ──────────────────────────────────────────────────────────────

def detect_move(api_key: str, image_path: str, move: str,
                game: str = "chess", model: str = MODEL,
                temperature: float = 0.0,
                reasoning_effort: str = "none",
                overlay_path: Optional[str] = None) -> DetectedMove:
    """
    Detect a game move on a board image using the Mistral endpoint.

    Sends the screenshot and the move (e.g. "e4") to Mistral Large 4 and
    resolves it into a from/to pair: the source square (piece being moved)
    and the destination square, returned both in algebraic notation and as
    normalized bounding boxes (with pixel centers).

    api_key:         Mistral API key.
    image_path:      path to the board screenshot (PNG/JPEG/...).
    move:            the move to detect, e.g. "e4", "Nf3", "exd5", "O-O".
    game:            game name (default "chess"); used in the prompts.
    model:           Mistral model (default mistral-large-4).
    temperature:     sampling temperature (default 0 = greedy).
    reasoning_effort: Mistral reasoning level — "none" (default, no
                     chain-of-thought), "low", "medium", or "high".
    overlay_path:    if given, draw the detected move (source/destination
                     boxes + drag arrow) on the picture and save it here.

    For chess, the reported source -> destination is checked against the
    move's geometry; if it is illegal (e.g. a knight move whose source was
    not in the notation), the model is asked to re-examine the board and
    retry (up to 2 times).

    Returns a DetectedMove (with .overlay_path set when a picture was drawn).
    Raises RuntimeError on API/transport errors.
    """
    if not api_key:
        raise ValueError("api_key is required (set MISTRAL_API_KEY)")
    if not move or not move.strip():
        raise ValueError("move is required (e.g. 'e4')")

    mv = move.strip()
    img_w, img_h = _image_size(image_path)
    image_b64, mime_type = _encode_image(image_path)
    system_prompt = build_system_prompt(game)
    user_prompt = build_user_prompt(mv, game, img_w, img_h)
    messages = [_msg_system(system_prompt),
                _msg_user_image(user_prompt, image_b64, mime_type)]
    raw = _chat_completion(api_key, messages, model, temperature, reasoning_effort)
    det = parse_detection(raw, mv, game, img_w, img_h)

    # Chess self-correction: if the reported source -> destination is not a
    # legal move for the piece (e.g. a knight whose source was not in the
    # notation), ask the model to re-examine the board and try again.
    if game == "chess":
        for _ in range(2):
            corrective = _chess_corrective_prompt(mv, det)
            if corrective is None:
                break
            messages.append(_msg_assistant(raw))
            messages.append(_msg_user_text(corrective))
            raw = _chat_completion(api_key, messages, model, temperature,
                                   reasoning_effort)
            det = parse_detection(raw, mv, game, img_w, img_h)

    # For chess, recompute the pixel boxes from the detected board grid: the
    # model reliably gets the from/to squares (verified above) but often
    # mis-localizes a square's pixel box (e.g. Nc3's destination landed on c2).
    # The detected grid gives exact, consistent boxes for both squares.
    if game == "chess" and det.from_square and det.to_square and img_w and img_h:
        board = _detect_chess_board(image_path)
        if board:
            sb_px = _square_px_box(board, det.from_square)
            db_px = _square_px_box(board, det.to_square)
            det.source_box = _px_to_norm_box(sb_px, img_w, img_h)
            det.dest_box = _px_to_norm_box(db_px, img_w, img_h)
            det.source_point = _px_box_center(sb_px)
            det.dest_point = _px_box_center(db_px)

    if overlay_path:
        draw_detected_move(image_path, det, overlay_path)
        det.overlay_path = overlay_path
    return det


def detect_move_from_to(api_key: str, image_path: str, move: str,
                        game: str = "chess", model: str = MODEL) -> tuple:
    """
    Convenience wrapper: detect a move and return (from_square, to_square),
    e.g. ("e2", "e4"). Either may be None if the model could not resolve it.
    """
    return detect_move(api_key, image_path, move, game=game, model=model).from_to


def detect_and_draw(api_key: str, image_path: str, move: str,
                    out_path: str, game: str = "chess", model: str = MODEL,
                    temperature: float = 0.0,
                    reasoning_effort: str = "none") -> tuple[DetectedMove, str]:
    """
    Detect a move and draw it on the picture in one call.
    Returns (DetectedMove, out_path).
    """
    det = detect_move(api_key, image_path, move, game=game, model=model,
                      temperature=temperature, reasoning_effort=reasoning_effort,
                      overlay_path=out_path)
    return det, out_path


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python3 move_detector.py <image_path> <move> "
              "[api_key] [overlay_out.png]")
        sys.exit(1)
    img = sys.argv[1]
    mv = sys.argv[2]
    key = sys.argv[3] if len(sys.argv) > 3 else os.environ.get('MISTRAL_API_KEY', '')
    out = (sys.argv[4] if len(sys.argv) > 4
           else os.path.join(tempfile.gettempdir(), f'move_detector_{mv}.png'))
    if not key:
        print("Error: Set MISTRAL_API_KEY")
        sys.exit(1)
    result = detect_move(key, img, mv, overlay_path=out)
    print(result.describe())
    print('annotated photo ->', out)
    print()
    print('--- raw API response ---')
    print(result.raw)
