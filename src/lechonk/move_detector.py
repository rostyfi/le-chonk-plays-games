#!/usr/bin/env python3
"""
Move detector — uses the Mistral endpoint to detect a move on a screenshot
and resolve it into a source/destination pair.

Given a screenshot and a move description, this module asks Mistral Large 4
(vision) to locate the objects involved in the move and returns them as
normalized bounding boxes (pixel centers included), following the
object-detection recipe from the Mistral cookbook:

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

Prompts and behaviour config are loaded from prompts.json via prompt_loader.
The Mistral HTTP endpoint is called with the standard library (urllib) only,
so it has no third-party dependency. PIL is used to read the image size and
to draw the move overlay; it is optional for detection (required only when
drawing).

It can also SHOW the move on the picture: pass overlay_path=... to
detect_move (or use detect_and_draw / the CLI) and it draws the source box,
destination box and a drag arrow over the screenshot and saves the annotated
image.

Run as a script (detects the move and writes an annotated picture):
    python3 move_detector.py <image_path> <move> [api_key] [overlay_out.png]
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

try:
    from .prompt_loader import get_detector_system, build_detector_user_prompt
except ImportError:  # run directly as a script: python3 move_detector.py ...
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from lechonk.prompt_loader import get_detector_system, build_detector_user_prompt

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

# A square in algebraic notation: file a-h, rank 1-8.
_SQUARE = r'[a-hA-H][1-8]'


# ── Prompts (loaded from prompts.json via prompt_loader) ────────────

def build_system_prompt(game_key: str = "default") -> str:
    """System prompt loaded from prompts.json (detector.system.<game_key>)."""
    return get_detector_system(game_key)


def build_user_prompt(move: str, game_key: str = "default",
                      img_w: Optional[int] = None,
                      img_h: Optional[int] = None) -> str:
    """User prompt built from the template in prompts.json
    (detector.user_template.<game_key>) with {game}, {move}, {size} substituted."""
    return build_detector_user_prompt(move, game_key, img_w, img_h)


# ── Result type ─────────────────────────────────────────────────────────────

@dataclass
class DetectedMove:
    """A detected move resolved into a source/destination pair."""
    move: str                       # the input move text
    game_key: str                   # game key from prompts.json
    from_square: Optional[str]      # algebraic source square (board games)
    to_square: Optional[str]        # algebraic destination square (board games)
    source_box: Optional[tuple]     # normalized (xmin, ymin, xmax, ymax) 0-999
    dest_box: Optional[tuple]       # normalized (xmin, ymin, xmax, ymax) 0-999
    source_point: Optional[tuple]   # pixel (x, y) center of the source box
    dest_point: Optional[tuple]     # pixel (x, y) center of the destination box
    image_size: Optional[tuple]     # (width, height) of the image in pixels
    raw: str                        # raw model reply
    overlay_path: Optional[str] = None  # annotated picture, if one was drawn
    description: Optional[str] = None   # natural-language move desc (non-board games)

    @property
    def from_to(self) -> tuple:
        return (self.from_square, self.to_square)

    def describe(self) -> str:
        if self.from_square:
            ft = f"{self.from_square} -> {self.to_square}"
        elif self.description:
            ft = self.description
        else:
            ft = "?"
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
                   reasoning_effort: str = "low") -> dict:
    """Build the chat-completions request payload.

    temperature defaults to 0 (greedy sampling); the Mistral API rejects
    greedy sampling unless top_p is 1, so top_p is always set to 1.
    reasoning_effort defaults to "low" (light chain-of-thought), matching the
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
                     reasoning_effort: str = "low", timeout: int = 600) -> str:
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
            # models sometimes emit single-quoted pseudo-JSON
            try:
                results.append(json.loads(chunk.replace("'", '"')))
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


# CENTERS block: the drag endpoints as explicit pixel centers, e.g.
#   CENTERS:
#   source: (584, 562)
#   dest: (584, 464)
# When present, these pixel endpoints are used exactly as written.
CENTER_LINE = re.compile(
    r'(source|dest(?:ination)?)\b[^:\n]{0,15}:\s*\(\s*(-?\d+(?:\.\d+)?)\s*[, ]\s*'
    r'(-?\d+(?:\.\d+)?)\s*\)',
    re.IGNORECASE
)


def extract_centers(text_content: str) -> Optional[dict]:
    """Extract the CENTERS section: the move endpoints as pixel centers.

    Returns {'source': (x, y), 'dest': (x, y)} (image pixels) or None.
    """
    if re.search(r'CENTERS\s*:\s*NONE', text_content, re.IGNORECASE):
        return None
    centers: dict = {}
    for m in CENTER_LINE.finditer(text_content):
        label = m.group(1).lower()
        point = (float(m.group(2)), float(m.group(3)))
        if label.startswith('source'):
            centers['source'] = point
        elif label.startswith('dest'):
            centers['dest'] = point
    if 'source' in centers and 'dest' in centers:
        return centers
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


def extract_description(text_content: str) -> Optional[str]:
    """Extract the move description line.

    Looks for 'move: <description>' lines. Returns None if not found.
    """
    m = re.search(r'\bmove\s*:\s*(.+)', text_content, re.IGNORECASE)
    if m:
        desc = m.group(1).strip()
        if desc and desc.upper() != 'NONE':
            return desc
    return None


def _box_center_px(box: tuple, w: int, h: int) -> tuple[int, int]:
    """Convert a normalized (0-999) box to its pixel center (x, y)."""
    x1, y1, x2, y2 = (max(0.0, min(999.0, float(v))) for v in box)
    return (round((x1 + x2) / 2 / 1000 * w), round((y1 + y2) / 2 / 1000 * h))


def parse_detection(raw: str, move: str, game_key: str,
                    img_w: Optional[int], img_h: Optional[int],
                    behavior: Optional[dict] = None) -> DetectedMove:
    """Parse a model reply into a DetectedMove.

    Endpoint pixels come from the CENTERS block when present (used exactly
    as written); otherwise from the box_2d dictionaries (normalized 0-999,
    centers computed in pixels). Boxes with any coordinate > 999 are treated
    as pixel coordinates, not normalized ones.

    behavior: dict from the prompt's 'behavior' config. When
    behavior['move_format'] is 'algebraic' or 'squares', the reply is
    expected to contain from/to square lines; otherwise a natural-language
    description line is extracted.
    """
    behavior = behavior or {}
    move_format = behavior.get("move_format")
    has_squares = move_format in ("algebraic", "squares")
    boxes = extract_boxes(raw)
    centers = extract_centers(raw)
    if re.search(r'(?:BOXES|CENTERS)\s*:\s*NONE', raw, re.IGNORECASE):
        # the model explicitly could not locate the move — do not fabricate
        # endpoints from a degenerate CENTERS block (e.g. source: (0, 0))
        boxes = None
        centers = None
    squares = extract_squares(raw) if has_squares else None
    description = None if has_squares else extract_description(raw)
    source_box = boxes['source'] if boxes else None
    dest_box = boxes['dest'] if boxes else None
    if centers and 'source' in centers and 'dest' in centers and img_w and img_h:
        # CENTERS are authoritative pixel endpoints; drop boxes that disagree
        # with them (the model sometimes emits pixel coords as box_2d, which
        # must not be drawn or scaled as normalized).
        for name in ('source', 'dest'):
            box = source_box if name == 'source' else dest_box
            if box is None:
                continue
            cx = (box[0] + box[2]) / 2 / 1000 * img_w
            cy = (box[1] + box[3]) / 2 / 1000 * img_h
            px, py = centers[name]
            if abs(cx - px) > 40 or abs(cy - py) > 40:
                if name == 'source':
                    source_box = None
                else:
                    dest_box = None
    if (source_box and max(source_box) > 999) or (dest_box and max(dest_box) > 999):
        # Coordinates beyond the 0-999 normalized range are pixels, not
        # normalized values — interpret the boxes as pixel rectangles.
        if not centers:
            centers = {}
        if source_box and 'source' not in centers:
            centers['source'] = ((source_box[0] + source_box[2]) / 2,
                                 (source_box[1] + source_box[3]) / 2)
        if dest_box and 'dest' not in centers:
            centers['dest'] = ((dest_box[0] + dest_box[2]) / 2,
                               (dest_box[1] + dest_box[3]) / 2)
        source_box = dest_box = None
    source_point = dest_point = None
    if centers and 'source' in centers and 'dest' in centers:
        # explicit pixel centers — used exactly as written (clamped to image)
        sx, sy = centers['source']
        dx, dy = centers['dest']
        if img_w and img_h:
            sx = min(max(sx, 0), img_w - 1)
            sy = min(max(sy, 0), img_h - 1)
            dx = min(max(dx, 0), img_w - 1)
            dy = min(max(dy, 0), img_h - 1)
        source_point = (round(sx), round(sy))
        dest_point = (round(dx), round(dy))
    else:
        if source_box and img_w and img_h:
            source_point = _box_center_px(source_box, img_w, img_h)
        if dest_box and img_w and img_h:
            dest_point = _box_center_px(dest_box, img_w, img_h)
    return DetectedMove(
        move=move,
        game_key=game_key,
        from_square=squares[0] if squares else None,
        to_square=squares[1] if squares else None,
        source_box=source_box,
        dest_box=dest_box,
        source_point=source_point,
        dest_point=dest_point,
        image_size=(img_w, img_h) if img_w and img_h else None,
        raw=raw,
        description=description,
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

    # labels: from/to squares or description
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
                  f"{det.game_key}: {det.move}  {det.from_square} -> "
                  f"{det.to_square}", fill=(255, 255, 255))
    elif det.description:
        draw.text((8, 8),
                  f"{det.game_key}: {det.move}  ({det.description})",
                  fill=(255, 255, 255))

    img.save(out_path)
    return out_path


# ── Move self-correction (deterministic geometry check + retry) ──────────────
#
# A move like "Nf3" does not name the source square — several pieces of the
# same type may exist, so the model must read the board to pick the one that
# can legally reach the destination. Models sometimes guess the wrong source.
# To make detection reliable we validate the reported source -> destination
# against the move's geometry (no board needed) and, if it is inconsistent,
# ask the model to re-examine the board and try again.
#
# Enabled by behavior['validate_move'] = true in prompts.json.

_PIECE_NAMES = {'N': 'knight', 'B': 'bishop', 'R': 'rook',
                'Q': 'queen', 'K': 'king'}


def _parse_algebraic_move(move: str) -> dict:
    """Parse a move in algebraic notation into its components.

    Returns {'piece': 'N'|'B'|'R'|'Q'|'K'|'P', 'capture': bool, 'to': 'e4',
    'castle': bool, 'promo': str|None}. Only geometry-relevant parts are
    extracted; this is NOT a full rules engine.
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


def _move_corrective_prompt(move: str, det: DetectedMove) -> Optional[str]:
    """If the model's from/to is not a legal move for the piece, build a
    corrective follow-up message; otherwise return None (no retry needed)."""
    info = _parse_algebraic_move(move)
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


# ── Board grid detection (deterministic pixel boxes for grid-based games) ───
#
# The model reliably resolves the from/to SQUARES (with the legality retry),
# but it often mis-localizes a square's pixel box. So when behavior['board_grid']
# is true, we detect the regular grid straight from the image (its evenly-spaced
# edges) and compute the square pixel boxes ourselves, overriding the model's boxes.

_BOARD_CACHE: dict = {}


def _find_grid_lines(strength: list) -> Optional[list]:
    """Find 9 evenly-spaced strong peaks (the boundaries of 8 cells).

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


def _detect_board_grid(image_path: str) -> Optional[tuple]:
    """Detect an 8x8 grid on the image. Returns (left, top, cell_w, cell_h)
    in pixels, or None if no regular grid is found. Cached per image path."""
    key = os.path.abspath(image_path)
    if key in _BOARD_CACHE:
        return _BOARD_CACHE[key]
    result = None
    try:
        from PIL import Image
        img = Image.open(image_path).convert('L')
        W, H = img.size
        gray = list(img.getdata())  # ints 0-255, length W*H

        gx = [0.0] * (W - 1)  # vertical-edge strength per column
        for y in range(H):
            base = y * W
            row = gray[base:base + W]
            for x in range(W - 1):
                gx[x] += abs(row[x + 1] - row[x])
        gy = [0.0] * (H - 1)  # horizontal-edge strength per row
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
            # sanity: cells are reasonably sized and roughly square
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
                game_key: str = "default", behavior: Optional[dict] = None,
                model: str = MODEL,
                temperature: float = 0.0,
                reasoning_effort: str = "low",
                overlay_path: Optional[str] = None) -> DetectedMove:
    """
    Detect a move on a screenshot using the Mistral endpoint.

    Sends the screenshot and the move to Mistral Large 4 and resolves it
    into source/destination bounding boxes (with pixel centers).

    api_key:         Mistral API key.
    image_path:      path to the screenshot (PNG/JPEG/...).
    move:            the move to detect.
    game_key:        key into prompts.json detector.system / detector.user_template.
    behavior:        dict from the prompt's 'behavior' config. Recognised keys:
                       action        — "drag" or "click" (informational)
                       grab_y        — grab at this fraction of source box height
                       board_grid    — detect a regular grid and compute exact boxes
                       validate_move — validate reported source→dest geometry and retry
                       move_format   — "algebraic" or "squares" (expect from/to lines)
                       right_click_word — word in move text that means right-click
    model:           Mistral model (default mistral-large-4).
    temperature:     sampling temperature (default 0 = greedy).
    reasoning_effort: Mistral reasoning level — "low" (default), "none",
                     "medium", or "high".
    overlay_path:    if given, draw the detected move on the picture and save it.

    When behavior['validate_move'] is true, the reported source -> destination
    is checked against the move's geometry; if inconsistent, the model is asked
    to re-examine the board and retry (up to 2 times).

    When behavior['board_grid'] is true and from/to squares are resolved,
    the pixel boxes are recomputed from the detected grid for accuracy.

    Returns a DetectedMove (with .overlay_path set when a picture was drawn).
    Raises RuntimeError on API/transport errors.
    """
    if not api_key:
        raise ValueError("api_key is required (set MISTRAL_API_KEY)")
    if not move or not move.strip():
        raise ValueError("move is required")

    behavior = behavior or {}
    mv = move.strip()
    img_w, img_h = _image_size(image_path)
    image_b64, mime_type = _encode_image(image_path)
    system_prompt = build_system_prompt(game_key)
    user_prompt = build_user_prompt(mv, game_key, img_w, img_h)
    messages = [_msg_system(system_prompt),
                _msg_user_image(user_prompt, image_b64, mime_type)]
    raw = _chat_completion(api_key, messages, model, temperature, reasoning_effort)
    det = parse_detection(raw, mv, game_key, img_w, img_h, behavior)

    # Move self-correction: if the reported source -> destination is not a
    # legal move for the piece, ask the model to re-examine the board.
    if behavior.get("validate_move"):
        for _ in range(2):
            corrective = _move_corrective_prompt(mv, det)
            if corrective is None:
                break
            messages.append(_msg_assistant(raw))
            messages.append(_msg_user_text(corrective))
            raw = _chat_completion(api_key, messages, model, temperature,
                                   reasoning_effort)
            det = parse_detection(raw, mv, game_key, img_w, img_h, behavior)

    # When a board grid is expected, recompute the pixel boxes from the
    # detected grid: the model reliably gets the from/to squares (verified
    # above) but often mis-localizes a square's pixel box.
    if behavior.get("board_grid") and det.from_square and det.to_square \
            and img_w and img_h:
        board = _detect_board_grid(image_path)
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
                        game_key: str = "default", behavior: Optional[dict] = None,
                        model: str = MODEL) -> tuple:
    """
    Convenience wrapper: detect a move and return (from_square, to_square).
    Either may be None if the model could not resolve it.
    """
    return detect_move(api_key, image_path, move, game_key=game_key,
                       behavior=behavior, model=model).from_to


def detect_and_draw(api_key: str, image_path: str, move: str,
                    out_path: str, game_key: str = "default",
                    behavior: Optional[dict] = None, model: str = MODEL,
                    temperature: float = 0.0,
                    reasoning_effort: str = "low") -> tuple[DetectedMove, str]:
    """
    Detect a move and draw it on the picture in one call.
    Returns (DetectedMove, out_path).
    """
    det = detect_move(api_key, image_path, move, game_key=game_key,
                      behavior=behavior, model=model,
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
