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
        f"square it moves to). Report their bounding boxes (labels 'source' "
        f"and 'destination') and the from/to squares."
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


def _build_payload(image_path: str, user_prompt: str, system_prompt: str,
                   model: str, temperature: float,
                   reasoning_effort: str = "none") -> dict:
    """Build the chat-completions request payload.

    temperature defaults to 0 (greedy sampling); the Mistral API rejects
    greedy sampling unless top_p is 1, so top_p is always set to 1.
    reasoning_effort defaults to "none" (no chain-of-thought), matching the
    deterministic, token-light behaviour used elsewhere in this project.
    """
    image_b64, mime_type = _encode_image(image_path)
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url",
                 "image_url": {"url": f"data:{mime_type};base64,{image_b64}"}},
            ]},
        ],
        "temperature": temperature,
        "top_p": 1,
        "reasoning_effort": reasoning_effort,
        "max_tokens": 4096,
    }


def _chat_completion(api_key: str, image_path: str, user_prompt: str,
                     system_prompt: str, model: str,
                     temperature: float = 0.0,
                     reasoning_effort: str = "none", timeout: int = 600) -> str:
    """Send image + prompts to the Mistral endpoint. Returns the reply text."""
    payload = _build_payload(image_path, user_prompt, system_prompt,
                             model, temperature, reasoning_effort)

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

    Returns a DetectedMove (with .overlay_path set when a picture was drawn).
    Raises RuntimeError on API/transport errors.
    """
    if not api_key:
        raise ValueError("api_key is required (set MISTRAL_API_KEY)")
    if not move or not move.strip():
        raise ValueError("move is required (e.g. 'e4')")

    img_w, img_h = _image_size(image_path)
    system_prompt = build_system_prompt(game)
    user_prompt = build_user_prompt(move.strip(), game, img_w, img_h)
    raw = _chat_completion(api_key, image_path, user_prompt, system_prompt,
                           model, temperature, reasoning_effort)
    det = parse_detection(raw, move.strip(), game, img_w, img_h)
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
