#!/usr/bin/env python3
"""
Minimal Mistral API client — sends a screenshot, gets move advice.
Model: mistral-large-4, temperature: 0, reasoning_effort: none.
"""

import base64
import json
import os
import sys
import urllib.request
import urllib.error

MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"
MODEL = "mistral-large-4"

PROMPTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts.json")

BASE_PROMPT = (
    "This is a Klondike Solitaire game (draw-1, standard rules). "
    "The image may contain interface around the playing table (window title "
    "bar, buttons, panels, desktop background) — IGNORE everything outside "
    "the playing table; only the cards and piles matter. "
    "First, locate the playing table in the image, then describe it: "
    "(1) For each of the 7 tableau columns (left to right), list every card "
    "top-to-bottom: rank, suit, face-up or face-down. "
    "EXPLICITLY state which columns are COMPLETELY EMPTY — an empty column "
    "is shown as a LIGHT GREEN RECTANGLE (a pale green placeholder box, "
    "lighter than the dark green felt background) with NO cards on it. "
    "Count these carefully — there may be zero, one, or several. "
    "(2) List the stock pile, waste pile, and 4 foundation piles "
    "(which suits are started, top card of each). "
    "(3) Note any visible score, move count, or timer. "
    "Do NOT guess or invent cards you cannot see. "
    "If you are unsure about a card, say so. "
    "Then, based ONLY on the visible cards, what is the single best legal move? "
    "Klondike rules: cards on the tableau build DOWN in alternating colors "
    "(e.g. 9♥ on 10♠). Foundations build UP by suit from Ace. "
    "ONLY a King (or a sequence starting with a King) can be placed "
    "on an EMPTY tableau column. "
    "Moving a King to an empty column is often the best move because "
    "it frees up the column it came from and opens new possibilities. "
    "If NO legal tableau or foundation move exists, the only correct answer is: "
    "'Draw from the stock pile.' Never suggest an illegal move. "
    "If the stock is empty and the waste cannot be recycled, say 'No moves available.' "
)

ADVICE_SUFFIX = (
    "You may briefly explain your reasoning, but the LAST sentence of your "
    "response must state the best move in one sentence — nothing may come after it."
)

# CHESS_PROMPT = (
#     "This is a game of CHESS. "
#     "First, describe the board exactly as you see it: an 8x8 grid of squares. "
#     "For every occupied square, give its coordinate (file a-h, rank 1-8 — use "
#     "the labels if the board shows them; otherwise describe the square by "
#     "position, e.g. 'the top-left square') and the piece on it (white or black "
#     "King, Queen, Rook, Bishop, Knight, or Pawn). "
#     "State which side is to move (White or Black). "
#     "Note castling rights (which Kings and Rooks have not yet moved) and any "
#     "en passant target square, if visible. "
#     "Do NOT guess or invent pieces you cannot see. "
#     "If you are unsure about a square, say so. "
#     "If the board is not fully visible, describe only what you can see. "
#     "COORDINATE PROTOCOL — follow it exactly when reporting pixel positions: "
#     "The image may contain extra UI around the board (clocks, move lists, "
#     "buttons, player names, panels) — IGNORE everything outside the 8x8 "
#     "checkerboard; it is not part of the board. "
#     "Step 1: locate the board — the square-shaped 8x8 grid of alternating "
#     "light and dark squares — and state its pixel bounds: the left, top, "
#     "right and bottom edges of the outermost squares, and the width of one "
#     "square in pixels. "
#     "Step 2: compute the pixel center of any square <file><rank> with this "
#     "exact formula: x = board_left + (file_number + 0.5) * square_width, "
#     "y = board_top + (8 - rank + 0.5) * square_height, where file_number is "
#     "a=0, b=1, c=2, d=3, e=4, f=5, g=6, h=7. Do not estimate — measure the "
#     "board edges and compute. "
#     "Step 3: report those computed pixel centers in the CENTERS section. "
#     "Coordinates are always in the WHOLE image: if the board does not fill "
#     "the image, the formula already includes the board's offset — never "
#     "report coordinates relative to the board or to any sub-region. "
#     "Then, based ONLY on the visible board, what is the single best legal move? "
#     "Chess rules: standard piece movement; never suggest an illegal move; "
#     "consider checks, captures, threats, and king safety. "
#     "State the move using algebraic notation (e.g. 'Nf3', 'exd5', 'O-O'). "
#     "If NO legal move exists, say 'No moves available.' "
#     "If the best move is a pawn promotion, say which piece to promote to and "
#     "add a 'CLICK at (x, y)' on the promotion piece in the ACTION block "
#     "(it runs after the DRAG). "
# )
CHESS_PROMPT = ("What's the best chess move?")

def encode_image(image_path: str) -> tuple[str, str]:
    """Encode image to base64. Returns (base64, mime_type)."""
    with open(image_path, 'rb') as f:
        data = base64.b64encode(f.read()).decode('utf-8')
    ext = os.path.splitext(image_path)[1].lower()
    mime = {'.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
            '.gif': 'image/gif', '.webp': 'image/webp'}.get(ext, 'image/png')
    return data, mime


def _extract_text(message: dict) -> str:
    """Extract text from API response message (handles string, list, thinking parts)."""
    content = message.get('content', '')
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for part in content:
            if isinstance(part, dict):
                if part.get('type') == 'text':
                    texts.append(part.get('text', ''))
                elif 'text' in part:
                    texts.append(part['text'])
            elif isinstance(part, str):
                texts.append(part)
        return '\n'.join(texts)
    return str(content)


def image_size(image_path: str) -> tuple[int | None, int | None]:
    """Return (width, height) of the image, or (None, None) if unreadable."""
    try:
        from PIL import Image
        with Image.open(image_path) as img:
            return img.width, img.height
    except Exception:
        return None, None


def _call_api(api_key: str, image_path: str, user_prompt: str, model: str) -> str:
    """Send image + prompt to Mistral API. Returns response text."""
    image_b64, mime_type = encode_image(image_path)

    # Debug: log image info
    img_kb = len(image_b64) * 3 // 4 // 1024
    try:
        from PIL import Image
        with Image.open(image_path) as img:
            dims = f"{img.width}x{img.height}"
    except Exception:
        dims = "?"
    print(f"  Image: {dims}, ~{img_kb}KB, {mime_type}", file=sys.stderr)

    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url",
                 "image_url": {"url": f"data:{mime_type};base64,{image_b64}"}},
            ],
        }],
        "max_tokens": 128000,
        "temperature": 0,
        "top_p": 1,
        "reasoning_effort": "none",
    }

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
        with urllib.request.urlopen(req, timeout=600) as resp:
            result = json.loads(resp.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"API error ({e.code}): {e.read().decode('utf-8')[:500]}")

    actual = result.get('model', '?')
    print(f"  Served by: {actual}", file=sys.stderr)

    choices = result.get('choices', [])
    if not choices:
        raise RuntimeError(f"No choices: {json.dumps(result)[:500]}")

    text = _extract_text(choices[0]['message'])
    if not text.strip():
        raise RuntimeError(f"Empty response (finish_reason={choices[0].get('finish_reason', '?')})")

    return text


def _actions_suffix(image_path: str) -> str:
    """Output-format contract appended when the move must be executable.

    Structured after the Mistral cookbook's object-detection prompt
    (mistral/image_understanding/bbox_getstarted): a short objective, the
    box_2d task definition, the standardized response format with a
    concrete example, and free generation — the dictionaries are parsed
    afterwards with a regex, like the cookbook's extract_results.
    """
    w, h = image_size(image_path)
    size_note = f"The screenshot is {w}x{h} pixels. " if w and h else ""
    return (
        "You may briefly explain your reasoning. "
        "You are an object-detection vision model for a game-playing "
        "agent. Given a game screenshot, output the bounding boxes of the "
        "two objects that define the next move: the object to grab (label "
        "'source') and the destination (label 'destination'). "
        "If an object is not present, reply with NONE. "
        + size_note
        + "Use pixel coordinates in this image: (0,0) is the TOP-LEFT corner, "
        "x grows to the right, y grows downward. "
        "Coordinates are always in the WHOLE image — if the game area does "
        "not fill the image (panels, clocks, or margins around it), include "
        "that offset; never report coordinates relative to the game area or "
        "a sub-region. "
        "# Task\n"
        "box_2d: detect the bounding box of each object requested, with 4 "
        "coordinates [xmin, ymin, xmax, ymax]. "
        "Bracket the objects tightly: the 'source' box is the card or piece "
        "to grab (if moving a sequence, the TOP card of the sequence); the "
        "'destination' box is the destination (the TOP card of the "
        "destination pile, or the empty destination square/placeholder). "
        "# Response format\n"
        "Reply with a list of dictionaries in this standardized format, in "
        "a BOXES section with exactly two boxes, one per line: "
        'BOXES:\n{"box_2d": [xmin, ymin, xmax, ymax], "label": "source"}\n'
        '{"box_2d": [xmin, ymin, xmax, ymax], "label": "destination"} '
        "Example — the boxes "
        '{"box_2d": [421, 607, 458, 661], "label": "source"} and '
        '{"box_2d": [421, 496, 458, 550], "label": "destination"} '
        "bracket the objects whose pixel centers are (584, 562) and "
        "(584, 464) in a 1329x887 image. "
        "Every coordinate must be normalized between 0 and 1000 (valid "
        "range 0-999: 0 = left/top edge of the image, 999 = right/bottom "
        "edge). "
        "For exact execution, also compute the PIXEL CENTER of each of "
        "the two move endpoints (the exact points where the DRAG should "
        "start and end) and report them in a CENTERS section, in PIXEL "
        "coordinates of this image, in this format: "
        'CENTERS:\nsource: (584, 562)\ndest: (584, 464) '
        "Compute the pixel coordinates carefully — the DRAG will go from "
        "the source center to the dest center, used EXACTLY as written. "
        "If the move needs no DRAG (e.g. only clicking the stock pile), "
        "write 'CENTERS: NONE' and 'BOXES: NONE'. "
        "Then include an ACTION block with any remaining commands — do NOT "
        "write a DRAG in it (the DRAG is computed from the CENTERS; only if "
        "you cannot provide centers, write "
        "'DRAG from (x1, y1) to (x2, y2)' with pixel coordinates as a "
        "fallback): a line reading 'ACTION:' followed by ONE COMMAND PER "
        "LINE, using EXACTLY this syntax — "
        "'CLICK at (x, y)' to click something (a pile, button, or piece); "
        "'DOUBLE_CLICK at (x, y)' to double-click something (e.g. a card to "
        "send it to a foundation); "
        "'WAIT <seconds>' to pause between commands. "
        "If there are no remaining commands, or no move is possible, write "
        "'ACTION:' followed by 'NONE' on the next line. "
        "Write each section (CENTERS, BOXES, ACTION) exactly once — do not "
        "repeat them. "
        "After the CENTERS, BOXES and ACTION sections, leave one blank line. "
        "The LAST sentence of your response must state the best move in one "
        "sentence — nothing may come after it."
    )


def ask_best_move(
    api_key: str,
    image_path: str,
    game_hint: str = None,
    history: list[str] = None,
    want_actions: bool = False,
    prompt: str = None,
) -> str:
    """
    Send a game screenshot to Mistral Large 4 and get the best move advice.
    Anti-hallucination prompt: model must only use visible cards.

    prompt: custom base prompt text (e.g. from the app's prompt library).
        If None, the built-in BASE_PROMPT is used.
    want_actions: append the ACTION-block output contract so move_parser can
        execute the move automatically. The move is still the LAST sentence.
    """
    text = prompt if prompt is not None else BASE_PROMPT
    text += _actions_suffix(image_path) if want_actions else ADVICE_SUFFIX
    if history:
        text = "Previous moves:\n" + "\n".join(f"  {h}" for h in history) + "\n\n" + text

    print(f"Trying {MODEL}...", file=sys.stderr)
    return _call_api(api_key, image_path, text, MODEL)


# ── Prompt library (prompts.json) ──────────────────────────────────

def default_prompts() -> list[dict]:
    """Built-in prompts used to seed prompts.json on first run."""
    return [
        {"title": "Klondike Solitaire (default)", "text": BASE_PROMPT},
        {"title": "Chess (default)", "text": CHESS_PROMPT},
    ]


def load_prompts(path: str = PROMPTS_FILE) -> tuple[list[dict], str | None]:
    """
    Load the prompt library from disk.
    Returns (prompts, active_title); falls back to defaults if missing/corrupt.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        prompts = data.get("prompts", [])
        if prompts and all(
            isinstance(p, dict) and "title" in p and "text" in p for p in prompts
        ):
            return prompts, data.get("active")
    except (OSError, json.JSONDecodeError, AttributeError):
        pass
    prompts = default_prompts()
    return prompts, prompts[0]["title"]


def save_prompts(prompts: list[dict], active: str = None,
                 path: str = PROMPTS_FILE) -> None:
    """Persist the prompt library (and the active title) to disk."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"prompts": prompts, "active": active}, f,
                  indent=2, ensure_ascii=False)


def ensure_prompts_file(path: str = PROMPTS_FILE) -> tuple[list[dict], str | None]:
    """Load the prompt library, creating prompts.json with defaults on first run."""
    if not os.path.exists(path):
        prompts = default_prompts()
        save_prompts(prompts, prompts[0]["title"], path)
        return prompts, prompts[0]["title"]
    return load_prompts(path)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 mistral_client.py <image_path> [api_key]")
        sys.exit(1)
    key = sys.argv[2] if len(sys.argv) > 2 else os.environ.get('MISTRAL_API_KEY', '')
    if not key:
        print("Error: Set MISTRAL_API_KEY")
        sys.exit(1)
    print(ask_best_move(key, sys.argv[1]))
