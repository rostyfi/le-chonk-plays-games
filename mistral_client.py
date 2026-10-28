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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from move_parser import extract_move_text

MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"
MODEL = "mistral-large-4"

PROMPTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts.json")

BASE_PROMPT = (
    "This is a Klondike Solitaire game (draw-1, standard rules). "
    "Think carefully about the position, then respond with EXACTLY ONE LINE "
    "starting with 'MOVE: ' followed by a short description of the single "
    "best LEGAL move. No analysis in the output — just the MOVE: line. "
    "Klondike rules — follow them EXACTLY: "
    "1. TABLEAU: build DOWN by 1 in ALTERNATING COLORS. "
    "   Red (♥♦) goes on Black (♠♣) and Black goes on Red. "
    "   LEGAL: 9♥ (red) on 10♠ (black). LEGAL: 2♣ (black) on 3♥ (red). "
    "   ILLEGAL: 2♣ (black) on 3♣ (black) — SAME COLOR! "
    "   ILLEGAL: 7♥ (red) on 8♥ (red) — SAME COLOR! "
    "2. FOUNDATIONS: build UP by 1 in the SAME SUIT, starting from Ace. "
    "   Example: A♠ then 2♠ then 3♠ on the spades foundation. "
    "3. EMPTY tableau column: ONLY a King (or sequence starting with King). "
    "Before answering, VERIFY: is the move legal? Does it follow the "
    "alternating color rule? Never suggest an illegal move. "
    "Examples of correct output: "
    "'MOVE: Move 9♥ from column 3 to column 5 (on 10♠)', "
    "'MOVE: Move A♦ to diamonds foundation', "
    "'MOVE: Move K♠ to empty column 4', "
    "'MOVE: Draw from the stock pile'. "
    "If no legal move exists: 'MOVE: Draw from the stock pile'. "
    "If stock is empty and no moves exist: 'MOVE: No moves available'. "
    "Do NOT guess or invent cards you cannot see."
)

ADVICE_SUFFIX = (
    "Think step-by-step about the position, but your final output must be "
    "ONLY the MOVE: line — no other text."
)

CHESS_PROMPT = (
    "This is a game of CHESS. "
    "Think carefully about the position, then respond with EXACTLY ONE LINE "
    "starting with 'MOVE: ' followed by the best move in algebraic notation. "
    "No analysis in the output — just the MOVE: line. "
    "Examples: 'MOVE: Nf3', 'MOVE: e4', 'MOVE: exd5', 'MOVE: O-O'. "
    "If the best move is a pawn promotion, include it: 'MOVE: e8=Q'. "
    "If NO legal move exists, reply: 'MOVE: No moves available'. "
    "Do NOT report pixel coordinates or bounding boxes."
)

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


def _call_api(api_key: str, image_path: str, user_prompt: str, model: str,
              reasoning: str = "none") -> str:
    """Send image + prompt to Mistral API. Returns response text.

    reasoning: "none" (fast, no chain-of-thought) or "max" (extended
    reasoning, higher token budget).
    """
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

    is_max = reasoning == "max"
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
        "max_tokens": 32768 if is_max else 4096,
        "temperature": 0,
        "top_p": 1,
        "reasoning_effort": "high" if is_max else "none",
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

    import time as _time
    t0 = _time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"API error ({e.code}): {e.read().decode('utf-8')[:500]}")
    elapsed = _time.monotonic() - t0

    actual = result.get('model', '?')
    usage = result.get('usage', {})
    prompt_tok = usage.get('prompt_tokens', '?')
    compl_tok = usage.get('completion_tokens', '?')
    print(f"  Served by: {actual} | {elapsed:.1f}s | "
          f"prompt={prompt_tok} completion={compl_tok} tokens", file=sys.stderr)

    choices = result.get('choices', [])
    if not choices:
        raise RuntimeError(f"No choices: {json.dumps(result)[:500]}")

    text = _extract_text(choices[0]['message'])
    finish = choices[0].get('finish_reason', '?')
    if not text.strip():
        if finish == 'length':
            # Model used all tokens on reasoning — retry without reasoning
            print("  [retry] Empty response (length) — retrying without reasoning...", file=sys.stderr)
            payload['reasoning_effort'] = 'none'
            payload['max_tokens'] = 4096
            req = urllib.request.Request(
                MISTRAL_API_URL,
                data=json.dumps(payload).encode('utf-8'),
                headers={
                    'Content-Type': 'application/json',
                    'Authorization': f'Bearer {api_key}',
                },
                method='POST',
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read().decode('utf-8'))
            choices = result.get('choices', [])
            if not choices:
                raise RuntimeError(f"No choices on retry: {json.dumps(result)[:500]}")
            text = _extract_text(choices[0]['message'])
            if not text.strip():
                raise RuntimeError(f"Empty response on retry (finish_reason={choices[0].get('finish_reason', '?')})")
        else:
            raise RuntimeError(f"Empty response (finish_reason={finish})")

    return text


def ask_best_move(
    api_key: str,
    image_path: str,
    game_hint: str = None,
    history: list[str] = None,
    prompt: str = None,
    reasoning: str = "none",
) -> str:
    """
    Send a game screenshot to Mistral Large 4 and get the best move advice.
    Anti-hallucination prompt: model must only use visible cards.

    prompt: custom base prompt text (e.g. from the app's prompt library).
        If None, a game-appropriate default is chosen: CHESS_PROMPT when
        game_hint indicates chess, otherwise BASE_PROMPT (Solitaire).
    game_hint: used to select the default prompt when prompt is None.
    reasoning: "none" (fast) or "max" (extended chain-of-thought).
    """
    if prompt is not None:
        text = prompt
    elif game_hint and "chess" in game_hint.lower():
        text = CHESS_PROMPT
    else:
        text = BASE_PROMPT
    text += ADVICE_SUFFIX
    if history:
        text = "Previous moves:\n" + "\n".join(f"  {h}" for h in history) + "\n\n" + text

    print(f"Trying {MODEL} (reasoning={reasoning})...", file=sys.stderr)
    return _call_api(api_key, image_path, text, MODEL, reasoning=reasoning)


def ask_best_move_parsed(
    api_key: str,
    image_path: str,
    game: str = "chess",
    game_hint: str = None,
    history: list[str] = None,
    prompt: str = None,
    reasoning: str = "none",
) -> tuple[str, str]:
    """
    Ask Mistral for the best move and extract the move text from the response.

    Combines ask_best_move() with move_parser.extract_move_text() so callers
    get both the raw AI response and the parsed move (e.g. "Nf3") in one call.

    reasoning: "none" (fast) or "max" (extended chain-of-thought).

    Returns (raw_response, move_text). move_text is empty string if no move
    could be extracted.
    """
    response = ask_best_move(
        api_key, image_path,
        game_hint=game_hint or game, history=history, prompt=prompt,
        reasoning=reasoning,
    )
    move_text = extract_move_text(response, game=game)
    return response, move_text


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
