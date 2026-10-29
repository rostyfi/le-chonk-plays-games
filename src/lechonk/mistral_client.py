#!/usr/bin/env python3
"""
Minimal Mistral API client — sends a screenshot, gets move advice.
Model: mistral-large-4, temperature: 0, reasoning_effort: low.

All prompts are loaded from prompts.json via prompt_loader — no prompt
text is hardcoded here.
"""

import base64
import json
import os
import sys
import urllib.request
import urllib.error

from .move_parser import extract_move_text
from .prompt_loader import get_advice_suffix

MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"
MODEL = "mistral-large-4"

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
              reasoning: str = "low") -> str:
    """Send image + prompt to Mistral API. Returns response text.

    reasoning: "low" (default, light chain-of-thought), "none" (fastest),
    or "max" (extended reasoning, higher token budget).
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
        "reasoning_effort": "high" if is_max else reasoning,
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
    prompt: str,
    history: list[str] = None,
    reasoning: str = "low",
) -> str:
    """
    Send a screenshot to Mistral Large 4 and get the best move advice.

    prompt: the advice prompt text (from the prompt library in prompts.json).
    reasoning: "low" (default, light chain-of-thought), "none" (fast),
    or "max" (extended chain-of-thought).
    """
    text = prompt + get_advice_suffix()
    if history:
        text = "Previous moves:\n" + "\n".join(f"  {h}" for h in history) + "\n\n" + text

    print(f"Trying {MODEL} (reasoning={reasoning})...", file=sys.stderr)
    return _call_api(api_key, image_path, text, MODEL, reasoning=reasoning)


def ask_best_move_parsed(
    api_key: str,
    image_path: str,
    prompt: str,
    behavior: dict = None,
    history: list[str] = None,
    reasoning: str = "low",
) -> tuple[str, str]:
    """
    Ask Mistral for the best move and extract the move text from the response.

    Combines ask_best_move() with move_parser.extract_move_text() so callers
    get both the raw AI response and the parsed move in one call.

    prompt:   the advice prompt text (from the prompt library).
    behavior: behaviour config dict (from the prompt's 'behavior' field).
              Controls how the move text is extracted (e.g. algebraic notation).
    reasoning: "low" (default, light chain-of-thought), "none" (fast),
    or "max" (extended chain-of-thought).

    Returns (raw_response, move_text). move_text is empty string if no move
    could be extracted.
    """
    response = ask_best_move(
        api_key, image_path,
        prompt=prompt, history=history,
        reasoning=reasoning,
    )
    move_text = extract_move_text(response, behavior=behavior)
    return response, move_text


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 mistral_client.py <image_path> [api_key]")
        sys.exit(1)
    key = sys.argv[2] if len(sys.argv) > 2 else os.environ.get('MISTRAL_API_KEY', '')
    if not key:
        print("Error: Set MISTRAL_API_KEY")
        sys.exit(1)
    print(ask_best_move(key, sys.argv[1]))
