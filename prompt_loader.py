#!/usr/bin/env python3
"""
Prompt loader — single source of truth for all LLM prompts.

All prompts live in prompts.json. This module loads them and provides
typed getters so that mistral_client.py, move_detector.py, and gui.py
never hardcode prompt text.

JSON structure:
    {
      "advice_suffix": "...",
      "prompts": [
        {"title": "...", "game": "solitaire", "text": "..."},
        ...
      ],
      "active": "Klondike Solitaire (default)",
      "detector": {
        "system": {
          "chess": "...{game}...",
          "solitaire": "...",
          "checkers": "...",
          "minesweeper": "..."
        },
        "user_template": {
          "chess": "...{game}...{move}...{size}...",
          ...
        }
      }
    }

Template placeholders:
    {game}  — game name (e.g. "chess")
    {move}  — move text (e.g. "Nf3") — detector user templates only
    {size}  — image dimensions string — detector user templates only

Substitution uses str.replace() (not str.format()) because the detector
system prompts contain literal JSON curly braces like {"box_2d": ...}.
"""

import json
import os

PROMPTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts.json")


# ── Fallback defaults (only used if prompts.json is missing/corrupt) ──

_FALLBACK_ADVICE_SUFFIX = (
    "Think step-by-step about the position, but your final output must be "
    "ONLY the MOVE: line — no other text."
)

_FALLBACK_ADVICE = {
    "solitaire": (
        "This is a Klondike Solitaire game. Respond with EXACTLY ONE LINE "
        "starting with 'MOVE: ' followed by the best legal move."
    ),
    "chess": (
        "This is a game of CHESS. Respond with EXACTLY ONE LINE starting "
        "with 'MOVE: ' followed by the best move in algebraic notation."
    ),
    "checkers": (
        "This is a game of CHECKERS. Respond with EXACTLY ONE LINE starting "
        "with 'MOVE: ' followed by the best legal move."
    ),
    "minesweeper": (
        "This is a game of MINESWEEPER. Respond with EXACTLY ONE LINE "
        "starting with 'MOVE: ' followed by the best action."
    ),
}

_FALLBACK_DETECTOR_SYSTEM = {
    "chess": (
        "You are an object-detection vision model. The game is {game}. "
        "Locate the source and destination squares for the given move. "
        'End with: {{"box_2d": [xmin, ymin, xmax, ymax], "label": "source"}} '
        '{{"box_2d": [xmin, ymin, xmax, ymax], "label": "destination"}} '
        "from: <square> to: <square>"
    ),
    "solitaire": (
        "You are an object-detection vision model. The game is {game}. "
        "Locate the source card/pile and destination. "
        'End with: {{"box_2d": [xmin, ymin, xmax, ymax], "label": "source"}} '
        '{{"box_2d": [xmin, ymin, xmax, ymax], "label": "destination"}} '
        "move: <description>"
    ),
    "checkers": (
        "You are an object-detection vision model. The game is {game}. "
        "Locate the source and destination squares for the given move. "
        'End with: {{"box_2d": [xmin, ymin, xmax, ymax], "label": "source"}} '
        '{{"box_2d": [xmin, ymin, xmax, ymax], "label": "destination"}} '
        "from: <square> to: <square>"
    ),
    "minesweeper": (
        "You are an object-detection vision model. The game is {game}. "
        "Locate the single cell to click. Use the same box for source and "
        "destination. "
        'End with: {{"box_2d": [xmin, ymin, xmax, ymax], "label": "source"}} '
        '{{"box_2d": [xmin, ymin, xmax, ymax], "label": "destination"}} '
        "move: <description>"
    ),
}

_FALLBACK_DETECTOR_USER = {
    "chess": (
        'The image shows a game of {game}.{size} '
        'The player wants to play the move "{move}". '
        "Detect the source and destination squares."
    ),
    "solitaire": (
        'The image shows a game of {game}.{size} '
        'The player wants to play: "{move}". '
        "Detect the source object and the destination object."
    ),
    "checkers": (
        'The image shows a game of {game}.{size} '
        'The player wants to play: "{move}". '
        "Detect the source and destination squares."
    ),
    "minesweeper": (
        'The image shows a game of {game}.{size} '
        'The player wants to: "{move}". '
        "Locate the single cell to click."
    ),
}


# ── Game key normalisation ──────────────────────────────────────────

def _game_key(game: str) -> str:
    """Normalise a game name/hint to a canonical key."""
    g = (game or "").lower()
    if "checkers" in g or "draughts" in g:
        return "checkers"
    if "minesweeper" in g:
        return "minesweeper"
    if "chess" in g:
        return "chess"
    if "solitaire" in g or "klondike" in g:
        return "solitaire"
    return "solitaire"  # default


# ── JSON load / save ────────────────────────────────────────────────

def _load_json() -> dict:
    """Load the full prompts.json. Returns {} if missing/corrupt."""
    try:
        with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _save_json(data: dict) -> None:
    """Write the full prompts.json."""
    with open(PROMPTS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _build_default_json() -> dict:
    """Build a minimal default prompts.json structure."""
    titles = {
        "solitaire": "Klondike Solitaire (default)",
        "chess": "Chess (default)",
        "checkers": "Checkers (default)",
        "minesweeper": "Minesweeper (default)",
    }
    prompts = [
        {"title": titles[g], "game": g, "text": _FALLBACK_ADVICE[g]}
        for g in ("solitaire", "chess", "checkers", "minesweeper")
    ]
    return {
        "advice_suffix": _FALLBACK_ADVICE_SUFFIX,
        "prompts": prompts,
        "active": titles["solitaire"],
        "detector": {
            "system": dict(_FALLBACK_DETECTOR_SYSTEM),
            "user_template": dict(_FALLBACK_DETECTOR_USER),
        },
    }


def ensure_prompts_file() -> tuple[list[dict], str | None]:
    """Create prompts.json with defaults on first run.
    Returns (prompts_list, active_title)."""
    data = _load_json()
    if not data or "prompts" not in data:
        data = _build_default_json()
        _save_json(data)
    return data.get("prompts", []), data.get("active")


# ── Advice prompts (step 1: what move to play) ──────────────────────

def get_advice_suffix() -> str:
    """Return the suffix appended to every advice prompt."""
    data = _load_json()
    return data.get("advice_suffix") or _FALLBACK_ADVICE_SUFFIX


def get_advice_prompt(game: str = "solitaire") -> str:
    """Return the advice prompt text for a game.

    Looks up the prompt by its 'game' field in prompts.json.
    Falls back to a minimal default if not found.
    """
    key = _game_key(game)
    data = _load_json()
    for p in data.get("prompts", []):
        if p.get("game") == key:
            return p.get("text", "")
    return _FALLBACK_ADVICE.get(key, _FALLBACK_ADVICE["solitaire"])


# ── Detector prompts (step 2: where to move on screen) ─────────────

def get_detector_system_prompt(game: str = "chess") -> str:
    """Return the move-detector system prompt for a game.

    The template contains a {game} placeholder which is replaced with
    the actual game name before returning.
    """
    key = _game_key(game)
    data = _load_json()
    template = (
        data.get("detector", {}).get("system", {}).get(key)
        or _FALLBACK_DETECTOR_SYSTEM.get(key)
        or _FALLBACK_DETECTOR_SYSTEM["chess"]
    )
    return template.replace("{game}", game)


def get_detector_user_template(game: str = "chess") -> str:
    """Return the move-detector user prompt template for a game.

    The returned template contains {game}, {move}, and {size}
    placeholders. Callers must substitute {move} and {size} themselves.
    """
    key = _game_key(game)
    data = _load_json()
    template = (
        data.get("detector", {}).get("user_template", {}).get(key)
        or _FALLBACK_DETECTOR_USER.get(key)
        or _FALLBACK_DETECTOR_USER["chess"]
    )
    return template


def build_detector_user_prompt(move: str, game: str = "chess",
                               img_w: int | None = None,
                               img_h: int | None = None) -> str:
    """Build the full detector user prompt by substituting all placeholders."""
    template = get_detector_user_template(game)
    size = f" The screenshot is {img_w}x{img_h} pixels." if img_w and img_h else ""
    return (template
            .replace("{game}", game)
            .replace("{move}", move)
            .replace("{size}", size))


# ── GUI prompt library ──────────────────────────────────────────────

def load_prompts() -> tuple[list[dict], str | None]:
    """Return (prompts_list, active_title) for the GUI prompt library."""
    return ensure_prompts_file()


def save_prompts(prompts: list[dict], active: str = None) -> None:
    """Save the GUI prompt library (prompts array + active title).

    Preserves the detector section and advice_suffix — only the
    'prompts' array and 'active' key are updated.
    """
    data = _load_json()
    if not data:
        data = _build_default_json()
    data["prompts"] = prompts
    if active is not None:
        data["active"] = active
    _save_json(data)


def default_prompts() -> list[dict]:
    """Return the built-in default prompts (for seeding / reference)."""
    titles = {
        "solitaire": "Klondike Solitaire (default)",
        "chess": "Chess (default)",
        "checkers": "Checkers (default)",
        "minesweeper": "Minesweeper (default)",
    }
    return [
        {"title": titles[g], "game": g, "text": _FALLBACK_ADVICE[g]}
        for g in ("solitaire", "chess", "checkers", "minesweeper")
    ]
