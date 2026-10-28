#!/usr/bin/env python3
"""
Prompt loader — single source of truth for all LLM prompts.

All prompts live in prompts.json. This module loads them and provides
typed getters so that callers never hardcode prompt text or game names.

JSON structure:
    {
      "advice_suffix": "...",
      "prompts": [
        {
          "title": "...",
          "game": "...",          ← key into detector.system / detector.user_template
          "text": "...",           ← advice prompt sent to the model
          "behavior": {            ← game-agnostic behaviour config
            "action": "drag",      ← "drag" or "click"
            "grab_y": 0.8,         ← grab point as fraction of source box height
            "board_grid": true,    ← detect a regular grid and compute exact boxes
            "validate_move": true, ← validate reported source→dest geometry
            "move_format": "algebraic", ← "algebraic" (SAN-like), "squares", or null
            "right_click_word": "flag"  ← word in move text that means right-click
          }
        },
        ...
      ],
      "active": "...",
      "detector": {
        "system": { "<game>": "...{game}...", ... },
        "user_template": { "<game>": "...{game}...{move}...{size}...", ... }
      }
    }

Template placeholders:
    {game}  — game name (from the prompt's 'game' field)
    {move}  — move text — detector user templates only
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

_FALLBACK_ADVICE_TEXT = (
    "You are playing a game. Look at the screenshot and respond with "
    "EXACTLY ONE LINE starting with 'MOVE: ' followed by the best action."
)

_FALLBACK_DETECTOR_SYSTEM = (
    "You are an object-detection vision model. The game is {game}. "
    "Locate the source and destination for the given move. "
    'End with: {{"box_2d": [xmin, ymin, xmax, ymax], "label": "source"}} '
    '{{"box_2d": [xmin, ymin, xmax, ymax], "label": "destination"}} '
    "move: <description>"
)

_FALLBACK_DETECTOR_USER = (
    'The image shows a game of {game}.{size} '
    'The player wants to play: "{move}". '
    "Detect the source and destination. Report their bounding boxes "
    "(labels 'source' and 'destination') and a short description."
)


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
    """Build a minimal default prompts.json structure (game-agnostic)."""
    return {
        "advice_suffix": _FALLBACK_ADVICE_SUFFIX,
        "prompts": [
            {
                "title": "Default",
                "game": "default",
                "text": _FALLBACK_ADVICE_TEXT,
                "behavior": {"action": "drag"},
            }
        ],
        "active": "Default",
        "detector": {
            "system": {"default": _FALLBACK_DETECTOR_SYSTEM},
            "user_template": {"default": _FALLBACK_DETECTOR_USER},
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


# ── Prompt lookup by title ──────────────────────────────────────────

def get_prompt_by_title(title: str) -> dict | None:
    """Return the full prompt dict (title, game, text, behavior) for a title."""
    data = _load_json()
    for p in data.get("prompts", []):
        if p.get("title") == title:
            return p
    return None


def get_active_prompt() -> dict | None:
    """Return the full prompt dict for the active prompt."""
    data = _load_json()
    active = data.get("active")
    if active:
        return get_prompt_by_title(active)
    prompts = data.get("prompts", [])
    return prompts[0] if prompts else None


def get_behavior(prompt: dict) -> dict:
    """Return the behaviour config dict from a prompt (empty dict if none)."""
    return prompt.get("behavior", {}) if prompt else {}


# ── Detector prompts (step 2: where to move on screen) ─────────────

def get_detector_system(game_key: str) -> str:
    """Return the move-detector system prompt for a game key.

    Looks up detector.system[game_key] in prompts.json. The template
    contains a {game} placeholder which is replaced with the game key.
    """
    data = _load_json()
    template = (
        data.get("detector", {}).get("system", {}).get(game_key)
        or _FALLBACK_DETECTOR_SYSTEM
    )
    return template.replace("{game}", game_key)


def get_detector_user_template(game_key: str) -> str:
    """Return the move-detector user prompt template for a game key.

    The returned template contains {game}, {move}, and {size}
    placeholders. Callers must substitute {move} and {size} themselves.
    """
    data = _load_json()
    return (
        data.get("detector", {}).get("user_template", {}).get(game_key)
        or _FALLBACK_DETECTOR_USER
    )


def build_detector_user_prompt(move: str, game_key: str,
                               img_w: int | None = None,
                               img_h: int | None = None) -> str:
    """Build the full detector user prompt by substituting all placeholders."""
    template = get_detector_user_template(game_key)
    size = f" The screenshot is {img_w}x{img_h} pixels." if img_w and img_h else ""
    return (template
            .replace("{game}", game_key)
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
