#!/usr/bin/env python3
"""
Move parser — extracts ACTION commands from Mistral's response and executes them.

Parses structured commands like:
    DRAG from (x1, y1) to (x2, y2)
    CLICK at (x, y)
    DOUBLE_CLICK at (x, y)
    KEY <key_name>
    TYPE <text>
    SCROLL <up|down> <clicks> at (x, y)
    WAIT <seconds>
"""

import copy
import re
import time
import sys
import os
from dataclasses import dataclass
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from input_controller import X11Input


@dataclass
class Action:
    """A single parsed action."""
    type: str          # drag, click, double_click, key, type, scroll, wait
    x1: Optional[int] = None
    y1: Optional[int] = None
    x2: Optional[int] = None
    y2: Optional[int] = None
    key: Optional[str] = None
    text: Optional[str] = None
    direction: Optional[str] = None
    clicks: Optional[int] = None
    seconds: Optional[float] = None
    raw: str = ""

    def describe(self) -> str:
        if self.type == 'drag':
            return f"DRAG ({self.x1},{self.y1}) -> ({self.x2},{self.y2})"
        elif self.type == 'click':
            return f"CLICK ({self.x1},{self.y1})"
        elif self.type == 'double_click':
            return f"DOUBLE_CLICK ({self.x1},{self.y1})"
        elif self.type == 'key':
            return f"KEY {self.key}"
        elif self.type == 'type':
            return f"TYPE '{self.text}'"
        elif self.type == 'scroll':
            return f"SCROLL {self.direction} x{self.clicks} at ({self.x1},{self.y1})"
        elif self.type == 'wait':
            return f"WAIT {self.seconds}s"
        return self.raw


def offset_actions(actions: list[Action], dx: int, dy: int) -> list[Action]:
    """
    Return a copy of the actions with all coordinates shifted by (dx, dy).
    The input actions are not modified — safe to keep and re-translate later.
    """
    out = []
    for a in actions:
        b = copy.copy(a)
        if b.x1 is not None:
            b.x1 += dx
        if b.y1 is not None:
            b.y1 += dy
        if b.x2 is not None:
            b.x2 += dx
        if b.y2 is not None:
            b.y2 += dy
        out.append(b)
    return out


# ── Bounding boxes (Mistral cookbook format) ───────────────────────
#
# The model returns normalized bounding boxes (all coordinates between 0
# and 1000), e.g.:
#   BOXES:
#   {"box_2d": [xmin, ymin, xmax, ymax], "label": "source"}
#   {"box_2d": [xmin, ymin, xmax, ymax], "label": "destination"}
# Pixel conversion: x_px = x / 1000 * image_width (same for y).

BOX_OBJ = re.compile(
    r'\{\s*["\']?box_2d["\']?\s*:\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*'
    r'(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]'
    r'\s*,\s*["\']?label["\']?\s*:\s*["\']?(\w+)',
    re.IGNORECASE
)
BOX_LINE = re.compile(
    r'(source|dest(?:ination)?)\s*:\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*'
    r'(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]',
    re.IGNORECASE
)


def extract_boxes(response: str) -> dict | None:
    """
    Extract the source/destination bounding boxes (normalized 0-1000) from
    the model's BOXES section. Accepts the cookbook dict format and, as a
    fallback, simple 'source: [x1, y1, x2, y2]' lines.
    Returns {'source': (x1,y1,x2,y2), 'dest': (x1,y1,x2,y2)} or None.
    """
    if re.search(r'BOXES\s*:\s*NONE', response, re.IGNORECASE):
        return None
    boxes = {}
    for m in BOX_OBJ.finditer(response):
        label = m.group(5).lower()
        box = tuple(float(m.group(i)) for i in range(1, 5))
        if label.startswith('source'):
            boxes['source'] = box
        elif label.startswith('dest'):
            boxes['dest'] = box
    if 'source' in boxes and 'dest' in boxes:
        return boxes
    for m in BOX_LINE.finditer(response):
        label = m.group(1).lower()
        box = tuple(float(m.group(i)) for i in range(2, 6))
        if label.startswith('source'):
            boxes['source'] = box
        elif label.startswith('dest'):
            boxes['dest'] = box
    if 'source' in boxes and 'dest' in boxes:
        return boxes
    return None


def boxes_to_drag(boxes: dict, img_w: int, img_h: int) -> Action:
    """
    Build a DRAG action from normalized (0-1000) bounding boxes: start at the
    center of the source box, end at the center of the destination box.
    """
    def center(box, w, h):
        x1, y1, x2, y2 = (max(0.0, min(1000.0, v)) for v in box)
        return round((x1 + x2) / 2 / 1000 * w), round((y1 + y2) / 2 / 1000 * h)

    x1, y1 = center(boxes['source'], img_w, img_h)
    x2, y2 = center(boxes['dest'], img_w, img_h)
    return Action(
        type='drag', x1=x1, y1=y1, x2=x2, y2=y2,
        raw=f"BOXES source={boxes['source']} dest={boxes['dest']}",
    )


def parse_actions_with_boxes(response: str, img_w: int = None,
                             img_h: int = None) -> list[Action]:
    """
    Parse actions from the response. When a BOXES section is present, the
    DRAG is derived from it (box centers computed exactly in pixels) and
    placed first; remaining ACTION commands (CLICK, DOUBLE_CLICK, WAIT, ...)
    follow. Falls back to the plain ACTION block when no usable boxes exist.
    """
    parsed = parse_actions(response)
    boxes = extract_boxes(response)
    if boxes and img_w and img_h:
        drag = boxes_to_drag(boxes, img_w, img_h)
        others = [a for a in parsed if a.type != 'drag']
        return [drag] + others
    return parsed


CENTER_LINE = re.compile(
    r'(source|dest(?:ination)?)\b[^:\n]{0,15}:\s*\(\s*(-?\d+(?:\.\d+)?)\s*[, ]\s*'
    r'(-?\d+(?:\.\d+)?)\s*\)',
    re.IGNORECASE
)


def extract_centers(response: str) -> dict | None:
    """
    Extract the CENTERS section: the move endpoints as pixel centers.
    Returns {'source': (x, y), 'dest': (x, y)} (pixels) or None.
    """
    if re.search(r'CENTERS\s*:\s*NONE', response, re.IGNORECASE):
        return None
    centers = {}
    for m in CENTER_LINE.finditer(response):
        label = m.group(1).lower()
        point = (float(m.group(2)), float(m.group(3)))
        if label.startswith('source'):
            centers['source'] = point
        elif label.startswith('dest'):
            centers['dest'] = point
    if 'source' in centers and 'dest' in centers:
        return centers
    return None


def parse_move_actions(response: str, img_w: int = None,
                       img_h: int = None) -> list[Action]:
    """
    Parse the move actions from the response.
    The DRAG endpoints come from the CENTERS section (pixel centers, used
    exactly as written) when present; otherwise from the BOXES section
    (normalized 0-1000 boxes, centers computed in pixels). Remaining ACTION
    commands (CLICK, DOUBLE_CLICK, WAIT, ...) follow the DRAG. Falls back to
    the plain ACTION block when neither section is usable.
    """
    parsed = parse_actions(response)
    centers = extract_centers(response)
    if centers:
        drag = Action(
            type='drag',
            x1=round(centers['source'][0]), y1=round(centers['source'][1]),
            x2=round(centers['dest'][0]), y2=round(centers['dest'][1]),
            raw=f"CENTERS source={centers['source']} dest={centers['dest']}",
        )
        others = [a for a in parsed if a.type != 'drag']
        return [drag] + others
    return parse_actions_with_boxes(response, img_w, img_h)


def drag_to_clicks(actions: list[Action]) -> list[Action]:
    """
    Convert the DRAG action into CLICK source + CLICK destination, keeping
    any other commands. For apps that only support click-to-select /
    click-to-place instead of drag-and-drop.
    """
    out = []
    converted = False
    for a in actions:
        if a.type == 'drag' and not converted and a.x1 is not None:
            out.append(Action(type='click', x1=a.x1, y1=a.y1, raw=a.raw))
            out.append(Action(type='click', x1=a.x2, y1=a.y2, raw=a.raw))
            converted = True
        else:
            out.append(a)
    return out


# Regex patterns for each action type (lenient — handles common variations)
PATTERNS = {
    'drag': re.compile(
        r'(?:DRAG|MOVE|SLIDE)\s+(?:from\s*)?\(?\s*(\d+)\s*[, ]\s*(\d+)\s*\)?\s*(?:to|->|→)\s*\(?\s*(\d+)\s*[, ]\s*(\d+)\s*\)?',
        re.IGNORECASE
    ),
    'click': re.compile(
        r'(?<!DOUBLE[\s_])CLICK(?:\s+(?:at|on|on\s+the))?\s*\(?\s*(\d+)\s*[, ]\s*(\d+)\s*\)?',
        re.IGNORECASE
    ),
    'double_click': re.compile(
        r'DOUBLE[\s_]?CLICK(?:\s+(?:at|on))?\s*\(?\s*(\d+)\s*[, ]\s*(\d+)\s*\)?',
        re.IGNORECASE
    ),
    'key': re.compile(
        r'(?:KEY|PRESS)\s+(?:key\s+)?["\']?(\w+)["\']?',
        re.IGNORECASE
    ),
    'type': re.compile(
        r'TYPE\s+["\'](.+?)["\']',
        re.IGNORECASE
    ),
    'scroll': re.compile(
        r'SCROLL\s+(up|down)\s+(\d+)\s+(?:at\s*)?\(?\s*(\d+)\s*[, ]\s*(\d+)\s*\)?',
        re.IGNORECASE
    ),
    'wait': re.compile(
        r'WAIT\s+(\d+(?:\.\d+)?)',
        re.IGNORECASE
    ),
}


def extract_action_block(response: str) -> str:
    """
    Extract the ACTION: block from the AI response.
    Returns the text between 'ACTION:' and the next section or end.
    """
    # Find ACTION: section
    match = re.search(r'ACTION:\s*\n(.*?)(?:\n\n|\n[A-Z_]+:|$)', response, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # Fallback: look for any action-like lines in the whole response
    return response


def extract_move_description(response: str) -> str:
    """
    Extract the MOVE: one-sentence description from the AI response.
    Returns empty string if not found.
    """
    match = re.search(r'MOVE:\s*(.+)', response, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""


def parse_actions(response: str) -> list[Action]:
    """
    Parse all ACTION commands from the AI response.
    Returns a list of Action objects in order.
    """
    action_text = extract_action_block(response)
    actions = []

    # Split into lines and parse each
    lines = action_text.strip().split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue

        action = None

        # Try each pattern (order matters: double_click before click)
        m = PATTERNS['double_click'].search(line)
        if m:
            action = Action(
                type='double_click',
                x1=int(m.group(1)), y1=int(m.group(2)),
                raw=line
            )
        else:
            m = PATTERNS['drag'].search(line)
            if m:
                action = Action(
                    type='drag',
                    x1=int(m.group(1)), y1=int(m.group(2)),
                    x2=int(m.group(3)), y2=int(m.group(4)),
                    raw=line
                )
            else:
                m = PATTERNS['click'].search(line)
                if m:
                    action = Action(
                        type='click',
                        x1=int(m.group(1)), y1=int(m.group(2)),
                        raw=line
                    )
                else:
                    m = PATTERNS['scroll'].search(line)
                    if m:
                        action = Action(
                            type='scroll',
                            direction=m.group(1).lower(),
                            clicks=int(m.group(2)),
                            x1=int(m.group(3)), y1=int(m.group(4)),
                            raw=line
                        )
                    else:
                        m = PATTERNS['key'].search(line)
                        if m:
                            action = Action(
                                type='key',
                                key=m.group(1),
                                raw=line
                            )
                        else:
                            m = PATTERNS['type'].search(line)
                            if m:
                                action = Action(
                                    type='type',
                                    text=m.group(1),
                                    raw=line
                                )
                            else:
                                m = PATTERNS['wait'].search(line)
                                if m:
                                    action = Action(
                                        type='wait',
                                        seconds=float(m.group(1)),
                                        raw=line
                                    )

        if action:
            actions.append(action)

    return actions


def execute_actions(actions: list[Action], dry_run: bool = False,
                    inp: X11Input = None) -> list[str]:
    """
    Execute a list of actions using X11 input injection.
    Returns a list of execution log messages.
    """
    log = []
    own_input = inp is None
    if own_input and not dry_run:
        inp = X11Input()

    try:
        for action in actions:
            desc = action.describe()
            if dry_run:
                log.append(f"[DRY RUN] {desc}")
                continue

            log.append(f"Executing: {desc}")

            if action.type == 'drag':
                inp.drag(action.x1, action.y1, action.x2, action.y2)

            elif action.type == 'click':
                inp.click(action.x1, action.y1)

            elif action.type == 'double_click':
                inp.click(action.x1, action.y1, double=True)

            elif action.type == 'key':
                inp.key(action.key)

            elif action.type == 'type':
                inp.type_text(action.text)

            elif action.type == 'scroll':
                inp.move(action.x1, action.y1)
                time.sleep(0.1)
                inp.scroll(action.clicks, action.direction)

            elif action.type == 'wait':
                time.sleep(action.seconds)

            # Small pause between actions
            time.sleep(0.15)

    finally:
        if own_input and inp is not None:
            inp.close()

    return log


def parse_and_execute(response: str, dry_run: bool = False) -> tuple[list[Action], list[str]]:
    """
    Convenience: parse actions from AI response and execute them.
    Returns (actions, execution_log).
    """
    actions = parse_actions(response)
    log = execute_actions(actions, dry_run=dry_run)
    return actions, log


if __name__ == '__main__':
    # Test parser with a sample response
    sample = """
GAME: Klondike Solitaire (draw-1)
CONFIDENCE: high

BOARD STATE:
Tableau col 1: [K♠ face-down, Q♥ face-up]
...

BEST MOVE:
Move the 7♦ from tableau column 3 to tableau column 5

WHY:
This unlocks a face-down card in column 3.

ACTION:
DRAG from (450, 320) to (780, 320)
WAIT 0.5
"""
    actions = parse_actions(sample)
    for a in actions:
        print(a.describe())
