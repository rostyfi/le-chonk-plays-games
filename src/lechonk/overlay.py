#!/usr/bin/env python3
"""
Move overlay — draw the planned move on a screenshot.

Renders the source/destination bounding boxes, a drag arrow, and click
markers, so the planned move can be seen on the captured image before it
is executed (and in dry-run mode). Action coordinates are image-relative
pixels; boxes are normalized 0-1000 (Mistral cookbook format).
"""

import math

SRC_COLOR = (0, 220, 0)      # green — source object
DST_COLOR = (255, 80, 0)     # orange — destination object
MOVE_COLOR = (255, 235, 0)   # yellow — drag arrow
CLICK_COLOR = (0, 200, 255)  # cyan — click targets


def _boxes_match_endpoints(drag, boxes: dict, w: int, h: int,
                           tol: int = 40) -> bool:
    """Do the normalized boxes agree with the drag endpoints (in pixels)?
    The model sometimes emits boxes in pixel coordinates instead of
    normalized ones — those must not be drawn or scaled as normalized."""
    for key, (px, py) in (('source', (drag.x1, drag.y1)),
                          ('dest', (drag.x2, drag.y2))):
        if key not in boxes:
            return False
        bx1, by1, bx2, by2 = boxes[key]
        cx = (bx1 + bx2) / 2 / 1000 * w
        cy = (by1 + by2) / 2 / 1000 * h
        if abs(cx - px) > tol or abs(cy - py) > tol:
            return False
    return True


def _arrow(draw, x1, y1, x2, y2, color, width):
    """Line from (x1, y1) to (x2, y2) with an arrowhead at the end."""
    draw.line([(x1, y1), (x2, y2)], fill=color, width=width)
    angle = math.atan2(y2 - y1, x2 - x1)
    head = 18
    for da in (math.radians(150), math.radians(-150)):
        draw.line(
            [(x2, y2),
             (x2 + head * math.cos(angle + da),
              y2 + head * math.sin(angle + da))],
            fill=color, width=width,
        )


def _marker(draw, x, y, color, radius=12, double=False):
    """Circle + crosshair marker for a click target."""
    draw.ellipse([x - radius, y - radius, x + radius, y + radius],
                 outline=color, width=3)
    if double:
        r = radius + 7
        draw.ellipse([x - r, y - r, x + r, y + r], outline=color, width=2)
    draw.line([(x - radius - 5, y), (x + radius + 5, y)], fill=color, width=2)
    draw.line([(x, y - radius - 5), (x, y + radius + 5)], fill=color, width=2)


def move_changed_board(before_path: str, after_path: str, actions: list,
                       boxes: dict | None = None, min_changed: int = 50) -> bool:
    """
    Did the board change where the move happened? Compares the regions
    around the move's source and destination (the bounding boxes if given,
    else squares around the drag endpoints) between two screenshots.
    Regions outside the move (clocks, status text, ...) are ignored, so a
    ticking clock does not count as a change.
    """
    from PIL import Image, ImageChops

    img_a = Image.open(before_path).convert('RGB')
    img_b = Image.open(after_path).convert('RGB')
    if img_a.size != img_b.size:
        return True
    w, h = img_a.size

    regions = []
    drags = [a for a in actions if a.type == 'drag' and a.x1 is not None]
    # Use the model's boxes only when they agree with the drag endpoints;
    # pixel-drifted boxes would point the comparison at the wrong regions.
    boxes_ok = bool(boxes) and (not drags
                                or _boxes_match_endpoints(drags[0], boxes, w, h))
    if boxes_ok:
        for name in ('source', 'dest'):
            if name in boxes:
                x1, y1, x2, y2 = boxes[name]
                regions.append((x1 / 1000 * w, y1 / 1000 * h,
                                x2 / 1000 * w, y2 / 1000 * h))
    if not regions:
        for a in drags:
            regions.append((a.x1 - 40, a.y1 - 40, a.x1 + 40, a.y1 + 40))
            regions.append((a.x2 - 40, a.y2 - 40, a.x2 + 40, a.y2 + 40))
    if not regions:
        # nowhere specific to look — fall back to the whole image
        diff = ImageChops.difference(img_a, img_b)
        return diff.getbbox() is not None

    pad = 10
    changed = 0
    total = 0
    for rx1, ry1, rx2, ry2 in regions:
        rx1 = max(0, int(rx1) - pad)
        ry1 = max(0, int(ry1) - pad)
        rx2 = min(w, int(rx2) + pad)
        ry2 = min(h, int(ry2) + pad)
        if rx2 <= rx1 or ry2 <= ry1:
            continue
        total += (rx2 - rx1) * (ry2 - ry1)
        crop_a = img_a.crop((rx1, ry1, rx2, ry2))
        crop_b = img_b.crop((rx1, ry1, rx2, ry2))
        diff = ImageChops.difference(crop_a, crop_b)
        mask = diff.point(lambda v: 255 if v > 8 else 0)
        hist = mask.convert('L').histogram()
        changed += sum(hist[1:])  # any channel changed by more than 8

    return changed > max(min_changed, 0.02 * total)


def draw_move_overlay(image_path: str, output_path: str,
                      actions: list, boxes: dict | None = None) -> str:
    """
    Draw the planned move on the screenshot and save it to output_path.

    actions: parsed Action list (image-relative pixel coordinates).
    boxes: {'source': (x1,y1,x2,y2), 'dest': (x1,y1,x2,y2)} with coordinates
           normalized 0-1000; drawn as rectangles around the objects.
    Returns output_path.
    """
    from PIL import Image, ImageDraw

    img = Image.open(image_path).convert('RGB')
    w, h = img.size
    draw = ImageDraw.Draw(img)
    lw = max(2, w // 400)

    drag = next((a for a in actions
                 if a.type == 'drag' and a.x1 is not None), None)
    # Draw the model's boxes only when they agree with the drag endpoints;
    # the model sometimes emits boxes in pixel coordinates, which would be
    # scaled wrongly and drawn in the wrong place.
    boxes_ok = bool(boxes) and (drag is None
                                or _boxes_match_endpoints(drag, boxes, w, h))
    if boxes_ok:
        for name, color in (('source', SRC_COLOR), ('dest', DST_COLOR)):
            if name in boxes:
                bx1, by1, bx2, by2 = boxes[name]
                rect = (bx1 / 1000 * w, by1 / 1000 * h,
                        bx2 / 1000 * w, by2 / 1000 * h)
                draw.rectangle(rect, outline=color, width=lw + 1)
    elif drag is not None:
        # fixed squares at the drag endpoints so the move is always visible
        half = max(24, w // 50)
        for (px, py), color in (((drag.x1, drag.y1), SRC_COLOR),
                                ((drag.x2, drag.y2), DST_COLOR)):
            draw.rectangle([px - half, py - half, px + half, py + half],
                           outline=color, width=lw + 1)

    for a in actions:
        if a.type == 'drag' and a.x1 is not None:
            _arrow(draw, a.x1, a.y1, a.x2, a.y2, MOVE_COLOR, lw)
            draw.ellipse([a.x1 - 6, a.y1 - 6, a.x1 + 6, a.y1 + 6],
                         fill=MOVE_COLOR)
            draw.ellipse([a.x2 - 6, a.y2 - 6, a.x2 + 6, a.y2 + 6],
                         fill=MOVE_COLOR)
        elif a.type == 'click' and a.x1 is not None:
            _marker(draw, a.x1, a.y1, CLICK_COLOR)
        elif a.type == 'double_click' and a.x1 is not None:
            _marker(draw, a.x1, a.y1, CLICK_COLOR, double=True)

    img.save(output_path)
    return output_path
