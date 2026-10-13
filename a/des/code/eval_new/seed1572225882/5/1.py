from collections import Counter
import random

import numpy as np


_DIRECTIONS = ("up", "right", "down", "left")
_ROTATIONS = {"up": 0, "right": -1, "down": 2, "left": 1}


def _background(grid):
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _components(grid, bg):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) != bg:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            rows = [p[0] for p in cells]
            cols = [p[1] for p in cells]
            comps.append(
                {
                    "cells": cells,
                    "set": set(cells),
                    "bbox": (min(rows), max(rows), min(cols), max(cols)),
                }
            )
    return comps


def _perimeter(r0, r1, c0, c1):
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _frame_info(grid, comp, bg):
    colors = {int(grid[r, c]) for r, c in comp["cells"]}
    if len(colors) != 1:
        return None
    color = next(iter(colors))
    r0, r1, c0, c1 = comp["bbox"]
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None

    border = _perimeter(r0, r1, c0, c1)
    cells = comp["set"]
    if any(cell not in border for cell in cells):
        return None
    missing = [cell for cell in border if cell not in cells]
    if len(missing) != 1:
        return None
    mr, mc = missing[0]
    if int(grid[mr, mc]) != bg or len(cells) != len(border) - 1:
        return None
    if np.any(grid[r0 + 1 : r1, c0 + 1 : c1] != bg):
        return None

    if mr == r0 and c0 < mc < c1:
        gap = "up"
    elif mc == c1 and r0 < mr < r1:
        gap = "right"
    elif mr == r1 and c0 < mc < c1:
        gap = "down"
    elif mc == c0 and r0 < mr < r1:
        gap = "left"
    else:
        return None

    return {"color": color, "bbox": comp["bbox"], "gap": gap, "component": comp}


def _sprite_info(grid, comp, bg):
    h, w = grid.shape
    cells = comp["set"]
    boundary_colors = []
    for r, c in comp["cells"]:
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if not (0 <= nr < h and 0 <= nc < w) or (nr, nc) not in cells:
                boundary_colors.append(int(grid[r, c]))
                break
    if not boundary_colors:
        return None

    outline = Counter(boundary_colors).most_common(1)[0][0]
    r0, r1, c0, c1 = comp["bbox"]
    image = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=grid.dtype)
    for r, c in comp["cells"]:
        image[r - r0, c - c0] = grid[r, c]
    return {"color": outline, "bbox": comp["bbox"], "image": image, "component": comp}


def _bbox_distance(a, b):
    ar0, ar1, ac0, ac1 = a
    br0, br1, bc0, bc1 = b
    return abs((ar0 + ar1) - (br0 + br1)) + abs((ac0 + ac1) - (bc0 + bc1))


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate each color-matched sprite toward the gap in its frame and place it inside.

    Input structure:
    The background is the most common color. Non-background components consist of
    one-cell-thick monochrome rectangular frames, each with one missing side cell,
    and separate small multicolor sprites whose outer contour color matches a
    frame color. Frame interiors are empty background.

    Rule:
    Pair each frame with the unused sprite whose outline color matches the frame
    color. The frame gap gives the desired marked-side direction: top keeps the
    sprite unchanged, right rotates it 90 degrees clockwise, bottom rotates it
    180 degrees, and left rotates it 90 degrees counterclockwise. Erase the
    original outside sprite and paste the rotated sprite centered in the frame
    interior.

    Key insight:
    The frame color identifies the source sprite, while the missing border cell
    independently encodes the sprite rotation.
    """
    bg = _background(grid)
    components = _components(grid, bg)
    frames = []
    sprites = []
    for comp in components:
        frame = _frame_info(grid, comp, bg)
        if frame is None:
            sprite = _sprite_info(grid, comp, bg)
            if sprite is not None:
                sprites.append(sprite)
        else:
            frames.append(frame)

    out = grid.copy()
    for sprite in sprites:
        for r, c in sprite["component"]["cells"]:
            out[r, c] = bg

    used = set()
    frames.sort(key=lambda item: item["bbox"])
    for frame in frames:
        matches = [
            i
            for i, sprite in enumerate(sprites)
            if i not in used and sprite["color"] == frame["color"]
        ]
        if not matches:
            continue
        index = min(matches, key=lambda i: _bbox_distance(frame["bbox"], sprites[i]["bbox"]))
        used.add(index)
        sprite = sprites[index]
        rotated = np.rot90(sprite["image"], _ROTATIONS[frame["gap"]])

        r0, r1, c0, c1 = frame["bbox"]
        inner_h = r1 - r0 - 1
        inner_w = c1 - c0 - 1
        sh, sw = rotated.shape
        if sh > inner_h or sw > inner_w:
            continue
        top = r0 + 1 + (inner_h - sh) // 2
        left = c0 + 1 + (inner_w - sw) // 2
        mask = rotated != bg
        target = out[top : top + sh, left : left + sw]
        target[mask] = rotated[mask]
    return out


def _make_sprite(outline, colors, height, width):
    marker, body, accent = colors
    sprite = np.full((height, width), outline, dtype=int)
    sprite[1 : height - 1, 1 : width - 1] = body
    sprite[1, 1 : width - 1] = marker
    if height > 4 and width > 4:
        sprite[height - 2, width // 2] = accent
        sprite[height // 2, 1] = accent
    return sprite


def _place_rect(occupied, height, width, rng):
    rows, cols = occupied.shape
    if height + 2 > rows or width + 2 > cols:
        return None
    for _ in range(300):
        r = rng.randint(1, rows - height - 1)
        c = rng.randint(1, cols - width - 1)
        if not occupied[r : r + height, c : c + width].any():
            occupied[max(0, r - 1) : min(rows, r + height + 1), max(0, c - 1) : min(cols, c + width + 1)] = True
            return r, c
    return None


def _draw_frame(grid, top, left, height, width, color, gap, rng):
    bottom = top + height - 1
    right = left + width - 1
    grid[top, left : right + 1] = color
    grid[bottom, left : right + 1] = color
    grid[top : bottom + 1, left] = color
    grid[top : bottom + 1, right] = color

    if gap == "up":
        grid[top, rng.randint(left + 1, right - 1)] = 0
    elif gap == "right":
        grid[rng.randint(top + 1, bottom - 1), right] = 0
    elif gap == "down":
        grid[bottom, rng.randint(left + 1, right - 1)] = 0
    else:
        grid[rng.randint(top + 1, bottom - 1), left] = 0


def _attempt_generate(rng):
    rows = rng.randint(22, 30)
    cols = rng.randint(22, 30)
    max_objects = 3 if rows * cols >= 650 else 2
    count = rng.randint(1, max_objects)
    outlines = rng.sample(range(1, 10), count)
    occupied = np.zeros((rows, cols), dtype=bool)
    objects = []

    for outline in outlines:
        sprite_h = rng.randint(4, 6)
        sprite_w = rng.randint(4, 6)
        gap = rng.choice(_DIRECTIONS)
        rotated_h, rotated_w = (sprite_h, sprite_w) if gap in ("up", "down") else (sprite_w, sprite_h)
        frame_h = rotated_h + rng.randint(4, 7)
        frame_w = rotated_w + rng.randint(4, 7)

        frame_pos = _place_rect(occupied, frame_h, frame_w, rng)
        if frame_pos is None:
            return None
        sprite_pos = _place_rect(occupied, sprite_h, sprite_w, rng)
        if sprite_pos is None:
            return None

        inner_palette = [color for color in range(1, 10) if color != outline]
        colors = rng.sample(inner_palette, 3)
        sprite = _make_sprite(outline, colors, sprite_h, sprite_w)
        objects.append((outline, gap, frame_pos, frame_h, frame_w, sprite_pos, sprite))

    grid = np.zeros((rows, cols), dtype=int)
    for outline, gap, frame_pos, frame_h, frame_w, sprite_pos, sprite in objects:
        fr, fc = frame_pos
        sr, sc = sprite_pos
        _draw_frame(grid, fr, fc, frame_h, frame_w, outline, gap, rng)
        grid[sr : sr + sprite.shape[0], sc : sc + sprite.shape[1]] = sprite
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(200):
        grid = _attempt_generate(rng)
        if grid is not None:
            return grid

    grid = np.zeros((24, 24), dtype=int)
    sprite = _make_sprite(2, [3, 4, 5], 5, 4)
    _draw_frame(grid, 2, 2, 10, 10, 2, rng.choice(_DIRECTIONS), rng)
    grid[15 : 15 + sprite.shape[0], 15 : 15 + sprite.shape[1]] = sprite
    return grid
