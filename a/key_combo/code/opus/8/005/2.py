from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GRAY = 5

LEFT_WIDTH = 4
RIGHT_WIDTH = 10
RIGHT_COLORS = (BLUE, RED, GREEN, YELLOW)

_GLYPH_POINTS = ((0, 1), (1, 3), (2, 0), (3, 2))
GLYPH_BLOCKS: dict[int, np.ndarray] = {}
for value in range(1, 5):
    block = np.full((4, 4), GRAY, dtype=int)
    for r, c in _GLYPH_POINTS[:value]:
        block[r, c] = BLACK
    GLYPH_BLOCKS[value] = block


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _decode_roll_value(grid: np.ndarray) -> int:
    left = np.asarray(grid[:, :LEFT_WIDTH], dtype=int)
    gray_coords = np.argwhere(left == GRAY)
    assert gray_coords.size > 0, "expected a gray glyph block in the left zone"
    r0, c0, r1, c1 = _bbox(gray_coords)
    assert r1 - r0 == 3 and c1 - c0 == 3, "glyph block must be 4x4"
    block = left[r0 : r1 + 1, c0 : c1 + 1]

    for value, glyph in GLYPH_BLOCKS.items():
        if np.array_equal(block, glyph):
            return int(value)

    black_count = int(np.sum(block == BLACK))
    assert 1 <= black_count <= 4, "glyph must encode a value from 1 to 4"
    return black_count


def _full_rows(grid: np.ndarray, color: int) -> list[int]:
    return [int(r) for r in range(grid.shape[0]) if np.all(grid[r] == int(color))]


def _visible_border_rows(grid: np.ndarray, color: int) -> list[int]:
    rows = set(np.where((grid[:, 0] == int(color)) | (grid[:, -1] == int(color)))[0].tolist())
    return sorted(int(r) for r in rows)


def _color_interval(grid: np.ndarray, color: int) -> dict | None:
    rows_full = _full_rows(grid, color)
    if len(rows_full) != 2:
        return None
    top, bottom = rows_full
    border_rows = _visible_border_rows(grid, color)
    if not border_rows:
        return None
    complete = border_rows[0] >= top and border_rows[-1] <= bottom
    return {
        "color": int(color),
        "top": int(top),
        "bottom": int(bottom),
        "complete": bool(complete),
    }


def _recolor_complete_intervals(grid: np.ndarray) -> np.ndarray:
    colors = [int(c) for c in np.unique(grid) if int(c) != BLACK]
    intervals = []
    for color in colors:
        interval = _color_interval(grid, color)
        if interval is not None:
            intervals.append(interval)

    complete = [item for item in intervals if item["complete"]]
    out = grid.copy()
    for item in complete:
        containers = [
            parent
            for parent in complete
            if parent["top"] < item["top"] and item["bottom"] < parent["bottom"]
        ]
        if not containers:
            continue
        outermost = min(containers, key=lambda parent: (parent["top"], -parent["bottom"]))
        out[out == item["color"]] = int(outermost["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the left glyph, roll the nested right-side frames downward, and recolor complete inner frames by their new outermost complete container.

    Input structure:
    - A black background grid is split into a 4-column left zone and a 10-column right zone.
    - The left zone contains one 4x4 gray glyph block with fixed black cutouts that encode a value from 1 to 4.
    - The right zone contains 3 or 4 distinct-colored nested rectangular frame intervals that all touch the left and right edges of that zone.

    Rule:
    - Decode the 4x4 glyph to obtain a downward cyclic roll amount r.
    - Remove the left zone and cyclically roll the right zone down by r rows.
    - After the roll, treat only colors whose visible border rows still lie between their two full-width rows as complete frames; wrapped colors are fragments and keep their color.
    - Recolor each complete non-outermost frame to the color of the outermost complete frame that strictly contains it, leaving outermost complete frames and fragments unchanged.

    Key insight:
    - The glyph controls how far the right panel is rolled, and that roll decides which nested frame intervals stay complete versus wrap into fragments, which in turn changes the containment-based recoloring source.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape[1] >= LEFT_WIDTH + 3, "expected a left glyph zone and a right panel"

    roll = _decode_roll_value(g)
    right = g[:, LEFT_WIDTH:].copy()
    rolled = np.roll(right, roll, axis=0)
    return _recolor_complete_intervals(rolled)


def _draw_interval_panel(height: int, intervals: list[tuple[int, int, int]]) -> np.ndarray:
    panel = np.zeros((int(height), RIGHT_WIDTH), dtype=int)
    for color, top, bottom in intervals:
        panel[top, :] = int(color)
        panel[bottom, :] = int(color)
        panel[top : bottom + 1, 0] = int(color)
        panel[top : bottom + 1, RIGHT_WIDTH - 1] = int(color)
    return panel


def _choose_configuration(
    roll: int,
    count: int,
    recolor_mode: bool,
) -> tuple[list[int], list[int]] | None:
    if count == 3:
        tops = [0, 2, 4]
        if recolor_mode:
            bottom_margins = [0, roll, roll + 2]
        else:
            if roll < 2:
                return None
            bottom_margins = [0, 1, roll]
    else:
        tops = [0, 1, 3, 5]
        if recolor_mode:
            if random.random() < 0.5:
                bottom_margins = [0, roll, roll + 1, roll + 2]
            else:
                if roll < 2:
                    return None
                bottom_margins = [0, 1, roll, roll + 2]
        else:
            if roll < 3:
                return None
            bottom_margins = [0, 1, 2, roll]
    return tops, bottom_margins


def _make_intervals(height: int, roll: int, count: int, recolor_mode: bool) -> list[tuple[int, int, int]] | None:
    config = _choose_configuration(roll, count, recolor_mode)
    if config is None:
        return None
    tops, bottom_margins = config
    if len(tops) != count or len(bottom_margins) != count:
        return None
    if bottom_margins[-1] >= height - 2:
        return None

    colors = random.sample(RIGHT_COLORS, count)
    intervals = []
    previous_bottom = height
    for color, top, margin in zip(colors, tops, bottom_margins):
        bottom = height - 1 - int(margin)
        if not (0 <= top < bottom < height):
            return None
        if bottom >= previous_bottom:
            return None
        if bottom - top + 1 < 3:
            return None
        intervals.append((int(color), int(top), int(bottom)))
        previous_bottom = bottom
    return intervals


def _embed_glyph(height: int, roll: int) -> np.ndarray:
    left = np.zeros((int(height), LEFT_WIDTH), dtype=int)
    start_row = random.randint(0, height - 4)
    left[start_row : start_row + 4, :] = GLYPH_BLOCKS[int(roll)]
    return left


def _attempt_generate() -> np.ndarray | None:
    height = random.randint(14, 18)
    roll = random.randint(1, 4)
    count = random.choice([3, 4])

    mode_choices = [True]
    if (count == 3 and roll >= 2) or (count == 4 and roll >= 2):
        mode_choices.append(False)
    recolor_mode = random.choice(mode_choices)

    intervals = _make_intervals(height, roll, count, recolor_mode)
    if intervals is None:
        return None

    panel = _draw_interval_panel(height, intervals)
    grid = np.hstack([_embed_glyph(height, roll), panel])

    rolled = np.roll(panel, roll, axis=0)
    output = main(grid)
    if output.shape != (height, RIGHT_WIDTH):
        return None

    if recolor_mode:
        if np.array_equal(output, rolled):
            return None
    else:
        if not np.array_equal(output, rolled):
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
