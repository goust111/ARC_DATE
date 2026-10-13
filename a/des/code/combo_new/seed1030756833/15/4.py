from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
WALL = 1
LABEL = 5
ON = 9
MARKER_COLORS = (2, 3, 4, 6, 7, 8)


def _is_blue_rectangle(g: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> bool:
    if r1 - r0 < 2 or c1 - c0 < 2:
        return False
    if not np.all(g[r0, c0 : c1 + 1] == WALL):
        return False
    if not np.all(g[r1, c0 : c1 + 1] == WALL):
        return False
    if not np.all(g[r0 : r1 + 1, c0] == WALL):
        return False
    if not np.all(g[r0 : r1 + 1, c1] == WALL):
        return False
    return not bool(np.any(g[r0 + 1 : r1, c0 + 1 : c1] == WALL))


def _blue_rectangles(g: np.ndarray) -> list[tuple[int, int, int, int]]:
    h, w = g.shape
    rows = [r for r in range(h) if np.any(g[r] == WALL)]
    cols = [c for c in range(w) if np.any(g[:, c] == WALL)]
    rects: list[tuple[int, int, int, int]] = []
    for i, r0 in enumerate(rows):
        for r1 in rows[i + 1 :]:
            for j, c0 in enumerate(cols):
                for c1 in cols[j + 1 :]:
                    if _is_blue_rectangle(g, r0, c0, r1, c1):
                        rects.append((r0, c0, r1, c1))
    return rects


def _left_label_count(g: np.ndarray, rect: tuple[int, int, int, int]) -> int | None:
    r0, c0, r1, _c1 = rect
    found: list[int] = []
    for c in range(max(0, c0 - 3), c0):
        rows = [r for r in range(r0 + 1, r1) if int(g[r, c]) == LABEL]
        if not rows:
            continue
        band = g[r0 : r1 + 1, c]
        if not np.all((band == BG) | (band == LABEL)):
            continue
        if max(rows) - min(rows) + 1 != len(rows):
            continue
        found.append(len(rows))
    if len(found) == 1:
        return found[0]
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """Tile each walled room with the legend pattern selected by its marker count.

    Input structure:
    - Black background contains one-pixel blue horizontal and vertical walls forming closed rectangular rooms.
    - A corner legend has small blue-framed sample rectangles; each sample has a vertical stack of gray pixels on its left whose length is the marker count key.
    - Sample interiors use white-on-black binary patterns, while real room interiors contain zero or more single-pixel non-blue, non-gray, non-white marker colors.

    Rule:
    - Detect all atomic blue-framed rectangles; those with a left gray count stack and a black/white interior are legend templates.
    - For every non-legend room, count its marker pixels. Rooms with no markers remain black inside.
    - Look up the template with the same marker count, tile it over the room interior, recoloring white pattern cells to the room's majority marker color and leaving black pattern cells black.

    Key insight:
    The gray stack beside each sample is a visible lookup key linking a marker count to a reusable binary pattern, while the room markers provide both the key and the output color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h <= 30 and w <= 30

    rects = _blue_rectangles(g)
    assert rects, "expected at least one blue-framed rectangle"

    templates: dict[int, np.ndarray] = {}
    template_rects: set[tuple[int, int, int, int]] = set()
    for rect in rects:
        key = _left_label_count(g, rect)
        if key is None or key == 0:
            continue
        r0, c0, r1, c1 = rect
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        if not np.all((interior == BG) | (interior == ON)):
            continue
        pattern = interior == ON
        assert np.any(pattern), "legend pattern cannot be empty"
        if key in templates:
            assert np.array_equal(templates[key], pattern), "template keys must be unique"
        templates[key] = pattern
        template_rects.add(rect)

    assert templates, "expected count-keyed legend templates"

    out = g.copy()
    reserved = {BG, WALL, LABEL, ON}
    for rect in rects:
        if rect in template_rects:
            continue
        r0, c0, r1, c1 = rect
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        marker_mask = np.ones(interior.shape, dtype=bool)
        for color in reserved:
            marker_mask &= interior != color
        marker_values = interior[marker_mask]
        marker_count = int(marker_values.size)
        if marker_count == 0:
            continue

        assert marker_count in templates, "room marker count must have a matching template"
        counts = Counter(map(int, marker_values))
        best_count = max(counts.values())
        winners = [color for color, count in counts.items() if count == best_count]
        assert len(winners) == 1, "room marker colors must have a unique majority"
        fill_color = int(winners[0])

        pattern = templates[marker_count]
        ph, pw = pattern.shape
        for rr in range(r1 - r0 - 1):
            for cc in range(c1 - c0 - 1):
                out[r0 + 1 + rr, c0 + 1 + cc] = fill_color if pattern[rr % ph, cc % pw] else BG

    return out


# --------------- generator ---------------

GEN_RETRIES = 300


def _random_pattern(rng: random.Random, ph: int, pw: int, used: set[tuple[int, ...]]) -> np.ndarray:
    for _ in range(100):
        density = rng.uniform(0.30, 0.70)
        vals = [1 if rng.random() < density else 0 for _ in range(ph * pw)]
        count = sum(vals)
        key = tuple(vals)
        if 1 < count < ph * pw and key not in used:
            used.add(key)
            return np.array(vals, dtype=bool).reshape(ph, pw)

    vals = [0] * (ph * pw)
    offset = len(used) % pw
    for r in range(ph):
        vals[r * pw + ((r + offset) % pw)] = 1
    key = tuple(vals)
    used.add(key)
    return np.array(vals, dtype=bool).reshape(ph, pw)


def _draw_frame(grid: np.ndarray, r0: int, c0: int, inner_h: int, inner_w: int) -> tuple[int, int, int, int]:
    r1 = r0 + inner_h + 1
    c1 = c0 + inner_w + 1
    grid[r0, c0 : c1 + 1] = WALL
    grid[r1, c0 : c1 + 1] = WALL
    grid[r0 : r1 + 1, c0] = WALL
    grid[r0 : r1 + 1, c1] = WALL
    return r0, c0, r1, c1


def _draw_template(
    grid: np.ndarray,
    top: int,
    block_left: int,
    key: int,
    pattern: np.ndarray,
) -> tuple[int, int, int, int]:
    ph, pw = pattern.shape
    label_col = block_left
    frame_col = block_left + 2
    grid[top + 1 : top + 1 + key, label_col] = LABEL
    rect = _draw_frame(grid, top, frame_col, ph, pw)
    r0, c0, _r1, _c1 = rect
    interior = grid[r0 + 1 : r0 + 1 + ph, c0 + 1 : c0 + 1 + pw]
    interior[pattern] = ON
    return rect


def _draw_floorplan(
    grid: np.ndarray,
    top: int,
    left: int,
    heights: list[int],
    widths: list[int],
) -> list[tuple[int, int, int, int]]:
    row_lines = [top]
    for room_h in heights:
        row_lines.append(row_lines[-1] + room_h + 1)
    col_lines = [left]
    for room_w in widths:
        col_lines.append(col_lines[-1] + room_w + 1)

    r0, r1 = row_lines[0], row_lines[-1]
    c0, c1 = col_lines[0], col_lines[-1]
    for r in row_lines:
        grid[r, c0 : c1 + 1] = WALL
    for c in col_lines:
        grid[r0 : r1 + 1, c] = WALL

    rooms: list[tuple[int, int, int, int]] = []
    for ri in range(len(heights)):
        for ci in range(len(widths)):
            rooms.append((row_lines[ri], col_lines[ci], row_lines[ri + 1], col_lines[ci + 1]))
    return rooms


def _tile_room(
    out: np.ndarray,
    rect: tuple[int, int, int, int],
    pattern: np.ndarray,
    color: int,
) -> None:
    r0, c0, r1, c1 = rect
    ph, pw = pattern.shape
    for rr in range(r1 - r0 - 1):
        for cc in range(c1 - c0 - 1):
            out[r0 + 1 + rr, c0 + 1 + cc] = color if pattern[rr % ph, cc % pw] else BG


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    max_key = rng.choice([2, 3])
    ph = rng.randint(max(3, max_key), 4)
    pw = rng.randint(3, 4)

    template_span = max_key * (pw + 4) + (max_key - 1)
    min_w = max(22, template_span + 2)
    if min_w > 30:
        return None
    height = rng.randint(20, 30)
    width = rng.randint(min_w, 30)
    grid = np.zeros((height, width), dtype=int)

    used_patterns: set[tuple[int, ...]] = set()
    patterns: dict[int, np.ndarray] = {}
    for key in range(1, max_key + 1):
        pattern = _random_pattern(rng, ph, pw, used_patterns)
        patterns[key] = pattern
        block_left = 1 + (key - 1) * (pw + 5)
        _draw_template(grid, 1, block_left, key, pattern)

    room_shape_choices = [(1, 3), (2, 2), (2, 3)] if max_key == 2 else [(2, 2), (2, 3)]
    display_bottom = 1 + ph + 1
    min_room_top = display_bottom + 2
    floor = None
    for _ in range(120):
        room_rows, room_cols = rng.choice(room_shape_choices)
        heights = [rng.randint(3, 5) for _ in range(room_rows)]
        widths = [rng.randint(3, 6) for _ in range(room_cols)]
        total_h = sum(heights) + room_rows + 1
        total_w = sum(widths) + room_cols + 1
        if total_h > height - min_room_top:
            continue
        if total_w > width - 2:
            continue
        top = rng.randint(min_room_top, height - total_h)
        left = rng.randint(1, width - total_w - 1)
        floor = (top, left, heights, widths)
        break
    if floor is None:
        return None

    top, left, heights, widths = floor
    rooms = _draw_floorplan(grid, top, left, heights, widths)

    keys = [0] + list(range(1, max_key + 1))
    while len(keys) < len(rooms):
        keys.append(rng.randint(0, max_key))
    rng.shuffle(keys)

    expected = grid.copy()
    used_multicolor = False
    for rect, key in zip(rooms, keys):
        if key == 0:
            continue

        r0, c0, r1, c1 = rect
        cells = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
        marker_cells = rng.sample(cells, key)

        colors = list(MARKER_COLORS)
        rng.shuffle(colors)
        main_color = colors[0]
        marker_colors = [main_color] * key
        if key >= 3 and (not used_multicolor or rng.random() < 0.45):
            marker_colors = [main_color, main_color, colors[1]]
            used_multicolor = True
        rng.shuffle(marker_colors)

        for (r, c), color in zip(marker_cells, marker_colors):
            grid[r, c] = color
        _tile_room(expected, rect, patterns[key], main_color)

    try:
        if not np.array_equal(main(grid), expected):
            return None
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
