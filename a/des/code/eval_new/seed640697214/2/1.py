from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
YELLOW = 4
GREY = 5
PALETTE = [1, 2, 3, 6, 7, 8, 9]
GEN_RETRIES = 400


def _components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _colors_present(grid: np.ndarray, exclude: set[int]) -> list[int]:
    colors = sorted(int(v) for v in np.unique(grid) if int(v) not in exclude)
    return colors


def _full_height_yellow_runs(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cols = [c for c in range(c0, c1 + 1) if np.all(grid[r0 : r1 + 1, c] == YELLOW)]
    runs: list[tuple[int, int]] = []
    for c in cols:
        if not runs or c != runs[-1][1] + 1:
            runs.append((c, c))
        else:
            runs[-1] = (runs[-1][0], c)
    return runs


def _find_canvas_cells(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    comps = _components_mask(grid == YELLOW)
    assert comps, "canvas must be marked by yellow lines"
    canvas_pts = max(comps, key=len)
    r0, c0, r1, c1 = _bbox(canvas_pts)
    assert np.all(grid[r0, c0 : c1 + 1] == YELLOW)
    assert np.all(grid[r1, c0 : c1 + 1] == YELLOW)
    runs = _full_height_yellow_runs(grid, (r0, c0, r1, c1))
    assert len(runs) >= 2, "canvas must contain at least one cell"

    cells: list[tuple[int, int, int, int]] = []
    for left, right in zip(runs, runs[1:]):
        cell_c0 = left[1] + 1
        cell_c1 = right[0] - 1
        if cell_c0 <= cell_c1:
            cells.append((r0 + 1, cell_c0, r1 - 1, cell_c1))
    assert cells, "canvas parsing produced no cells"
    return cells


def _parse_key(grid: np.ndarray, color: int, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    width = c1 - c0 + 1
    body = grid[r1, c0 : c1 + 1]
    assert np.all(body == color), "key body must be a solid horizontal bar"
    teeth = np.zeros(width, dtype=bool)
    if r1 > r0:
        teeth = np.any(grid[r0:r1, c0 : c1 + 1] == color, axis=0)
    return teeth


def _parse_lock(
    grid: np.ndarray, color: int, bbox: tuple[int, int, int, int]
) -> dict[str, object]:
    r0, c0, r1, c1 = bbox
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    assert height >= 5 and width >= 6, "lock frame too small"
    assert np.all(grid[r0, c0 : c1 + 1] == color)
    assert np.all(grid[r1, c0 : c1 + 1] == color)
    assert np.all(grid[r0 : r1 + 1, c0] == color)
    assert np.all(grid[r0 : r1 + 1, c1] == color)

    notch_col = c0 + 1
    notches = grid[r0 + 1 : r1, notch_col] == BG
    pattern_mask = grid[r0 + 1 : r1, c0 + 2 : c1] == GREY
    return {
        "color": color,
        "bbox": bbox,
        "notches": notches,
        "pattern_mask": pattern_mask,
        "pattern_origin": (r0 + 1, c0 + 2),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Unlock colored lock frames by matching key teeth to lock notches and releasing the hidden pattern.

    Input structure:
    - Black background.
    - On the left are colored keys: each key is a short horizontal bar whose top edge has 1-cell tooth protrusions.
    - On the right are same-colored lock frames: each lock is a rectangular frame whose inner left wall contains black notch cells, and each lock encloses a grey pattern.
    - Along the bottom is a yellow-bordered canvas strip divided into one empty cell per lock.

    Rule:
    - Pair each key with the lock of the same color.
    - Read the key teeth from left to right and the lock notches from top to bottom; the lock opens only when these boolean sequences match exactly.
    - For every opened lock, recolor its grey interior pattern to the lock color and copy that recolored mask into the canvas cell whose left-to-right position matches the lock's top-to-bottom order.
    - Locks that do not open keep their grey pattern and leave their canvas cell empty.

    Key insight:
    - Color only identifies candidate pairs; the decisive test is the geometric equality between the key's tooth pattern and the lock's notch pattern.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    cells = _find_canvas_cells(g)
    for r0, c0, r1, c1 in cells:
        out[r0 : r1 + 1, c0 : c1 + 1] = BG

    keys_by_color: dict[int, np.ndarray] = {}
    locks: list[dict[str, object]] = []

    for color in _colors_present(g, exclude={BG, YELLOW, GREY}):
        comps = _components_mask(g == color)
        assert len(comps) >= 2, "each visible color should appear in a key and a lock"
        for pts in comps:
            box = _bbox(pts)
            h = box[2] - box[0] + 1
            if h <= 2:
                keys_by_color[color] = _parse_key(g, color, box)
            else:
                locks.append(_parse_lock(g, color, box))

    locks.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))  # type: ignore[index]
    assert len(locks) == len(cells), "lock count and canvas cell count must match"

    for idx, lock in enumerate(locks):
        color = int(lock["color"])
        assert color in keys_by_color, "missing matching key color"
        key_teeth = keys_by_color[color]
        notches = np.asarray(lock["notches"], dtype=bool)
        pattern_mask = np.asarray(lock["pattern_mask"], dtype=bool)
        assert len(key_teeth) == len(notches), "key width must match lock comparison height"

        if not np.array_equal(key_teeth, notches):
            continue

        pr0, pc0 = lock["pattern_origin"]  # type: ignore[misc]
        cr0, cc0, cr1, cc1 = cells[idx]
        assert pattern_mask.shape == (cr1 - cr0 + 1, cc1 - cc0 + 1)

        for rr in range(pattern_mask.shape[0]):
            for cc in range(pattern_mask.shape[1]):
                if not pattern_mask[rr, cc]:
                    continue
                out[pr0 + rr, pc0 + cc] = color
                out[cr0 + rr, cc0 + cc] = color

    return out


def _random_bool_pattern(length: int, rng: random.Random) -> np.ndarray:
    size = rng.randint(1, length - 1)
    picks = rng.sample(range(length), size)
    arr = np.zeros(length, dtype=bool)
    arr[picks] = True
    return arr


def _different_pattern(base: np.ndarray, rng: random.Random) -> np.ndarray:
    while True:
        cand = _random_bool_pattern(len(base), rng)
        if not np.array_equal(cand, base):
            return cand


def _random_blob(height: int, width: int, rng: random.Random) -> np.ndarray:
    area = height * width
    target = rng.randint(2, max(2, area - 1))
    start = (rng.randrange(height), rng.randrange(width))
    cells = {start}
    frontier = {start}

    while len(cells) < target:
        if frontier:
            fr, fc = rng.choice(tuple(frontier))
        else:
            fr, fc = rng.choice(tuple(cells))

        nbrs = []
        for nr, nc in ((fr - 1, fc), (fr + 1, fc), (fr, fc - 1), (fr, fc + 1)):
            if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                nbrs.append((nr, nc))
        if not nbrs:
            frontier.discard((fr, fc))
            if not frontier and len(cells) < target:
                frontier = set(cells)
            continue

        pick = rng.choice(nbrs)
        cells.add(pick)
        frontier.add(pick)

    mask = np.zeros((height, width), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _draw_key(grid: np.ndarray, top: int, left: int, color: int, teeth: np.ndarray) -> None:
    width = len(teeth)
    grid[top + 1, left : left + width] = color
    for idx, bit in enumerate(teeth):
        if bit:
            grid[top, left + idx] = color


def _draw_lock(
    grid: np.ndarray,
    top: int,
    left: int,
    color: int,
    notches: np.ndarray,
    pattern_mask: np.ndarray,
) -> None:
    inner_h, inner_w = pattern_mask.shape
    lock_h = inner_h + 2
    lock_w = inner_w + 3

    grid[top, left : left + lock_w] = color
    grid[top + lock_h - 1, left : left + lock_w] = color
    grid[top : top + lock_h, left] = color
    grid[top : top + lock_h, left + lock_w - 1] = color
    grid[top + 1 : top + lock_h - 1, left + 1] = color

    for idx, bit in enumerate(notches):
        if bit:
            grid[top + 1 + idx, left + 1] = BG

    for r in range(inner_h):
        for c in range(inner_w):
            if pattern_mask[r, c]:
                grid[top + 1 + r, left + 2 + c] = GREY


def _draw_canvas(
    grid: np.ndarray, top: int, left: int, cell_h: int, cell_w: int, count: int
) -> None:
    total_w = count * cell_w + (count + 1)
    total_h = cell_h + 2
    grid[top, left : left + total_w] = YELLOW
    grid[top + total_h - 1, left : left + total_w] = YELLOW
    for x in range(count + 1):
        col = left + x * (cell_w + 1)
        grid[top : top + total_h, col] = YELLOW


def _canvas_cells_from_params(
    top: int, left: int, cell_h: int, cell_w: int, count: int
) -> list[tuple[int, int, int, int]]:
    cells = []
    for idx in range(count):
        c0 = left + 1 + idx * (cell_w + 1)
        c1 = c0 + cell_w - 1
        cells.append((top + 1, c0, top + cell_h, c1))
    return cells


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    count = rng.randint(2, 4)
    max_inner_h = {2: 7, 3: 4, 4: 3}[count]
    inner_h = rng.randint(3, max_inner_h)
    inner_w = rng.randint(3, 5)
    lock_h = inner_h + 2
    lock_w = inner_w + 3
    canvas_h = inner_h + 2
    canvas_w = count * inner_w + (count + 1)
    stack_h = count * lock_h + (count - 1)
    base_h = 1 + stack_h + 1 + canvas_h
    if base_h > 30:
        return None

    height = rng.randint(base_h, 30)
    top_margin = height - stack_h - 1 - canvas_h
    width_min = max(canvas_w, inner_h + lock_w + 5)
    width = rng.randint(width_min, 30)

    grid = np.full((height, width), BG, dtype=int)
    colors = rng.sample(PALETTE, count)

    canvas_top = height - canvas_h
    canvas_left = rng.randint(0, width - canvas_w)
    _draw_canvas(grid, canvas_top, canvas_left, inner_h, inner_w, count)
    canvas_cells = _canvas_cells_from_params(canvas_top, canvas_left, inner_h, inner_w, count)

    key_x = rng.randint(0, min(2, max(0, width - inner_h - 1)))
    lock_x_min = key_x + inner_h + 3
    lock_x_max = width - lock_w
    if lock_x_min > lock_x_max:
        return None
    lock_x = rng.randint(lock_x_min, lock_x_max)

    unlock_count = rng.randint(1, count - 1)
    unlocked = set(rng.sample(range(count), unlock_count))
    lock_rows = [top_margin + idx * (lock_h + 1) for idx in range(count)]
    key_slots = [row + rng.randint(0, lock_h - 2) for row in lock_rows]
    key_rows = rng.sample(key_slots, len(key_slots))

    expected_canvas_fill = []
    for idx, color in enumerate(colors):
        notch_pattern = _random_bool_pattern(inner_h, rng)
        key_pattern = notch_pattern.copy() if idx in unlocked else _different_pattern(notch_pattern, rng)
        pattern_mask = _random_blob(inner_h, inner_w, rng)
        _draw_lock(grid, lock_rows[idx], lock_x, color, notch_pattern, pattern_mask)
        _draw_key(grid, key_rows[idx], key_x, color, key_pattern)
        expected_canvas_fill.append(idx in unlocked and np.any(pattern_mask))

    out = main(grid)
    seen_fill = []
    for cell in canvas_cells:
        r0, c0, r1, c1 = cell
        seen_fill.append(bool(np.any(out[r0 : r1 + 1, c0 : c1 + 1] != BG)))

    if seen_fill != expected_canvas_fill:
        return None
    if not np.any(out != grid):
        return None
    if len(set(seen_fill)) != 2:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid key-lock ARC puzzle")
