from __future__ import annotations

import random

import numpy as np


BG = 0
FRAME = 5
ARROW = 9
MARK_COLORS = (1, 2, 3, 4, 6, 7, 8)
DIRS = ("up", "down", "left", "right")
GEN_RETRIES = 300
PLACE_RETRIES = 120


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_frames(grid: np.ndarray, bg: int) -> list[tuple[int, int, int, int]]:
    frames: list[tuple[int, int, int, int]] = []
    for comp in _components(grid != bg):
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        if not any(int(grid[r, c]) == FRAME for r, c in comp):
            continue
        if not all(r in (r0, r1) or c in (c0, c1) for r, c in comp):
            continue
        ok = True
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                on_perimeter = r in (r0, r1) or c in (c0, c1)
                if on_perimeter and int(grid[r, c]) == bg:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            frames.append((int(r0), int(c0), int(r1), int(c1)))
    frames.sort()
    return frames


def _arrow_direction(
    grid: np.ndarray, bbox: tuple[int, int, int, int]
) -> tuple[str, list[tuple[int, int]]]:
    r0, c0, r1, c1 = bbox
    coords = [
        (r, c)
        for r in range(r0 + 1, r1)
        for c in range(c0 + 1, c1)
        if int(grid[r, c]) == ARROW
    ]
    assert len(coords) == 3, "each frame must contain one 3-pixel L arrow"

    ar0, ac0, ar1, ac1 = _bbox(coords)
    assert ar1 - ar0 == 1 and ac1 - ac0 == 1, "arrow must occupy a 2x2 box"
    local = {(r - ar0, c - ac0) for r, c in coords}
    all_cells = {(0, 0), (0, 1), (1, 0), (1, 1)}
    missing = list(all_cells - local)
    assert len(missing) == 1, "arrow must miss exactly one corner"

    direction = {
        (0, 1): "right",
        (0, 0): "up",
        (1, 0): "left",
        (1, 1): "down",
    }[missing[0]]
    return direction, coords


def _edge_sequence(
    grid: np.ndarray, bbox: tuple[int, int, int, int], direction: str, bg: int
) -> list[int]:
    r0, c0, r1, c1 = bbox
    if direction == "up":
        line = [int(grid[r0, c]) for c in range(c0, c1 + 1)]
    elif direction == "down":
        line = [int(grid[r1, c]) for c in range(c0, c1 + 1)]
    elif direction == "left":
        line = [int(grid[r, c0]) for r in range(r0, r1 + 1)]
    else:
        line = [int(grid[r, c1]) for r in range(r0, r1 + 1)]
    return [v for v in line if v not in (bg, FRAME)]


def _write_positions(bbox: tuple[int, int, int, int], direction: str) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    center_r = r0 + 1 + inner_h // 2
    center_c = c0 + 1 + inner_w // 2
    if direction == "up":
        return [(r0 + 1 + i, center_c) for i in range(inner_h)]
    if direction == "down":
        return [(r1 - 1 - i, center_c) for i in range(inner_h)]
    if direction == "left":
        return [(center_r, c0 + 1 + i) for i in range(inner_w)]
    return [(center_r, c1 - 1 - i) for i in range(inner_w)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the color sequence from the border edge indicated by the internal L-arrow
    and write that sequence through the frame's center, starting from the pointed side.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = int(np.bincount(g.reshape(-1)).argmax())

    out = g.copy()
    for bbox in _find_frames(g, bg):
        direction, arrow_cells = _arrow_direction(g, bbox)
        sequence = _edge_sequence(g, bbox, direction, bg)
        for r, c in arrow_cells:
            out[r, c] = bg
        for color, (r, c) in zip(sequence, _write_positions(bbox, direction), strict=False):
            out[r, c] = int(color)
    return out


def _paint_frame(grid: np.ndarray, r0: int, c0: int, inner_h: int, inner_w: int) -> tuple[int, int]:
    r1 = r0 + inner_h + 1
    c1 = c0 + inner_w + 1
    grid[r0, c0 : c1 + 1] = FRAME
    grid[r1, c0 : c1 + 1] = FRAME
    grid[r0 : r1 + 1, c0] = FRAME
    grid[r0 : r1 + 1, c1] = FRAME
    return r1, c1


def _side_marks(length: int, max_count: int) -> tuple[list[int], list[int]]:
    count = random.randint(1, min(length, max_count))
    offsets = sorted(random.sample(range(1, length + 1), count))
    colors = [random.choice(MARK_COLORS) for _ in range(count)]
    return offsets, colors


def _place_marks(
    grid: np.ndarray, bbox: tuple[int, int, int, int]
) -> dict[str, list[int]]:
    r0, c0, r1, c1 = bbox
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    max_count = min(inner_h, inner_w)
    sequences: dict[str, list[int]] = {}

    offsets, colors = _side_marks(inner_w, max_count)
    for off, color in zip(offsets, colors, strict=True):
        grid[r0, c0 + off] = color
    sequences["up"] = colors[:]

    offsets, colors = _side_marks(inner_w, max_count)
    for off, color in zip(offsets, colors, strict=True):
        grid[r1, c0 + off] = color
    sequences["down"] = colors[:]

    offsets, colors = _side_marks(inner_h, max_count)
    for off, color in zip(offsets, colors, strict=True):
        grid[r0 + off, c0] = color
    sequences["left"] = colors[:]

    offsets, colors = _side_marks(inner_h, max_count)
    for off, color in zip(offsets, colors, strict=True):
        grid[r0 + off, c1] = color
    sequences["right"] = colors[:]

    return sequences


def _preferred_arrow_positions(
    bbox: tuple[int, int, int, int], direction: str
) -> tuple[list[int], list[int]]:
    r0, c0, r1, c1 = bbox
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    center_r = r0 + 1 + inner_h // 2
    center_c = c0 + 1 + inner_w // 2

    row_options = list(range(r0 + 2, r1 - 2))
    col_options = list(range(c0 + 2, c1 - 2))
    assert row_options and col_options

    if direction == "up":
        pref_rows = [r for r in row_options if r <= center_r]
        pref_cols = col_options
    elif direction == "down":
        pref_rows = [r for r in row_options if r >= center_r - 1]
        pref_cols = col_options
    elif direction == "left":
        pref_rows = row_options
        pref_cols = [c for c in col_options if c <= center_c]
    else:
        pref_rows = row_options
        pref_cols = [c for c in col_options if c >= center_c - 1]

    if direction in ("up", "down") and random.random() < 0.5:
        overlap_cols = [c for c in (center_c - 1, center_c) if c in col_options]
        if overlap_cols:
            pref_cols = overlap_cols
    if direction in ("left", "right") and random.random() < 0.5:
        overlap_rows = [r for r in (center_r - 1, center_r) if r in row_options]
        if overlap_rows:
            pref_rows = overlap_rows

    return pref_rows or row_options, pref_cols or col_options


def _place_arrow(
    grid: np.ndarray, bbox: tuple[int, int, int, int], direction: str
) -> list[tuple[int, int]]:
    rows, cols = _preferred_arrow_positions(bbox, direction)
    ar = random.choice(rows)
    ac = random.choice(cols)
    missing = {
        "right": (0, 1),
        "up": (0, 0),
        "left": (1, 0),
        "down": (1, 1),
    }[direction]
    cells: list[tuple[int, int]] = []
    for dr in range(2):
        for dc in range(2):
            if (dr, dc) == missing:
                continue
            rr = ar + dr
            cc = ac + dc
            grid[rr, cc] = ARROW
            cells.append((int(rr), int(cc)))
    return cells


def _frame_spec(
    grid: np.ndarray, bbox: tuple[int, int, int, int], direction: str
) -> dict[str, object]:
    sequences = _place_marks(grid, bbox)
    arrow_cells = _place_arrow(grid, bbox, direction)
    return {
        "bbox": bbox,
        "direction": direction,
        "sequence": sequences[direction],
        "arrow_cells": arrow_cells,
    }


def _expected_output(grid: np.ndarray, specs: list[dict[str, object]]) -> np.ndarray:
    out = grid.copy()
    for spec in specs:
        bbox = spec["bbox"]
        direction = str(spec["direction"])
        sequence = [int(v) for v in spec["sequence"]]
        arrow_cells = spec["arrow_cells"]
        for r, c in arrow_cells:
            out[r, c] = BG
        for color, (r, c) in zip(sequence, _write_positions(bbox, direction), strict=False):
            out[r, c] = int(color)
    return out


def _random_inner_size(n_frames: int) -> tuple[int, int]:
    if n_frames == 1:
        choices = (5, 7, 9, 11)
    elif n_frames == 2:
        choices = (5, 7, 9)
    elif n_frames == 3:
        choices = (5, 7, 9)
    else:
        choices = (5, 7)
    return random.choice(choices), random.choice(choices)


def _try_place_bbox(
    occupied: np.ndarray, outer_h: int, outer_w: int
) -> tuple[int, int] | None:
    h, w = occupied.shape
    if outer_h + 2 > h or outer_w + 2 > w:
        return None
    for _ in range(PLACE_RETRIES):
        r0 = random.randint(1, h - outer_h - 1)
        c0 = random.randint(1, w - outer_w - 1)
        r1 = r0 + outer_h - 1
        c1 = c0 + outer_w - 1
        rs = slice(max(0, r0 - 1), min(h, r1 + 2))
        cs = slice(max(0, c0 - 1), min(w, c1 + 2))
        if not bool(occupied[rs, cs].any()):
            occupied[rs, cs] = True
            return int(r0), int(c0)
    return None


def _attempt() -> np.ndarray | None:
    n_frames = random.randint(1, 4)
    min_side = 14 + 2 * n_frames
    height = random.randint(min_side, 30)
    width = random.randint(min_side, 30)
    grid = np.full((height, width), BG, dtype=int)
    occupied = np.zeros((height, width), dtype=bool)
    specs: list[dict[str, object]] = []

    for _ in range(n_frames):
        inner_h, inner_w = _random_inner_size(n_frames)
        placement = _try_place_bbox(occupied, inner_h + 2, inner_w + 2)
        if placement is None:
            return None
        r0, c0 = placement
        r1, c1 = _paint_frame(grid, r0, c0, inner_h, inner_w)
        direction = random.choice(DIRS)
        specs.append(_frame_spec(grid, (r0, c0, r1, c1), direction))

    try:
        actual = main(grid)
    except Exception:
        return None
    expected = _expected_output(grid, specs)
    if not np.array_equal(actual, expected):
        return None
    if len(_find_frames(grid, BG)) != n_frames:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
