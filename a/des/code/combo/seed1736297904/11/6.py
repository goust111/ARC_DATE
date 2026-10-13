from __future__ import annotations

import random

import numpy as np


BG = 0
RED = 2
DIVIDER = 9
PATTERN_COLORS = (1, 3, 4, 5, 6, 7, 8)

GRID_HEIGHT_RANGE = (9, 16)
CANVAS_WIDTH_RANGE = (6, 12)
INSTR_WIDTH_RANGE = (3, 6)
INSTRUCTION_COUNT_RANGE = (1, 4)
MAX_SHAPE_ATTEMPTS = 80
MAX_GENERATION_ATTEMPTS = 300
MIN_STEP_DELTA = 2


def _find_divider(grid: np.ndarray) -> int:
    cols = np.flatnonzero(np.all(grid == DIVIDER, axis=0))
    assert len(cols) == 1, "expected exactly one full-height maroon divider"
    divider = int(cols[0])
    assert 0 < divider < grid.shape[1] - 1, "divider must leave instruction and canvas regions"
    return divider


def _instruction_rows(instr: np.ndarray) -> list[int]:
    assert np.all(np.isin(instr, (BG, RED))), "instruction area must contain only black and red"
    red_pixels = np.argwhere(instr == RED)
    assert len(red_pixels) >= 1, "instruction area must contain at least one red pixel"
    order = np.lexsort((red_pixels[:, 1], red_pixels[:, 0]))
    return [int(red_pixels[idx, 0]) for idx in order]


def _fold_once(canvas: np.ndarray, axis: int) -> np.ndarray:
    h, _w = canvas.shape
    assert 0 <= axis < h, "fold axis must be an in-bounds row"
    above = axis
    below = h - 1 - axis
    assert above != below, "fold axis must have a unique shorter side"

    out = canvas.copy()
    if above < below:
        for dist in range(1, above + 1):
            src = axis - dist
            dst = axis + dist
            src_row = canvas[src]
            mask = (src_row != BG) & (out[dst] == BG)
            out[dst, mask] = src_row[mask]
    else:
        for dist in range(1, below + 1):
            src = axis + dist
            dst = axis - dist
            src_row = canvas[src]
            mask = (src_row != BG) & (out[dst] == BG)
            out[dst, mask] = src_row[mask]
    return out


def _apply_folds(canvas: np.ndarray, axes: list[int]) -> tuple[np.ndarray, list[int]]:
    cur = np.asarray(canvas, dtype=int).copy()
    deltas: list[int] = []
    for axis in axes:
        nxt = _fold_once(cur, axis)
        deltas.append(int(np.count_nonzero(nxt != cur)))
        cur = nxt
    return cur, deltas


def main(grid: np.ndarray) -> np.ndarray:
    """Fold the canvas across instruction-encoded rows and return the final canvas."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    divider = _find_divider(g)
    instr = g[:, :divider]
    canvas = g[:, divider + 1 :]
    assert canvas.shape[1] >= 1, "canvas region must be non-empty"

    axes = _instruction_rows(instr)
    out, _deltas = _apply_folds(canvas, axes)
    return out


def _available_axis_rows(h: int) -> tuple[list[int], list[int]]:
    upper = []
    lower = []
    for r in range(1, h - 1):
        short_side = min(r, h - 1 - r)
        if short_side < 2:
            continue
        if r < h - 1 - r:
            upper.append(r)
        elif r > h - 1 - r:
            lower.append(r)
    return upper, lower


def _sample_axes(h: int) -> list[int]:
    upper, lower = _available_axis_rows(h)
    profiles: list[str] = []
    if upper:
        profiles.append("upper")
    if lower:
        profiles.append("lower")
    if upper and lower:
        profiles.append("mixed")
    assert profiles, "grid height must allow at least one valid fold row"

    profile = random.choice(profiles)
    max_count = min(INSTRUCTION_COUNT_RANGE[1], len(upper) + len(lower))
    min_count = INSTRUCTION_COUNT_RANGE[0]
    if profile == "mixed":
        min_count = max(min_count, 2)
    count = random.randint(min_count, max_count)

    if profile == "upper":
        axes = random.sample(upper, k=min(count, len(upper)))
    elif profile == "lower":
        axes = random.sample(lower, k=min(count, len(lower)))
    else:
        upper_count = random.randint(1, count - 1)
        lower_count = count - upper_count
        if upper_count > len(upper) or lower_count > len(lower):
            return _sample_axes(h)
        axes = random.sample(upper, k=upper_count) + random.sample(lower, k=lower_count)

    axes.sort()
    return axes


def _reflect_rows(rows: np.ndarray, axis: int) -> np.ndarray:
    return 2 * axis - rows


def _shape_cells(
    shape: str,
    row0: int,
    col0: int,
    height: int,
    width: int,
) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    if shape == "hbar":
        for c in range(col0, col0 + width):
            cells.append((row0, c))
    elif shape == "vbar":
        for r in range(row0, row0 + height):
            cells.append((r, col0))
    elif shape == "rect":
        for r in range(row0, row0 + height):
            for c in range(col0, col0 + width):
                cells.append((r, c))
    else:
        for c in range(col0, col0 + width):
            cells.append((row0, c))
        for r in range(row0, row0 + height):
            cells.append((r, col0))
    return sorted(set(cells))


def _try_place_shape(
    canvas: np.ndarray,
    occupied: np.ndarray,
    reserved: np.ndarray,
    axis: int,
    source_above: bool,
    color: int,
) -> bool:
    h, w = canvas.shape
    short_len = min(axis, h - 1 - axis)
    assert short_len >= 2

    shape = random.choice(("hbar", "vbar", "rect", "ell"))
    if shape == "hbar":
        height, width = 1, random.randint(2, min(5, w))
    elif shape == "vbar":
        height, width = random.randint(2, min(4, short_len)), 1
    else:
        height = random.randint(2, min(3, short_len))
        width = random.randint(2, min(4, w))

    if source_above:
        max_row0 = axis - height
        if max_row0 < 0:
            return False
        row0 = random.randint(0, max_row0)
    else:
        min_row0 = axis + 1
        max_row0 = h - height
        if min_row0 > max_row0:
            return False
        row0 = random.randint(min_row0, max_row0)

    col0 = random.randint(0, w - width)
    cells = _shape_cells(shape, row0, col0, height, width)
    rows = np.array([r for r, _c in cells], dtype=int)
    cols = np.array([c for _r, c in cells], dtype=int)
    mirror_rows = _reflect_rows(rows, axis)

    if np.any(mirror_rows < 0) or np.any(mirror_rows >= h):
        return False
    if occupied[rows, cols].any() or reserved[rows, cols].any():
        return False
    if occupied[mirror_rows, cols].any() or reserved[mirror_rows, cols].any():
        return False

    canvas[rows, cols] = color
    occupied[rows, cols] = True
    reserved[mirror_rows, cols] = True
    return True


def _add_axis_motifs(canvas: np.ndarray, axes: list[int], color: int) -> bool:
    occupied = np.zeros_like(canvas, dtype=bool)
    reserved = np.zeros_like(canvas, dtype=bool)

    for axis in axes:
        source_above = axis < canvas.shape[0] - 1 - axis
        motif_count = random.choice((1, 1, 2))
        placed = 0
        for _ in range(MAX_SHAPE_ATTEMPTS):
            if _try_place_shape(canvas, occupied, reserved, axis, source_above, color):
                placed += 1
                if placed >= motif_count:
                    break
        if placed < 1:
            return False
    return True


def _add_extra_shape(canvas: np.ndarray, color: int, blocked: np.ndarray) -> bool:
    h, w = canvas.shape
    shape = random.choice(("hbar", "vbar", "rect", "ell"))
    if shape == "hbar":
        height, width = 1, random.randint(2, min(4, w))
    elif shape == "vbar":
        height, width = random.randint(2, min(4, h)), 1
    else:
        height = random.randint(2, min(3, h))
        width = random.randint(2, min(4, w))

    row0 = random.randint(0, h - height)
    col0 = random.randint(0, w - width)
    cells = _shape_cells(shape, row0, col0, height, width)
    rows = np.array([r for r, _c in cells], dtype=int)
    cols = np.array([c for _r, c in cells], dtype=int)
    if blocked[rows, cols].any():
        return False
    canvas[rows, cols] = color
    blocked[rows, cols] = True
    return True


def _build_candidate() -> np.ndarray | None:
    h = random.randint(*GRID_HEIGHT_RANGE)
    canvas_w = random.randint(*CANVAS_WIDTH_RANGE)
    instr_w = random.randint(*INSTR_WIDTH_RANGE)
    color = random.choice(PATTERN_COLORS)
    axes = _sample_axes(h)

    canvas = np.zeros((h, canvas_w), dtype=int)
    if not _add_axis_motifs(canvas, axes, color):
        return None

    blocked = canvas != BG
    for _ in range(random.randint(0, 2)):
        for _attempt in range(MAX_SHAPE_ATTEMPTS):
            if _add_extra_shape(canvas, color, blocked):
                break

    out_canvas, deltas = _apply_folds(canvas, axes)
    if len(deltas) != len(axes) or min(deltas) < MIN_STEP_DELTA:
        return None
    if int(np.count_nonzero(out_canvas != canvas)) < max(3, len(axes) * MIN_STEP_DELTA):
        return None

    in_density = float(np.count_nonzero(canvas)) / float(canvas.size)
    out_density = float(np.count_nonzero(out_canvas)) / float(out_canvas.size)
    if not (0.05 <= in_density <= 0.35):
        return None
    if out_density > 0.7:
        return None

    grid = np.zeros((h, instr_w + 1 + canvas_w), dtype=int)
    grid[:, instr_w] = DIVIDER
    grid[:, instr_w + 1 :] = canvas
    for axis in axes:
        grid[axis, random.randrange(instr_w)] = RED

    try:
        solved = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, out_canvas):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
