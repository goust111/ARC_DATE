from __future__ import annotations

import random

import numpy as np


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows, cols = np.where(mask)
    assert rows.size > 0
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _runs(values: list[int]) -> list[tuple[int, int, int]]:
    runs: list[tuple[int, int, int]] = []
    start = 0
    while start < len(values):
        color = int(values[start])
        end = start + 1
        while end < len(values) and int(values[end]) == color:
            end += 1
        runs.append((start, end, color))
        start = end
    return runs


def _analyze_patch(patch: np.ndarray, orientation: str) -> dict | None:
    if orientation == "H":
        line_colors: list[int] = []
        for r in range(patch.shape[0]):
            if not bool(np.all(patch[r] == patch[r, 0])):
                return None
            line_colors.append(int(patch[r, 0]))
    else:
        line_colors = []
        for c in range(patch.shape[1]):
            if not bool(np.all(patch[:, c] == patch[0, c])):
                return None
            line_colors.append(int(patch[0, c]))

    runs = _runs(line_colors)
    if len(runs) < 2:
        return None

    widths = {end - start for start, end, _color in runs}
    if len(widths) != 1:
        return None

    stripe_colors = [color for _start, _end, color in runs]
    if len(stripe_colors) != len(set(stripe_colors)):
        return None

    return {
        "orientation": orientation,
        "stripe_width": next(iter(widths)),
        "runs": runs,
    }


def _bg_visible_outside(g: np.ndarray, bbox: tuple[int, int, int, int], bg: int) -> bool:
    r0, c0, r1, c1 = bbox
    outside = np.ones(g.shape, dtype=bool)
    outside[r0 : r1 + 1, c0 : c1 + 1] = False
    return bool(np.any(g[outside] == int(bg)))


def _find_structure(g: np.ndarray) -> tuple[int, tuple[int, int, int, int], dict]:
    candidates: list[tuple[int, tuple[int, int, int, int], dict]] = []

    for bg in np.unique(g):
        mask = g != int(bg)
        if not bool(mask.any()):
            continue

        box = _bbox(mask)
        r0, c0, r1, c1 = box
        patch = g[r0 : r1 + 1, c0 : c1 + 1]

        if bool(np.any(patch == int(bg))):
            continue
        if not _bg_visible_outside(g, box, int(bg)):
            continue

        valid = []
        for orientation in ("H", "V"):
            desc = _analyze_patch(patch, orientation)
            if desc is not None:
                valid.append(desc)
        if len(valid) == 1:
            candidates.append((int(bg), box, valid[0]))

    assert len(candidates) == 1, "expected exactly one stripe-pattern interpretation"
    return candidates[0]


def _weave(
    shape: tuple[int, int],
    bg: int,
    bbox: tuple[int, int, int, int],
    desc: dict,
) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    r0, c0, r1, c1 = bbox

    if desc["orientation"] == "H":
        span = c1 - c0 + 1
        for idx, (start, end, color) in enumerate(desc["runs"]):
            shift = 1 if idx % 2 == 1 else 0
            dst_c0 = c0 + shift
            dst_c1 = min(shape[1], dst_c0 + span)
            if dst_c0 < dst_c1:
                out[r0 + start : r0 + end, dst_c0:dst_c1] = int(color)
    else:
        span = r1 - r0 + 1
        for idx, (start, end, color) in enumerate(desc["runs"]):
            shift = 1 if idx % 2 == 1 else 0
            dst_r0 = r0 + shift
            dst_r1 = min(shape[0], dst_r0 + span)
            if dst_r0 < dst_r1:
                out[dst_r0:dst_r1, c0 + start : c0 + end] = int(color)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Weave a stripe block by shifting odd-indexed stripes one cell sideways.

    Input structure
    - A single rectangular block of equal-width horizontal or vertical stripes sits on a
      uniform background.
    - Adjacent stripes have different colors, and the stripes fully tile the block.

    Rule
    - Determine whether the stripes are horizontal or vertical.
    - Number stripes from one end of the block: 0, 1, 2, ...
    - Even-indexed stripes stay fixed.
    - Odd-indexed stripes shift one cell perpendicular to their length:
      horizontal stripes move right, vertical stripes move down.
    - Vacated cells become background and anything shifted beyond the grid is clipped.

    Key insight
    - The transformation depends on stripe order parity, not on absolute position or color.
    """
    g = np.asarray(grid, dtype=int)
    bg, box, desc = _find_structure(g)
    return _weave(g.shape, bg, box, desc)


# --------------- generator ---------------

_GRID_SIDE_RANGE = (8, 18)
_STRIPE_COUNT_RANGE = (3, 6)
_STRIPE_WIDTH_MAX = 3
_PARALLEL_SPAN_MIN = 4
_MAX_ATTEMPTS = 400
_CLIP_PROB = 0.5


def _sample_horizontal_dims(grid_h: int, grid_w: int, clip: bool) -> tuple[int, int, int] | None:
    max_count = min(_STRIPE_COUNT_RANGE[1], grid_h - 1)
    if max_count < _STRIPE_COUNT_RANGE[0]:
        return None

    stripe_count = random.randint(_STRIPE_COUNT_RANGE[0], max_count)
    max_width = min(_STRIPE_WIDTH_MAX, (grid_h - 1) // stripe_count)
    if max_width < 1:
        return None

    stripe_width = random.randint(1, max_width)
    block_h = stripe_count * stripe_width
    max_block_w = grid_w if clip else grid_w - 1
    if max_block_w < _PARALLEL_SPAN_MIN:
        return None

    block_w = random.randint(_PARALLEL_SPAN_MIN, max_block_w)
    return stripe_count, stripe_width, block_w


def _sample_vertical_dims(grid_h: int, grid_w: int, clip: bool) -> tuple[int, int, int] | None:
    max_count = min(_STRIPE_COUNT_RANGE[1], grid_w - 1)
    if max_count < _STRIPE_COUNT_RANGE[0]:
        return None

    stripe_count = random.randint(_STRIPE_COUNT_RANGE[0], max_count)
    max_width = min(_STRIPE_WIDTH_MAX, (grid_w - 1) // stripe_count)
    if max_width < 1:
        return None

    stripe_width = random.randint(1, max_width)
    block_w = stripe_count * stripe_width
    max_block_h = grid_h if clip else grid_h - 1
    if max_block_h < _PARALLEL_SPAN_MIN:
        return None

    block_h = random.randint(_PARALLEL_SPAN_MIN, max_block_h)
    return stripe_count, stripe_width, block_h


def _render_horizontal(
    grid_h: int,
    grid_w: int,
    bg: int,
    clip: bool,
    stripe_count: int,
    stripe_width: int,
    block_w: int,
) -> tuple[np.ndarray, int, tuple[int, int, int, int], dict]:
    block_h = stripe_count * stripe_width
    top = random.randint(0, grid_h - block_h)
    if clip:
        left = grid_w - block_w
    else:
        left = random.randint(0, grid_w - block_w - 1)

    colors = random.sample([c for c in range(10) if c != bg], stripe_count)
    g = np.full((grid_h, grid_w), int(bg), dtype=int)
    runs = []
    for idx, color in enumerate(colors):
        r0 = top + idx * stripe_width
        r1 = r0 + stripe_width
        g[r0:r1, left : left + block_w] = int(color)
        runs.append((idx * stripe_width, (idx + 1) * stripe_width, int(color)))

    bbox = (top, left, top + block_h - 1, left + block_w - 1)
    desc = {"orientation": "H", "stripe_width": stripe_width, "runs": runs}
    return g, int(bg), bbox, desc


def _render_vertical(
    grid_h: int,
    grid_w: int,
    bg: int,
    clip: bool,
    stripe_count: int,
    stripe_width: int,
    block_h: int,
) -> tuple[np.ndarray, int, tuple[int, int, int, int], dict]:
    block_w = stripe_count * stripe_width
    left = random.randint(0, grid_w - block_w)
    if clip:
        top = grid_h - block_h
    else:
        top = random.randint(0, grid_h - block_h - 1)

    colors = random.sample([c for c in range(10) if c != bg], stripe_count)
    g = np.full((grid_h, grid_w), int(bg), dtype=int)
    runs = []
    for idx, color in enumerate(colors):
        c0 = left + idx * stripe_width
        c1 = c0 + stripe_width
        g[top : top + block_h, c0:c1] = int(color)
        runs.append((idx * stripe_width, (idx + 1) * stripe_width, int(color)))

    bbox = (top, left, top + block_h - 1, left + block_w - 1)
    desc = {"orientation": "V", "stripe_width": stripe_width, "runs": runs}
    return g, int(bg), bbox, desc


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.randint(*_GRID_SIDE_RANGE)
    grid_w = random.randint(*_GRID_SIDE_RANGE)
    bg = random.randrange(10)
    orientation = random.choice(("H", "V"))
    clip = random.random() < _CLIP_PROB

    if orientation == "H":
        dims = _sample_horizontal_dims(grid_h, grid_w, clip)
        if dims is None:
            return None
        stripe_count, stripe_width, block_w = dims
        inp, bg, bbox, desc = _render_horizontal(
            grid_h, grid_w, bg, clip, stripe_count, stripe_width, block_w
        )
    else:
        dims = _sample_vertical_dims(grid_h, grid_w, clip)
        if dims is None:
            return None
        stripe_count, stripe_width, block_h = dims
        inp, bg, bbox, desc = _render_vertical(
            grid_h, grid_w, bg, clip, stripe_count, stripe_width, block_h
        )

    expected = _weave(inp.shape, bg, bbox, desc)
    if np.array_equal(inp, expected):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid woven-stripe input")
