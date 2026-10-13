from __future__ import annotations

import random
from collections import Counter

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))

    return components


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _paint_frame(grid: np.ndarray, r0: int, c0: int, inner_h: int, inner_w: int, color: int) -> None:
    outer_h = inner_h + 2
    outer_w = inner_w + 2
    grid[r0, c0 : c0 + outer_w] = color
    grid[r0 + outer_h - 1, c0 : c0 + outer_w] = color
    grid[r0 : r0 + outer_h, c0] = color
    grid[r0 : r0 + outer_h, c0 + outer_w - 1] = color


def _clear_all_cavities(
    grid: np.ndarray,
    row_starts: list[int],
    col_starts: list[int],
    inner_h: int,
    inner_w: int,
    bg: int,
) -> None:
    for r0 in row_starts:
        for c0 in col_starts:
            grid[r0 + 1 : r0 + 1 + inner_h, c0 + 1 : c0 + 1 + inner_w] = bg


def _infer_structure(grid: np.ndarray) -> tuple[int, int, list[int], list[int], int, int, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _get_bg(g)
    foreground = [int(color) for color in np.unique(g) if int(color) != bg]
    assert foreground, "expected scaffold frames"

    counts = {color: int(np.count_nonzero(g == color)) for color in foreground}
    scaffold_count = max(counts.values())
    scaffold_candidates = [color for color, count in counts.items() if count == scaffold_count]
    assert len(scaffold_candidates) == 1, "expected a unique scaffold color"
    scaffold = scaffold_candidates[0]

    components = _components4(g == scaffold)
    assert components, "expected scaffold components"

    bboxes = [_bbox(coords) for coords in components]
    outer_shapes = {(r1 - r0 + 1, c1 - c0 + 1) for r0, c0, r1, c1 in bboxes}
    assert len(outer_shapes) == 1, "all frames must share one geometry"
    outer_h, outer_w = next(iter(outer_shapes))
    assert outer_h >= 3 and outer_w >= 3, "frames must be hollow"

    inner_h = outer_h - 2
    inner_w = outer_w - 2
    assert len(components) == inner_h * inner_w, "frame count must match cavity size"

    frame_area = 2 * outer_h + 2 * outer_w - 4
    for coords, box in zip(components, bboxes):
        assert coords.shape[0] == frame_area, "each scaffold component must be a 1-cell-thick frame"
        r0, c0, r1, c1 = box
        expected = np.zeros((outer_h, outer_w), dtype=bool)
        expected[0, :] = True
        expected[-1, :] = True
        expected[:, 0] = True
        expected[:, -1] = True
        actual = np.zeros((outer_h, outer_w), dtype=bool)
        local = coords - np.array([r0, c0], dtype=int)
        actual[local[:, 0], local[:, 1]] = True
        assert np.array_equal(actual, expected), "scaffold components must be hollow rectangles"

    row_starts = sorted({r0 for r0, _, _, _ in bboxes})
    col_starts = sorted({c0 for _, c0, _, _ in bboxes})
    assert len(row_starts) == inner_h and len(col_starts) == inner_w, "frame lattice dimensions must match cavity size"

    if len(row_starts) > 1:
        row_steps = {row_starts[i + 1] - row_starts[i] for i in range(len(row_starts) - 1)}
        assert len(row_steps) == 1 and next(iter(row_steps)) > outer_h - 1, "row spacing must be uniform"
    if len(col_starts) > 1:
        col_steps = {col_starts[i + 1] - col_starts[i] for i in range(len(col_starts) - 1)}
        assert len(col_steps) == 1 and next(iter(col_steps)) > outer_w - 1, "column spacing must be uniform"

    frame_origins = {(r0, c0) for r0, c0, _, _ in bboxes}
    assert frame_origins == {(r0, c0) for r0 in row_starts for c0 in col_starts}, "frames must form a full lattice"

    map_candidates: list[np.ndarray] = []
    for r0 in row_starts:
        for c0 in col_starts:
            cavity = g[r0 + 1 : r0 + 1 + inner_h, c0 + 1 : c0 + 1 + inner_w]
            if np.any(cavity != bg):
                map_candidates.append(cavity.copy())

    assert len(map_candidates) == 1, "expected exactly one frame containing the miniature map"
    return bg, scaffold, row_starts, col_starts, inner_h, inner_w, map_candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the unique miniature map to recolor the matching frame cavities across the whole lattice.

    Input structure:
    - A black background contains an evenly spaced lattice of identical hollow rectangular frames in one scaffold color.
    - The lattice has R frame rows and C frame columns, and every frame cavity is exactly R x C cells.
    - Exactly one frame cavity contains an R x C miniature map; all other cavities are empty.

    Rule:
    - Infer the repeated frame lattice and read the miniature map from the only non-empty cavity.
    - For each map cell (i, j), if the miniature is non-black with color c, fill the entire cavity of frame (i, j) with c.
    - If a map cell is black, leave that frame cavity empty; the map-bearing frame is cleared first and then treated the same way.

    Key insight:
    - The drawing inside one frame is not decoration but a scale-isomorphic self-map of the whole frame array, including the frame that currently holds the map itself.
    """
    bg, scaffold, row_starts, col_starts, inner_h, inner_w, mini_map = _infer_structure(grid)

    out = np.asarray(grid, dtype=int).copy()
    _clear_all_cavities(out, row_starts, col_starts, inner_h, inner_w, bg)

    for i, r0 in enumerate(row_starts):
        for j, c0 in enumerate(col_starts):
            color = int(mini_map[i, j])
            if color != bg:
                out[r0 + 1 : r0 + 1 + inner_h, c0 + 1 : c0 + 1 + inner_w] = color

    return out


# --------------- generator ---------------

_BG = 0
_PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
_DIM_RANGE = (2, 4)
_GAP_RANGE = (1, 2)
_MARGIN_RANGE = (1, 2)
_MAX_SIDE = 30
_MAX_ATTEMPTS = 400
_BRANCHES = ("self_black", "self_color", "mixed")


def _sample_layout() -> tuple[int, int, int, int, int]:
    for _ in range(200):
        rows = random.randint(*_DIM_RANGE)
        cols = random.randint(*_DIM_RANGE)
        row_gap = random.randint(*_GAP_RANGE)
        col_gap = random.randint(*_GAP_RANGE)
        margin = random.randint(*_MARGIN_RANGE)
        height = 2 * margin + rows * (rows + 2) + (rows - 1) * row_gap
        width = 2 * margin + cols * (cols + 2) + (cols - 1) * col_gap
        if height <= _MAX_SIDE and width <= _MAX_SIDE:
            return rows, cols, row_gap, col_gap, margin
    raise RuntimeError("failed to sample a valid lattice layout")


def _frame_starts(count: int, inner: int, gap: int, margin: int) -> list[int]:
    outer = inner + 2
    step = outer + gap
    return [margin + idx * step for idx in range(count)]


def _make_base(
    rows: int,
    cols: int,
    row_gap: int,
    col_gap: int,
    margin: int,
    scaffold: int,
) -> tuple[np.ndarray, list[int], list[int]]:
    row_starts = _frame_starts(rows, rows, row_gap, margin)
    col_starts = _frame_starts(cols, cols, col_gap, margin)
    height = row_starts[-1] + rows + 2 + margin
    width = col_starts[-1] + cols + 2 + margin
    grid = np.full((height, width), _BG, dtype=int)

    for r0 in row_starts:
        for c0 in col_starts:
            _paint_frame(grid, r0, c0, rows, cols, scaffold)

    return grid, row_starts, col_starts


def _sample_palette(scaffold: int, non_black_count: int) -> list[int]:
    available = [color for color in _PALETTE if color != scaffold]
    max_size = min(4, len(available), max(1, non_black_count))
    if max_size == 1:
        size = 1
    else:
        size = random.randint(1, max_size)
        if non_black_count >= 3:
            size = max(2, size) if random.random() < 0.6 else size
    return random.sample(available, size)


def _sample_map(rows: int, cols: int, map_pos: tuple[int, int], scaffold: int) -> np.ndarray:
    total = rows * cols
    branch = random.choice(_BRANCHES)

    if branch == "self_black":
        non_black_count = random.randint(1, max(1, total - 1))
    elif random.random() < 0.8:
        non_black_count = random.randint(max(1, total // 3), max(1, total - 1))
    else:
        non_black_count = total

    palette = _sample_palette(scaffold, non_black_count)
    mini_map = np.zeros((rows, cols), dtype=int)

    indices = [(r, c) for r in range(rows) for c in range(cols)]
    random.shuffle(indices)
    chosen = indices[:non_black_count]
    for r, c in chosen:
        mini_map[r, c] = random.choice(palette)

    mr, mc = map_pos
    if branch == "self_black":
        mini_map[mr, mc] = _BG
        if not np.any(mini_map != _BG):
            rr, cc = random.choice([(r, c) for r in range(rows) for c in range(cols) if (r, c) != map_pos])
            mini_map[rr, cc] = random.choice(palette)
    elif branch == "self_color":
        mini_map[mr, mc] = random.choice(palette)

    if not np.any(mini_map != _BG):
        rr, cc = random.randrange(rows), random.randrange(cols)
        mini_map[rr, cc] = random.choice(palette)

    return mini_map


def _expected_from_map(
    rows: int,
    cols: int,
    row_starts: list[int],
    col_starts: list[int],
    scaffold_grid: np.ndarray,
    mini_map: np.ndarray,
) -> np.ndarray:
    out = scaffold_grid.copy()
    _clear_all_cavities(out, row_starts, col_starts, rows, cols, _BG)
    for r in range(rows):
        for c in range(cols):
            color = int(mini_map[r, c])
            if color != _BG:
                r0 = row_starts[r]
                c0 = col_starts[c]
                out[r0 + 1 : r0 + 1 + rows, c0 + 1 : c0 + 1 + cols] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    rows, cols, row_gap, col_gap, margin = _sample_layout()
    scaffold = random.choice(_PALETTE)
    base, row_starts, col_starts = _make_base(rows, cols, row_gap, col_gap, margin, scaffold)

    map_pos = (random.randrange(rows), random.randrange(cols))
    mini_map = _sample_map(rows, cols, map_pos, scaffold)

    inp = base.copy()
    r0 = row_starts[map_pos[0]]
    c0 = col_starts[map_pos[1]]
    inp[r0 + 1 : r0 + 1 + rows, c0 + 1 : c0 + 1 + cols] = mini_map

    expected = _expected_from_map(rows, cols, row_starts, col_starts, base, mini_map)

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
