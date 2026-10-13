from __future__ import annotations

from collections import deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6

_RING_COLORS = (BLUE, YELLOW, GREEN, MAGENTA)

_LEGEND_SAMPLE_SIZE = 7
_LEGEND_CELL_HEIGHT = 1 + _LEGEND_SAMPLE_SIZE
_LEGEND_COL_GAP = 1
_LEGEND_ROW_GAP = 0
_LEGEND_HEIGHT = 2 * _LEGEND_CELL_HEIGHT + _LEGEND_ROW_GAP
_LEGEND_WIDTH = 2 * _LEGEND_SAMPLE_SIZE + _LEGEND_COL_GAP

_GRID_MIN = 26
_GRID_MAX = 30
_GEN_RETRIES = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _ring_template(level: int) -> np.ndarray:
    size = 2 * level - 1
    patch = np.full((size, size), BG, dtype=int)
    for inset in range(level):
        color = _RING_COLORS[inset]
        lo = inset
        hi = size - 1 - inset
        patch[lo, lo : hi + 1] = color
        patch[hi, lo : hi + 1] = color
        patch[lo : hi + 1, lo] = color
        patch[lo : hi + 1, hi] = color
    return patch


def _find_rect_frames(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    rects: list[tuple[int, int, int, int]] = []
    for comp in _components_mask(grid == GREY):
        pts = set(comp)
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        border = set()
        for c in range(c0, c1 + 1):
            border.add((r0, c))
            border.add((r1, c))
        for r in range(r0 + 1, r1):
            border.add((r, c0))
            border.add((r, c1))
        if pts == border:
            rects.append((r0, c0, r1, c1))
    rects.sort()
    return rects


def _paint_centered_ring(out: np.ndarray, r0: int, c0: int, r1: int, c1: int, level: int) -> None:
    interior = out[r0 + 1 : r1, c0 + 1 : c1]
    template = _ring_template(level)
    ih, iw = interior.shape
    th, tw = template.shape
    assert ih >= th and iw >= tw
    interior[:, :] = BG
    sr = (ih - th) // 2
    sc = (iw - tw) // 2
    view = interior[sr : sr + th, sc : sc + tw]
    mask = template != BG
    view[mask] = template[mask]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each seed-filled grey enclosure with the concentric-ring template indicated by its seed count.

    Input structure:
    - Black background with several one-pixel-thick grey rectangular fences.
    - Each task fence encloses only black cells plus 1 to 4 red seed pixels.
    - A teaching area sits near the lower-right corner and shows the four ring templates for counts 1..4.

    Rule:
    - Find every grey hollow rectangle whose interior contains only black and red.
    - Count the red seeds inside that rectangle, clear those seeds, and choose the matching 1..4 layer ring template.
    - Center the chosen template inside the rectangle's interior, leaving any extra interior padding black.
    - Keep the grey fences and the teaching area unchanged.

    Key insight:
    The bottom-right teaching examples define a fixed count-to-template mapping, so each fenced region is solved independently by counting its red seeds and stamping the corresponding centered ring.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    for r0, c0, r1, c1 in _find_rect_frames(g):
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        colors = set(map(int, np.unique(interior)))
        if not colors.issubset({BG, RED}):
            continue
        seed_count = int(np.count_nonzero(interior == RED))
        if 1 <= seed_count <= 4:
            _paint_centered_ring(out, r0, c0, r1, c1, seed_count)
    return out


def _draw_task_frame(
    grid: np.ndarray,
    expected: np.ndarray,
    top: int,
    left: int,
    inner_h: int,
    inner_w: int,
    seed_count: int,
    rng: np.random.Generator,
) -> None:
    bottom = top + inner_h + 1
    right = left + inner_w + 1
    grid[top, left : right + 1] = GREY
    grid[bottom, left : right + 1] = GREY
    grid[top : bottom + 1, left] = GREY
    grid[top : bottom + 1, right] = GREY

    expected[top, left : right + 1] = GREY
    expected[bottom, left : right + 1] = GREY
    expected[top : bottom + 1, left] = GREY
    expected[top : bottom + 1, right] = GREY

    cell_ids = rng.choice(inner_h * inner_w, size=seed_count, replace=False)
    for cell_id in cell_ids:
        rr = int(cell_id // inner_w)
        cc = int(cell_id % inner_w)
        grid[top + 1 + rr, left + 1 + cc] = RED

    _paint_centered_ring(expected, top, left, bottom, right, seed_count)


def _draw_legend(grid: np.ndarray, top: int, left: int) -> None:
    for level in range(1, 5):
        cell_row = (level - 1) // 2
        cell_col = (level - 1) % 2
        cell_top = top + cell_row * (_LEGEND_CELL_HEIGHT + _LEGEND_ROW_GAP)
        cell_left = left + cell_col * (_LEGEND_SAMPLE_SIZE + _LEGEND_COL_GAP)

        seed_start = cell_left + (_LEGEND_SAMPLE_SIZE - level) // 2
        grid[cell_top, seed_start : seed_start + level] = RED

        template = _ring_template(level)
        th, tw = template.shape
        patch_top = cell_top + 1 + (_LEGEND_SAMPLE_SIZE - th) // 2
        patch_left = cell_left + (_LEGEND_SAMPLE_SIZE - tw) // 2
        patch = grid[patch_top : patch_top + th, patch_left : patch_left + tw]
        mask = template != BG
        patch[mask] = template[mask]


def _choose_inner_size(level: int, rng: np.random.Generator) -> int:
    base = 2 * level - 1
    if level == 4:
        extra = int(rng.choice([0, 2]))
    else:
        extra = int(rng.choice([0, 2, 4]))
    return base + extra


def _find_positions(occ: np.ndarray, obj_h: int, obj_w: int) -> list[tuple[int, int]]:
    h, w = occ.shape
    positions: list[tuple[int, int]] = []
    for top in range(1, h - obj_h):
        for left in range(1, w - obj_w):
            if occ[top - 1 : top + obj_h + 1, left - 1 : left + obj_w + 1].any():
                continue
            positions.append((top, left))
    return positions


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    height = int(rng.integers(_GRID_MIN, _GRID_MAX + 1))
    width = int(rng.integers(_GRID_MIN, _GRID_MAX + 1))

    grid = np.full((height, width), BG, dtype=int)
    expected = np.full((height, width), BG, dtype=int)
    occ = np.zeros((height, width), dtype=bool)

    legend_top = height - _LEGEND_HEIGHT - 1
    legend_left = width - _LEGEND_WIDTH - 1
    if legend_top < 1 or legend_left < 1:
        return None
    occ[legend_top - 1 : height, legend_left - 1 : width] = True
    _draw_legend(grid, legend_top, legend_left)
    _draw_legend(expected, legend_top, legend_left)

    counts = [1, 2, 3, 4]
    extra_count = int(rng.integers(0, 3))
    if extra_count:
        extras = rng.integers(1, 5, size=extra_count)
        counts.extend(int(x) for x in extras)
    rng.shuffle(counts)

    frame_specs = []
    for count in counts:
        inner_h = _choose_inner_size(count, rng)
        inner_w = _choose_inner_size(count, rng)
        frame_specs.append(
            {
                "count": count,
                "inner_h": inner_h,
                "inner_w": inner_w,
                "outer_h": inner_h + 2,
                "outer_w": inner_w + 2,
            }
        )
    frame_specs.sort(key=lambda spec: (spec["outer_h"] * spec["outer_w"]), reverse=True)

    for spec in frame_specs:
        positions = _find_positions(occ, int(spec["outer_h"]), int(spec["outer_w"]))
        if not positions:
            return None
        top, left = positions[int(rng.integers(len(positions)))]
        _draw_task_frame(
            grid,
            expected,
            top,
            left,
            int(spec["inner_h"]),
            int(spec["inner_w"]),
            int(spec["count"]),
            rng,
        )
        occ[top - 1 : top + int(spec["outer_h"]) + 1, left - 1 : left + int(spec["outer_w"]) + 1] = True

    try:
        if not np.array_equal(main(grid), expected):
            return None
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
