from __future__ import annotations

from collections import Counter
import itertools
import random

import numpy as np


COLORS = tuple(range(10))
SIDES = ("top", "bottom", "left", "right")
NEIGHBORS8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)

GRID_H_RANGE = (12, 18)
GRID_W_RANGE = (12, 18)
INNER_H_RANGE = (4, 8)
INNER_W_RANGE = (4, 8)
INNER_MARGIN = 3
PALETTE_SIZE_RANGE = (3, 6)
PATCH_COUNT_RANGE = (3, 7)
MAX_GENERATION_ATTEMPTS = 400


def _most_common_color(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    order = np.argsort(cnts)[::-1]
    if len(order) >= 2:
        assert int(cnts[order[0]]) > int(cnts[order[1]]), "background must be uniquely dominant"
    return int(vals[order[0]])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _band_components(grid: np.ndarray, bg: int) -> list[tuple[int, list[tuple[int, int]]]]:
    h, w = grid.shape
    band = np.zeros((h, w), dtype=bool)
    band[:2, :] = True
    band[-2:, :] = True
    band[:, :2] = True
    band[:, -2:] = True

    visited = np.zeros((h, w), dtype=bool)
    comps: list[tuple[int, list[tuple[int, int]]]] = []
    for r in range(h):
        for c in range(w):
            if not band[r, c] or visited[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            visited[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((int(cr), int(cc)))
                for dr, dc in NEIGHBORS8:
                    nr, nc = cr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if visited[nr, nc] or not band[nr, nc]:
                        continue
                    if int(grid[nr, nc]) != color:
                        continue
                    visited[nr, nc] = True
                    stack.append((nr, nc))
            comps.append((color, cells))
    return comps


def _classify_arrow(cells: list[tuple[int, int]], h: int, w: int) -> tuple[str, int] | None:
    s = set(cells)
    if len(s) != 3:
        return None

    top = [pt for pt in cells if pt[0] == 0]
    if len(top) == 1:
        c = top[0][1]
        if 1 <= c < w - 1 and s == {(0, c), (1, c - 1), (1, c + 1)}:
            return ("top", int(c))

    bottom = [pt for pt in cells if pt[0] == h - 1]
    if len(bottom) == 1:
        c = bottom[0][1]
        if 1 <= c < w - 1 and s == {(h - 1, c), (h - 2, c - 1), (h - 2, c + 1)}:
            return ("bottom", int(c))

    left = [pt for pt in cells if pt[1] == 0]
    if len(left) == 1:
        r = left[0][0]
        if 1 <= r < h - 1 and s == {(r, 0), (r - 1, 1), (r + 1, 1)}:
            return ("left", int(r))

    right = [pt for pt in cells if pt[1] == w - 1]
    if len(right) == 1:
        r = right[0][0]
        if 1 <= r < h - 1 and s == {(r, w - 1), (r - 1, w - 2), (r + 1, w - 2)}:
            return ("right", int(r))

    return None


def _parse_instance(grid: np.ndarray) -> tuple[np.ndarray, list[tuple[str, int, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert h >= 7 and w >= 7, "grid must leave room for a border band and inner area"

    bg = _most_common_color(g)
    interior_mask = np.zeros((h, w), dtype=bool)
    interior_mask[2 : h - 2, 2 : w - 2] = True
    coords = np.argwhere((g != bg) & interior_mask)
    assert len(coords) > 0, "expected a non-empty inner pattern"
    r0, c0, r1, c1 = _bbox(coords)
    inner = g[r0 : r1 + 1, c0 : c1 + 1].copy()

    arrows: list[tuple[str, int, int]] = []
    for color, cells in _band_components(g, bg):
        spec = _classify_arrow(cells, h, w)
        assert spec is not None, "every border-band component must be an arrow"
        side, absolute_idx = spec
        if side in ("top", "bottom"):
            local_idx = absolute_idx - c0
            assert 0 <= local_idx < inner.shape[1], "column arrow must target the inner area"
        else:
            local_idx = absolute_idx - r0
            assert 0 <= local_idx < inner.shape[0], "row arrow must target the inner area"
        arrows.append((side, int(local_idx), int(color)))

    assert arrows, "expected at least one arrow"
    return inner, arrows


def _shift_vectors(inner: np.ndarray, arrows: list[tuple[str, int, int]]) -> tuple[list[int], list[int]]:
    inner_h, inner_w = inner.shape
    row_shifts = [0] * inner_h
    col_shifts = [0] * inner_w
    used_rows: set[int] = set()
    used_cols: set[int] = set()

    for side, idx, color in arrows:
        if side == "left":
            assert idx not in used_rows, "each row may be controlled at most once"
            used_rows.add(idx)
            row_shifts[idx] = int(np.count_nonzero(inner[idx, :] == color))
        elif side == "right":
            assert idx not in used_rows, "each row may be controlled at most once"
            used_rows.add(idx)
            row_shifts[idx] = -int(np.count_nonzero(inner[idx, :] == color))
        elif side == "top":
            assert idx not in used_cols, "each column may be controlled at most once"
            used_cols.add(idx)
            col_shifts[idx] = int(np.count_nonzero(inner[:, idx] == color))
        else:
            assert side == "bottom"
            assert idx not in used_cols, "each column may be controlled at most once"
            used_cols.add(idx)
            col_shifts[idx] = -int(np.count_nonzero(inner[:, idx] == color))

    return row_shifts, col_shifts


def _apply_simultaneous_shifts(
    inner: np.ndarray, row_shifts: list[int], col_shifts: list[int]
) -> np.ndarray:
    inner_h, inner_w = inner.shape
    out = np.full(inner.shape, -1, dtype=int)
    for r in range(inner_h):
        for c in range(inner_w):
            rr = (r + col_shifts[c]) % inner_h
            cc = (c + row_shifts[r]) % inner_w
            assert out[rr, cc] == -1, "simultaneous row/column shifts must form a permutation"
            out[rr, cc] = int(inner[r, c])
    assert np.all(out != -1)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shift rows and columns according to inward border arrows, then return only the inner pattern.

    Each arrow controls the row/column aligned with its head center:
    - top  -> shift that column downward
    - bottom -> shift that column upward
    - left -> shift that row rightward
    - right -> shift that row leftward

    The shift amount equals the number of pixels of the arrow color in the targeted
    inner row/column. Row and column shifts are applied simultaneously.
    """
    inner, arrows = _parse_instance(np.asarray(grid, dtype=int))
    row_shifts, col_shifts = _shift_vectors(inner, arrows)
    return _apply_simultaneous_shifts(inner, row_shifts, col_shifts)


def _sample_layout() -> tuple[int, int, int, int, int, int]:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    max_inner_h = min(INNER_H_RANGE[1], h - 2 * INNER_MARGIN)
    max_inner_w = min(INNER_W_RANGE[1], w - 2 * INNER_MARGIN)
    if max_inner_h < INNER_H_RANGE[0] or max_inner_w < INNER_W_RANGE[0]:
        raise AssertionError("grid range leaves no room for the inner area")

    inner_h = random.randint(INNER_H_RANGE[0], max_inner_h)
    inner_w = random.randint(INNER_W_RANGE[0], max_inner_w)
    top = random.randint(INNER_MARGIN, h - INNER_MARGIN - inner_h)
    left = random.randint(INNER_MARGIN, w - INNER_MARGIN - inner_w)
    return h, w, top, left, inner_h, inner_w


def _sample_targets(inner_h: int, inner_w: int, sides: list[str]) -> dict[str, int]:
    targets: dict[str, int] = {}
    vertical_sides = [side for side in sides if side in ("top", "bottom")]
    horizontal_sides = [side for side in sides if side in ("left", "right")]

    if vertical_sides:
        cols = random.sample(list(range(inner_w)), len(vertical_sides))
        random.shuffle(cols)
        for side, col in zip(vertical_sides, cols):
            targets[side] = int(col)

    if horizontal_sides:
        rows = random.sample(list(range(inner_h)), len(horizontal_sides))
        random.shuffle(rows)
        for side, row in zip(horizontal_sides, rows):
            targets[side] = int(row)

    return targets


def _paint_pattern(inner_h: int, inner_w: int, palette: list[int]) -> np.ndarray:
    inner = np.array(
        [[random.choice(palette) for _ in range(inner_w)] for _ in range(inner_h)],
        dtype=int,
    )

    for _ in range(random.randint(*PATCH_COUNT_RANGE)):
        color = random.choice(palette)
        mode = random.choice(("rect", "row", "col"))
        if mode == "rect":
            r0 = random.randint(0, inner_h - 1)
            c0 = random.randint(0, inner_w - 1)
            r1 = random.randint(r0, inner_h - 1)
            c1 = random.randint(c0, inner_w - 1)
            inner[r0 : r1 + 1, c0 : c1 + 1] = color
        elif mode == "row":
            r = random.randint(0, inner_h - 1)
            c0 = random.randint(0, inner_w - 1)
            c1 = random.randint(c0, inner_w - 1)
            inner[r, c0 : c1 + 1] = color
        else:
            c = random.randint(0, inner_w - 1)
            r0 = random.randint(0, inner_h - 1)
            r1 = random.randint(r0, inner_h - 1)
            inner[r0 : r1 + 1, c] = color

    used = set(map(int, np.unique(inner)))
    missing = [color for color in palette if color not in used]
    if missing:
        cells = [(r, c) for r in range(inner_h) for c in range(inner_w)]
        random.shuffle(cells)
        for color, (r, c) in zip(missing, cells):
            inner[r, c] = color

    return inner


def _arrow_candidates(inner: np.ndarray, side: str, idx: int) -> list[tuple[int, int]]:
    if side in ("left", "right"):
        values = inner[idx, :].tolist()
        axis_len = inner.shape[1]
    else:
        values = inner[:, idx].tolist()
        axis_len = inner.shape[0]

    counts = Counter(map(int, values))
    return [
        (int(color), int(count))
        for color, count in counts.items()
        if 0 < int(count) < axis_len
    ]


def _choose_arrows(inner: np.ndarray, sides: list[str], targets: dict[str, int]) -> list[tuple[str, int, int]] | None:
    candidates_by_side: dict[str, list[tuple[int, int]]] = {}
    for side in sides:
        candidates = _arrow_candidates(inner, side, targets[side])
        if not candidates:
            return None
        random.shuffle(candidates)
        candidates_by_side[side] = candidates

    side_order = list(sides)
    valid: list[list[tuple[str, int, int]]] = []
    for combo in itertools.product(*(candidates_by_side[side] for side in side_order)):
        arrows = [
            (side, int(targets[side]), int(color))
            for side, (color, _count) in zip(side_order, combo)
        ]
        row_shifts, col_shifts = _shift_vectors(inner, arrows)
        try:
            shifted = _apply_simultaneous_shifts(inner, row_shifts, col_shifts)
        except AssertionError:
            continue
        if np.array_equal(shifted, inner):
            continue
        valid.append(arrows)

    if not valid:
        return None

    best_distinct = max(len({color for _side, _idx, color in arrows}) for arrows in valid)
    best = [arrows for arrows in valid if len({color for _side, _idx, color in arrows}) == best_distinct]
    return random.choice(best)


def _arrow_cells(side: str, absolute_idx: int, h: int, w: int) -> list[tuple[int, int]]:
    if side == "top":
        c = absolute_idx
        return [(0, c), (1, c - 1), (1, c + 1)]
    if side == "bottom":
        c = absolute_idx
        return [(h - 1, c), (h - 2, c - 1), (h - 2, c + 1)]
    if side == "left":
        r = absolute_idx
        return [(r, 0), (r - 1, 1), (r + 1, 1)]
    assert side == "right"
    r = absolute_idx
    return [(r, w - 1), (r - 1, w - 2), (r + 1, w - 2)]


def _render_grid(
    h: int,
    w: int,
    top: int,
    left: int,
    inner: np.ndarray,
    arrows: list[tuple[str, int, int]],
    bg: int,
) -> np.ndarray:
    grid = np.full((h, w), bg, dtype=int)
    inner_h, inner_w = inner.shape
    grid[top : top + inner_h, left : left + inner_w] = inner

    for side, local_idx, color in arrows:
        absolute_idx = left + local_idx if side in ("top", "bottom") else top + local_idx
        for r, c in _arrow_cells(side, absolute_idx, h, w):
            assert grid[r, c] == bg, "arrows must not overlap each other or the inner area"
            grid[r, c] = color

    return grid


def _background_is_unique_dominant(grid: np.ndarray, bg: int) -> bool:
    vals, cnts = np.unique(grid, return_counts=True)
    counts = {int(v): int(c) for v, c in zip(vals, cnts)}
    bg_count = counts.get(bg, 0)
    return all(color == bg or bg_count > count for color, count in counts.items())


def _generate_attempt() -> np.ndarray | None:
    h, w, top, left, inner_h, inner_w = _sample_layout()
    bg = random.choice(COLORS)

    k = random.randint(1, len(SIDES))
    sides = random.sample(list(SIDES), k)
    targets = _sample_targets(inner_h, inner_w, sides)

    non_bg = [color for color in COLORS if color != bg]
    palette_size = random.randint(
        max(PALETTE_SIZE_RANGE[0], min(len(sides), PALETTE_SIZE_RANGE[1])),
        PALETTE_SIZE_RANGE[1],
    )
    palette = random.sample(non_bg, palette_size)
    inner = _paint_pattern(inner_h, inner_w, palette)

    arrows = _choose_arrows(inner, sides, targets)
    if arrows is None:
        return None

    grid = _render_grid(h, w, top, left, inner, arrows, bg)
    if not _background_is_unique_dominant(grid, bg):
        return None

    try:
        expected = _apply_simultaneous_shifts(inner, *_shift_vectors(inner, arrows))
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None

    parsed_inner, parsed_arrows = _parse_instance(grid)
    if not np.array_equal(parsed_inner, inner):
        return None
    if sorted(parsed_arrows) != sorted(arrows):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
