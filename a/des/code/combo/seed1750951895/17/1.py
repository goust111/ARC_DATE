from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
COLORS = tuple(range(1, 10))
DIR8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _find_components(grid: np.ndarray, bg: int) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIR8:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) != bg:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _paint(out: np.ndarray, r: int, c: int, color: int, bg: int) -> None:
    h, w = out.shape
    assert 0 <= r < h and 0 <= c < w, "shifted pixel left the grid"
    assert int(out[r, c]) == bg, "shifted pixels collided"
    out[r, c] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected 2D grid"
    bg = _get_bg(g)
    out = np.full_like(g, bg)

    for coords in _find_components(g, bg):
        vals = g[coords[:, 0], coords[:, 1]]
        counts = Counter(map(int, vals))
        assert len(counts) == 2, "each object must use one body color and one marker color"
        marker, body = sorted(counts, key=lambda color: counts[color])
        assert counts[marker] < counts[body], "marker color must be the minority color"

        marker_mask = vals == marker
        body_coords = coords[vals == body]
        marker_coords = coords[marker_mask]

        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        height = int(r1 - r0 + 1)
        width = int(c1 - c0 + 1)
        assert height != width, "object bbox must reveal orientation"

        if width > height:
            marker_by_row: dict[int, int] = {}
            for r, c in marker_coords:
                rr = int(r)
                cc = int(c)
                assert rr not in marker_by_row, "wide object must have one marker per row"
                marker_by_row[rr] = cc
            assert set(marker_by_row) == set(range(int(r0), int(r1) + 1)), "marker line must span object height"

            for r, c in body_coords:
                rr = int(r)
                cc = int(c)
                crack_c = marker_by_row[rr]
                assert cc != crack_c, "body cannot occupy crack"
                _paint(out, rr, cc - 1 if cc < crack_c else cc + 1, body, bg)
        else:
            marker_by_col: dict[int, int] = {}
            for r, c in marker_coords:
                rr = int(r)
                cc = int(c)
                assert cc not in marker_by_col, "tall object must have one marker per column"
                marker_by_col[cc] = rr
            assert set(marker_by_col) == set(range(int(c0), int(c1) + 1)), "marker line must span object width"

            for r, c in body_coords:
                rr = int(r)
                cc = int(c)
                crack_r = marker_by_col[cc]
                assert rr != crack_r, "body cannot occupy crack"
                _paint(out, rr - 1 if rr < crack_r else rr + 1, cc, body, bg)

    return out


def _reserve_bbox(top: int, left: int, height: int, width: int) -> tuple[int, int, int, int]:
    if width > height:
        return top - 1, left - 2, top + height, left + width + 1
    return top - 2, left - 1, top + height + 1, left + width


def _overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _sample_marker_positions(length: int) -> list[int]:
    base = [0, 1] + [random.randrange(2) for _ in range(max(0, length - 2))]
    random.shuffle(base)
    return base


def _draw_object(
    grid: np.ndarray,
    top: int,
    left: int,
    height: int,
    width: int,
    body: int,
    marker: int,
) -> None:
    grid[top : top + height, left : left + width] = body
    short_side = min(height, width)
    offsets = _sample_marker_positions(short_side)

    if width > height:
        for i, dc in enumerate(offsets):
            grid[top + i, left + short_side - 1 + dc] = marker
    else:
        for i, dr in enumerate(offsets):
            grid[top + short_side - 1 + dr, left + i] = marker


def _place_specs(n: int, count: int) -> list[tuple[int, int, int, int]] | None:
    specs: list[tuple[int, int, int, int]] = []
    reserved: list[tuple[int, int, int, int]] = []

    if count == 1:
        orientations = [random.choice(["wide", "tall"])]
    else:
        wide_count = random.randint(1, count - 1)
        orientations = ["wide"] * wide_count + ["tall"] * (count - wide_count)
        random.shuffle(orientations)

    for orient in orientations:
        placed = False
        for _ in range(200):
            short_side = random.randint(2, 6)
            long_side = random.randint(short_side + 1, min(2 * short_side + 2, 10))
            height, width = (short_side, long_side) if orient == "wide" else (long_side, short_side)

            row_lo = 1 if orient == "wide" else 2
            row_hi = n - height - (1 if orient == "wide" else 2)
            col_lo = 2 if orient == "wide" else 1
            col_hi = n - width - (2 if orient == "wide" else 1)
            if row_lo > row_hi or col_lo > col_hi:
                continue

            top = random.randint(row_lo, row_hi)
            left = random.randint(col_lo, col_hi)
            box = _reserve_bbox(top, left, height, width)
            if any(_overlap(box, other) for other in reserved):
                continue

            specs.append((top, left, height, width))
            reserved.append(box)
            placed = True
            break
        if not placed:
            return None
    return specs


def generate_input() -> np.ndarray:
    for _ in range(400):
        n = random.randint(12, 24)
        count = random.randint(2, 4)
        specs = _place_specs(n, count)
        if specs is None:
            continue

        grid = np.full((n, n), BG, dtype=int)
        palette = random.sample(list(COLORS), 2 * count)
        random.shuffle(specs)
        for i, (top, left, height, width) in enumerate(specs):
            body = palette[2 * i]
            marker = palette[2 * i + 1]
            _draw_object(grid, top, left, height, width, body, marker)

        try:
            out = main(grid)
        except AssertionError:
            continue

        if np.array_equal(grid, out):
            continue
        return grid

    raise RuntimeError("failed to generate a valid input")
