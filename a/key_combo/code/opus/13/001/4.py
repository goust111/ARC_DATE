from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BACKGROUND = 0
RECT_COLOR = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_ATTEMPTS = 500


def _compress(values: list[int]) -> list[int]:
    out: list[int] = []
    for value in values:
        if not out or out[-1] != value:
            out.append(value)
    return out


def _axis_runs(g: np.ndarray, axis: int) -> tuple[bool, list[int]]:
    colors = [int(color) for color in np.unique(g)]
    color_means: dict[int, float] = {}
    color_totals: dict[int, int] = {}
    for color in colors:
        coords_all = np.argwhere(g == color)
        coords = coords_all[:, axis]
        color_means[color] = float(coords.mean())
        color_totals[color] = int(coords_all.shape[0])

    line_majorities: list[int] = []
    for index in range(g.shape[axis]):
        line = g[index, :] if axis == 0 else g[:, index]
        values, counts = np.unique(line, return_counts=True)
        max_count = int(counts.max())
        winners = [int(value) for value, count in zip(values, counts) if int(count) == max_count]
        if len(winners) == 1:
            line_majorities.append(winners[0])
            continue

        best_total = max(color_totals[color] for color in winners)
        winners = [color for color in winners if color_totals[color] == best_total]
        if len(winners) == 1:
            line_majorities.append(winners[0])
            continue

        distances = [(abs(color_means[color] - index), color) for color in winners]
        min_distance = min(distance for distance, _ in distances)
        winners = [color for distance, color in distances if distance == min_distance]
        if len(winners) == 1:
            line_majorities.append(winners[0])
            continue

        first_pos = {color: int(np.flatnonzero(line == color)[0]) for color in winners}
        line_majorities.append(min(winners, key=lambda color: first_pos[color]))

    runs = _compress(line_majorities)
    valid = len(runs) == len(colors) and set(runs) == set(colors) and 3 <= len(runs) <= 5
    return valid, runs


def _detect_band_sequence(g: np.ndarray) -> tuple[str, list[int]]:
    row_valid, row_runs = _axis_runs(g, axis=0)
    col_valid, col_runs = _axis_runs(g, axis=1)
    if row_valid and not col_valid:
        return "horizontal", row_runs
    if col_valid and not row_valid:
        return "vertical", col_runs
    raise AssertionError("expected exactly one valid band axis")


def _band_width(g: np.ndarray) -> int:
    width = 0
    while width < g.shape[1] and np.all(g[:, width] != BACKGROUND):
        width += 1
    assert width > 0
    return width


def _rect_components(g: np.ndarray, color: int) -> list[dict]:
    mask = g == color
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[dict] = []
    h, w = g.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            rows = [rr for rr, _ in cells]
            cols = [cc for _, cc in cells]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            area = len(cells)
            rect = g[r0 : r1 + 1, c0 : c1 + 1]
            assert area == rect.size and np.all(rect == color)
            comps.append({"bbox": (r0, c0, r1, c1), "area": area})
    return comps


def _paint_gap(out: np.ndarray, rects: list[dict], seq: list[int]) -> np.ndarray:
    a, b = rects
    ar0, ac0, ar1, ac1 = a["bbox"]
    br0, bc0, br1, bc1 = b["bbox"]
    same_rows = (ar0, ar1) == (br0, br1)
    same_cols = (ac0, ac1) == (bc0, bc1)
    assert same_rows != same_cols

    if same_rows:
        left, right = sorted(rects, key=lambda rect: rect["bbox"][1])
        r0, _, r1, lc1 = left["bbox"]
        _, rc0, _, _ = right["bbox"]
        for offset, c in enumerate(range(lc1 + 1, rc0)):
            out[r0 : r1 + 1, c] = seq[offset % len(seq)]
    else:
        top, bottom = sorted(rects, key=lambda rect: rect["bbox"][0])
        _, c0, tr1, c1 = top["bbox"]
        br0, _, _, _ = bottom["bbox"]
        for offset, r in enumerate(range(tr1 + 1, br0)):
            out[r, c0 : c1 + 1] = seq[offset % len(seq)]
    return out


def _build_band_area(seq: list[int], thicknesses: list[int], cross_dim: int, *, vertical: bool) -> np.ndarray:
    axis_dim = sum(thicknesses)
    offsets = np.zeros((len(thicknesses) - 1, cross_dim), dtype=int)
    for i in range(len(thicknesses) - 1):
        cur = random.choice([-1, 0, 1])
        offsets[i, 0] = cur
        for j in range(1, cross_dim):
            cur = max(-1, min(1, cur + random.choice([-1, 0, 1])))
            offsets[i, j] = cur

    boundaries = np.cumsum(thicknesses[:-1])[:, None] + offsets
    lines = np.arange(axis_dim)[:, None, None]
    band_indices = np.count_nonzero(lines >= boundaries.T[None, :, :], axis=2)
    band = np.asarray(seq, dtype=int)[band_indices]
    return band.T.copy() if vertical else band


def main(grid: np.ndarray) -> np.ndarray:
    """Use the compressed left color-band sequence as a repeating template for the gap between two gray rectangles.

    Input structure:
    - The left side is a solid non-black band field with jagged horizontal or vertical color bands.
    - The right side contains exactly two same-sized solid gray rectangles on a black background.
    - The rectangles are aligned either side-by-side with the same row span or stacked with the same column span.

    Rule:
    - Detect the left band block as the maximal zero-free prefix of columns, then recover its unique compressed majority-color sequence.
    - Horizontal bands produce a top-to-bottom sequence; vertical bands produce a left-to-right sequence.
    - Find the two gray rectangles and fill the black gap between them, one cell-wide slice at a time, cycling through the recovered sequence from the nearer rectangle toward the farther one.
    - Keep the band block and both gray rectangles unchanged.

    Key insight:
    - The band region simultaneously encodes which colors to use, how many colors are in the cycle, and whether the gap must be filled horizontally or vertically.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    band_w = _band_width(g)
    orientation, seq = _detect_band_sequence(g[:, :band_w])

    rects = _rect_components(g, RECT_COLOR)
    assert len(rects) == 2
    assert rects[0]["area"] == rects[1]["area"]

    ar0, ac0, ar1, ac1 = rects[0]["bbox"]
    br0, bc0, br1, bc1 = rects[1]["bbox"]
    same_rows = (ar0, ar1) == (br0, br1)
    same_cols = (ac0, ac1) == (bc0, bc1)
    assert same_rows != same_cols
    if orientation == "horizontal":
        assert same_rows
    else:
        assert same_cols

    return _paint_gap(g.copy(), rects, seq)


def _expected_grid(inp: np.ndarray, seq: list[int]) -> np.ndarray:
    return _paint_gap(inp.copy(), _rect_components(inp, RECT_COLOR), seq)


def _build_horizontal_case() -> tuple[np.ndarray, np.ndarray] | None:
    k = random.randint(3, 5)
    seq = random.sample(PALETTE, k)
    thicknesses = [random.randint(3, 5) for _ in range(k)]
    height = sum(thicknesses)
    band_w = random.choice([7, 9])
    rect_h = random.randint(3, min(5, height - 2))
    rect_w = random.randint(2, 4)
    gap = k * random.choice([1, 2, 2, 3])
    sep = random.randint(1, 3)
    right_margin = random.randint(1, 3)
    width = band_w + sep + rect_w + gap + rect_w + right_margin
    if height > 30 or width > 30:
        return None

    band = _build_band_area(seq, thicknesses, band_w, vertical=False)
    try:
        orientation, detected = _detect_band_sequence(band)
    except AssertionError:
        return None
    if orientation != "horizontal" or detected != seq:
        return None

    grid = np.zeros((height, width), dtype=int)
    grid[:, :band_w] = band
    row0 = random.randint(1, height - rect_h - 1)
    left_c0 = band_w + sep
    right_c0 = left_c0 + rect_w + gap
    grid[row0 : row0 + rect_h, left_c0 : left_c0 + rect_w] = RECT_COLOR
    grid[row0 : row0 + rect_h, right_c0 : right_c0 + rect_w] = RECT_COLOR
    out = _expected_grid(grid, seq)
    return grid, out


def _build_vertical_case() -> tuple[np.ndarray, np.ndarray] | None:
    k = random.randint(3, 5)
    seq = random.sample(PALETTE, k)
    thicknesses = [random.randint(2, 4) for _ in range(k)]
    band_w = sum(thicknesses)
    rect_h = random.randint(2, 4)
    rect_w = random.randint(3, 5)
    gap = k * random.choice([1, 2, 2, 3])
    top_margin = random.randint(1, 2)
    bottom_margin = random.randint(1, 2)
    height = top_margin + rect_h + gap + rect_h + bottom_margin
    sep = random.randint(1, 3)
    extra_right = random.randint(1, 3)
    width = band_w + sep + rect_w + extra_right
    if height > 30 or width > 30:
        return None

    band = _build_band_area(seq, thicknesses, height, vertical=True)
    try:
        orientation, detected = _detect_band_sequence(band)
    except AssertionError:
        return None
    if orientation != "vertical" or detected != seq:
        return None

    grid = np.zeros((height, width), dtype=int)
    grid[:, :band_w] = band
    col0 = random.randint(band_w + sep, width - rect_w)
    top_r0 = top_margin
    bottom_r0 = top_r0 + rect_h + gap
    grid[top_r0 : top_r0 + rect_h, col0 : col0 + rect_w] = RECT_COLOR
    grid[bottom_r0 : bottom_r0 + rect_h, col0 : col0 + rect_w] = RECT_COLOR
    out = _expected_grid(grid, seq)
    return grid, out


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        built = _build_horizontal_case() if random.choice([False, True]) else _build_vertical_case()
        if built is None:
            continue
        inp, expected = built
        try:
            actual = main(inp)
        except Exception:
            continue
        if np.array_equal(actual, expected) and not np.array_equal(inp, expected):
            return inp
    raise RuntimeError("failed to generate valid input")
