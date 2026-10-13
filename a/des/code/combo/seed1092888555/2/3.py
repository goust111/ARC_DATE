from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 8
LEGEND_DOT = 0
PALETTE = tuple(range(10))
GEN_ATTEMPTS = 400
ORIENTATIONS = ("H", "V", "D1", "D2")


def _background_color(g: np.ndarray) -> int:
    counts = Counter(map(int, g.ravel()))
    return int(counts.most_common(1)[0][0])


def _parse_legend(g: np.ndarray, bg: int) -> tuple[dict[int, int], set[tuple[int, int]]]:
    legend: dict[int, int] = {}
    legend_cells: set[tuple[int, int]] = set()

    for r in range(g.shape[0]):
        nz = np.flatnonzero(g[r] != bg)
        if len(nz) == 0 or int(nz[0]) != 0:
            continue
        run_len = 1
        dot_color = int(g[r, 0])
        while run_len < g.shape[1] and int(g[r, run_len]) == dot_color:
            run_len += 1
        if not 1 <= run_len <= 5:
            continue
        expected = list(range(run_len))
        rest = nz.tolist()[run_len:]
        if nz.tolist()[:run_len] != expected or len(rest) != 1:
            continue
        square_c = int(rest[0])
        if square_c <= run_len or int(g[r, square_c]) == dot_color:
            continue
        if any(int(g[r, c]) != bg for c in range(run_len, square_c)):
            continue
        legend[run_len] = int(g[r, square_c])
        for c in expected:
            legend_cells.add((int(r), int(c)))
        legend_cells.add((int(r), square_c))

    assert legend, "expected at least one legend row"
    return legend, legend_cells


def _midpoint(a: int, b: int) -> int:
    return int((a + b + 1) // 2)


def _perpendicular_offsets(dr: int, dc: int) -> list[tuple[int, int]]:
    if dr == 0:
        return [(-1, 0), (0, 0), (1, 0)]
    if dc == 0:
        return [(0, -1), (0, 0), (0, 1)]
    if dr == dc:
        return [(-1, 1), (0, 0), (1, -1)]
    return [(-1, -1), (0, 0), (1, 1)]


def _draw_bar(out: np.ndarray, center: tuple[int, int], dr: int, dc: int, color: int) -> None:
    cr, cc = center
    for off_r, off_c in _perpendicular_offsets(dr, dc):
        rr = cr + off_r
        cc2 = cc + off_c
        if 0 <= rr < out.shape[0] and 0 <= cc2 < out.shape[1]:
            out[rr, cc2] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Identify each same-color pixel pair, infer its direction and length, and
    draw a length-coded perpendicular bar at the pair midpoint.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _background_color(g)
    legend, legend_cells = _parse_legend(g, bg)

    out = g.copy()
    for r, c in legend_cells:
        out[r, c] = bg

    remaining: dict[int, list[tuple[int, int]]] = {}
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            if (r, c) in legend_cells or int(g[r, c]) == bg:
                continue
            remaining.setdefault(int(g[r, c]), []).append((int(r), int(c)))

    assert remaining, "expected at least one color pair"
    for color, coords in remaining.items():
        assert len(coords) == 2, f"color {color} must appear exactly twice outside the legend"
        (r1, c1), (r2, c2) = coords
        drow = int(r2 - r1)
        dcol = int(c2 - c1)

        if drow == 0:
            step_r, step_c = 0, 1
            length = abs(dcol)
        elif dcol == 0:
            step_r, step_c = 1, 0
            length = abs(drow)
        else:
            assert abs(drow) == abs(dcol), "pair must be horizontal, vertical, or diagonal"
            step_r = 1 if drow > 0 else -1
            step_c = 1 if dcol > 0 else -1
            length = abs(drow)

        assert length in legend, "missing legend entry for pair length"
        mid = (_midpoint(r1, r2), _midpoint(c1, c2))
        _draw_bar(out, mid, step_r, step_c, legend[length])
        out[r1, c1] = color
        out[r2, c2] = color

    return out


def _pair_spec(orientation: str) -> tuple[tuple[int, int], list[tuple[int, int]]]:
    if orientation == "H":
        return (0, 1), [(-1, 0), (0, 0), (1, 0)]
    if orientation == "V":
        return (1, 0), [(0, -1), (0, 0), (0, 1)]
    if orientation == "D1":
        return (1, 1), [(-1, 1), (0, 0), (1, -1)]
    return (1, -1), [(-1, -1), (0, 0), (1, 1)]


def _endpoints_from_mid(mid: tuple[int, int], orientation: str, length: int) -> tuple[tuple[int, int], tuple[int, int]]:
    (dr, dc), _bar = _pair_spec(orientation)
    lead = (length + 1) // 2
    tail = length // 2
    r, c = mid
    p1 = (r - lead * dr, c - lead * dc)
    p2 = (r + tail * dr, c + tail * dc)
    return p1, p2


def _bar_cells(mid: tuple[int, int], orientation: str) -> list[tuple[int, int]]:
    _dir, offsets = _pair_spec(orientation)
    r, c = mid
    return [(r + dr, c + dc) for dr, dc in offsets]


def _choose_lengths() -> list[int]:
    count = random.randint(3, 5)
    return sorted(random.sample([1, 2, 3, 4, 5], count))


def _legend_rows(lengths: list[int]) -> list[int]:
    rows = lengths[:]
    random.shuffle(rows)
    return rows


def _safe_zone(height: int, width: int, legend_rows: int, legend_side: str) -> tuple[int, int, int, int]:
    top = 1
    bottom = height - 2
    left = 1
    right = width - 2
    if legend_side == "top":
        top = legend_rows
    else:
        bottom = height - legend_rows - 1
    return top, bottom, left, right


def _place_pair(
    occupied_in: set[tuple[int, int]],
    occupied_out: set[tuple[int, int]],
    top: int,
    bottom: int,
    left: int,
    right: int,
    orientation: str,
    length: int,
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]] | None:
    mids = [
        (r, c)
        for r in range(top, bottom + 1)
        for c in range(left, right + 1)
    ]
    random.shuffle(mids)
    for mid in mids:
        p1, p2 = _endpoints_from_mid(mid, orientation, length)
        bar = _bar_cells(mid, orientation)
        cells = [p1, p2] + bar
        if len(set(cells)) != len(cells):
            continue
        if not all(top <= r <= bottom and left <= c <= right for r, c in cells):
            continue
        if any(cell in occupied_in for cell in (p1, p2)):
            continue
        if any(cell in occupied_out for cell in cells):
            continue
        return p1, p2, mid
    return None


def _expected_new_bar_cells(inp: np.ndarray, out: np.ndarray) -> set[tuple[int, int]]:
    return {
        (int(r), int(c))
        for r in range(inp.shape[0])
        for c in range(inp.shape[1])
        if int(inp[r, c]) == BG and int(out[r, c]) != BG
    }


def _gen_attempt() -> np.ndarray | None:
    heights = list(range(12, 19))
    widths = list(range(12, 19))
    H = random.choice(heights)
    W = random.choice(widths)
    lengths = _choose_lengths()
    legend_order = _legend_rows(lengths)
    legend_side = random.choice(["top", "bottom"])
    legend_gap = random.randint(1, 3)
    square_col = max(lengths) + legend_gap
    legend_h = len(lengths)

    if square_col >= W - 1:
        return None
    top, bottom, left, right = _safe_zone(H, W, legend_h, legend_side)
    if bottom - top < 3 or right - left < 3:
        return None

    available_colors = [c for c in PALETTE if c not in (BG, LEGEND_DOT)]
    random.shuffle(available_colors)
    legend_colors = available_colors[: len(lengths)]
    pair_palette = available_colors[len(lengths) :]
    if len(pair_palette) < 2:
        return None
    legend_map = dict(zip(lengths, legend_colors))

    pair_count = random.randint(2, min(4, len(pair_palette)))
    orientation_cycle = list(ORIENTATIONS)
    random.shuffle(orientation_cycle)
    orientations = [orientation_cycle[i % len(orientation_cycle)] for i in range(pair_count)]
    random.shuffle(orientations)
    pair_lengths = [random.choice(lengths) for _ in range(pair_count)]

    grid = np.full((H, W), BG, dtype=int)
    row_base = 0 if legend_side == "top" else H - legend_h
    for idx, length in enumerate(legend_order):
        r = row_base + idx
        grid[r, :length] = LEGEND_DOT
        grid[r, square_col] = legend_map[length]

    occupied_in = {(int(r), int(c)) for r in range(H) for c in range(W) if int(grid[r, c]) != BG}
    occupied_out = set(occupied_in)
    pair_colors = random.sample(pair_palette, pair_count)

    for pair_color, orientation, length in zip(pair_colors, orientations, pair_lengths):
        placed = _place_pair(occupied_in, occupied_out, top, bottom, left, right, orientation, length)
        if placed is None:
            return None
        p1, p2, mid = placed
        grid[p1] = pair_color
        grid[p2] = pair_color
        occupied_in.add(p1)
        occupied_in.add(p2)
        occupied_out.add(p1)
        occupied_out.add(p2)
        for cell in _bar_cells(mid, orientation):
            occupied_out.add(cell)

    out = main(grid)
    new_bar_cells = _expected_new_bar_cells(grid, out)
    if len(new_bar_cells) != 3 * pair_count:
        return None
    if any(int(out[r, c]) == BG for r, c in occupied_in if (r, c) not in {(rr, cc) for rr in range(H) for cc in range(W) if grid[rr, cc] == LEGEND_DOT or grid[rr, cc] in legend_colors}):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        sample = _gen_attempt()
        if sample is not None:
            return sample
    raise RuntimeError("failed to generate a valid input")
