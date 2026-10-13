from __future__ import annotations

import itertools
import random
from collections import Counter

import numpy as np


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _channel_axes(grid: np.ndarray, bg: int) -> tuple[list[int], list[int]]:
    horizontal = [r for r in range(grid.shape[0]) if np.all(grid[r, :] != bg)]
    vertical = [c for c in range(grid.shape[1]) if np.all(grid[:, c] != bg)]
    assert horizontal and vertical, "expected at least one horizontal and one vertical channel"
    return horizontal, vertical


def _channel_color(grid: np.ndarray, horizontal: list[int], vertical: list[int]) -> int:
    samples = [int(grid[r, c]) for r in horizontal for c in range(grid.shape[1])]
    samples.extend(int(grid[r, c]) for r in range(grid.shape[0]) for c in vertical)
    return int(Counter(samples).most_common(1)[0][0])


def _row_marker_colors(grid: np.ndarray, horizontal: list[int], channel: int) -> dict[int, int]:
    markers: dict[int, int] = {}
    w = grid.shape[1]
    for r in horizontal:
        left = int(grid[r, 0])
        right = int(grid[r, w - 1])
        if left != channel and right == channel:
            markers[int(r)] = left
        elif right != channel and left == channel:
            markers[int(r)] = right
        else:
            raise AssertionError(f"horizontal channel row {r} must have exactly one edge marker")
    return markers


def _col_marker_colors(grid: np.ndarray, vertical: list[int], channel: int) -> dict[int, int]:
    markers: dict[int, int] = {}
    h = grid.shape[0]
    for c in vertical:
        top = int(grid[0, c])
        bottom = int(grid[h - 1, c])
        if top != channel and bottom == channel:
            markers[int(c)] = top
        elif bottom != channel and top == channel:
            markers[int(c)] = bottom
        else:
            raise AssertionError(f"vertical channel col {c} must have exactly one edge marker")
    return markers


def _corner_candidates(shape: tuple[int, int], n: int) -> list[tuple[int, int]]:
    h, w = shape
    return [(0, 0), (0, w - n), (h - n, 0), (h - n, w - n)]


def _find_legend(
    grid: np.ndarray,
    bg: int,
    horizontal: list[int],
    vertical: list[int],
    n: int,
) -> np.ndarray:
    h_rows = set(horizontal)
    v_cols = set(vertical)
    candidates: list[np.ndarray] = []
    for r0, c0 in _corner_candidates(grid.shape, n):
        rows = range(r0, r0 + n)
        cols = range(c0, c0 + n)
        if any(r in h_rows for r in rows) or any(c in v_cols for c in cols):
            continue
        block = grid[r0 : r0 + n, c0 : c0 + n]
        if block.shape != (n, n):
            continue
        if np.all(block != bg):
            candidates.append(block.copy())
    assert len(candidates) == 1, f"expected exactly one corner legend, got {len(candidates)}"
    return candidates[0]


def _consistent_orders(
    legend: np.ndarray,
    row_colors: list[int],
    col_colors: list[int],
    clues: list[tuple[int, int, int]],
    max_solutions: int | None = None,
) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    solutions: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    for row_order in itertools.permutations(row_colors):
        row_index = {color: i for i, color in enumerate(row_order)}
        for col_order in itertools.permutations(col_colors):
            col_index = {color: i for i, color in enumerate(col_order)}
            ok = True
            for row_color, col_color, mix in clues:
                if int(legend[row_index[row_color], col_index[col_color]]) != int(mix):
                    ok = False
                    break
            if ok:
                solutions.append((row_order, col_order))
                if max_solutions is not None and len(solutions) >= max_solutions:
                    return solutions
    return solutions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each channel crossing with the mixed color decoded from the corner legend.

    Input structure:
    - bg is black. Several single-cell-wide horizontal grey channels span the full grid width, and several vertical grey channels span the full grid height.
    - Each horizontal channel has exactly one colored marker on its left or right edge; each vertical channel has exactly one colored marker on its top or bottom edge.
    - One corner contains a solid N×N color legend whose rows and columns correspond to the marker colors, but that row/column order is not labeled.
    - Some channel crossings are already prefilled with non-grey mixed colors as calibration clues.

    Rule:
    - Read each horizontal channel's marker color A and each vertical channel's marker color B.
    - Locate the corner legend and use the prefilled crossings to infer which legend row belongs to each horizontal marker color and which legend column belongs to each vertical marker color.
    - For every horizontal/vertical crossing, look up the corresponding legend cell and paint that crossing with the looked-up color.
    - Leave the non-crossing channel cells, the edge markers, and the legend unchanged.

    Key insight:
    - The corner block is a lookup table with hidden row and column labels; the already colored crossings reveal the permutation needed to decode that table.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _most_common_color(g)
    horizontal, vertical = _channel_axes(g, bg)
    channel = _channel_color(g, horizontal, vertical)

    row_markers = _row_marker_colors(g, horizontal, channel)
    col_markers = _col_marker_colors(g, vertical, channel)
    row_colors = sorted({int(color) for color in row_markers.values()})
    col_colors = sorted({int(color) for color in col_markers.values()})
    assert len(row_colors) == len(col_colors), "legend must be square"

    legend = _find_legend(g, bg, horizontal, vertical, len(row_colors))

    clues = []
    for r in horizontal:
        for c in vertical:
            value = int(g[r, c])
            if value != channel:
                clues.append((row_markers[r], col_markers[c], value))
    assert clues, "expected at least one prefilled crossing"

    matches = _consistent_orders(legend, row_colors, col_colors, clues, max_solutions=2)
    assert len(matches) == 1, "prefilled crossings must determine a unique legend decoding"
    row_order, col_order = matches[0]
    row_index = {color: i for i, color in enumerate(row_order)}
    col_index = {color: i for i, color in enumerate(col_order)}

    out = g.copy()
    for r in horizontal:
        for c in vertical:
            out[r, c] = int(legend[row_index[row_markers[r]], col_index[col_markers[c]]])
    return out


BG = 0
CHANNEL = 5
EDGE_SIDES = ("left", "right")
VERT_SIDES = ("top", "bottom")
LEGEND_CORNERS = ("tl", "tr", "bl", "br")
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
GRID_SIDE_RANGE = (20, 30)
SPACING = 2
GEN_ATTEMPTS = 400
LAYOUT_ATTEMPTS = 120
LEGEND_ATTEMPTS = 120
CLUE_ATTEMPTS = 240


def _spaced_positions(size: int, forbidden: set[int], count: int, min_gap: int = SPACING) -> list[int] | None:
    candidates = [i for i in range(1, size - 1) if i not in forbidden]
    if len(candidates) < count:
        return None
    for _ in range(LAYOUT_ATTEMPTS):
        order = candidates[:]
        random.shuffle(order)
        chosen: list[int] = []
        for pos in order:
            if all(abs(pos - prev) >= min_gap for prev in chosen):
                chosen.append(pos)
                if len(chosen) == count:
                    return sorted(chosen)
    return None


def _assign_colors(count: int, palette: list[int]) -> list[int]:
    colors = palette[:]
    while len(colors) < count:
        colors.append(random.choice(palette))
    random.shuffle(colors)
    return colors[:count]


def _legend_block(n: int, mix_colors: list[int]) -> np.ndarray | None:
    for _ in range(LEGEND_ATTEMPTS):
        legend = np.array(
            [[random.choice(mix_colors) for _ in range(n)] for _ in range(n)],
            dtype=int,
        )
        if len(set(map(int, legend.ravel()))) < min(2, len(mix_colors)):
            continue
        if len({tuple(map(int, row)) for row in legend.tolist()}) != n:
            continue
        if len({tuple(map(int, legend[:, c])) for c in range(n)}) != n:
            continue
        if any(len(set(map(int, row))) == 1 for row in legend):
            continue
        if any(len(set(map(int, legend[:, c]))) == 1 for c in range(n)):
            continue
        return legend
    return None


def _legend_origin(h: int, w: int, n: int, corner: str) -> tuple[int, int]:
    if corner == "tl":
        return 0, 0
    if corner == "tr":
        return 0, w - n
    if corner == "bl":
        return h - n, 0
    if corner == "br":
        return h - n, w - n
    raise ValueError(corner)


def _reserved_rows(h: int, n: int, corner: str) -> set[int]:
    if corner in ("tl", "tr"):
        return set(range(n))
    return set(range(h - n, h))


def _reserved_cols(w: int, n: int, corner: str) -> set[int]:
    if corner in ("tl", "bl"):
        return set(range(n))
    return set(range(w - n, w))


def _mix_lookup(
    legend: np.ndarray,
    row_order: tuple[int, ...],
    col_order: tuple[int, ...],
    row_color: int,
    col_color: int,
) -> int:
    return int(legend[row_order.index(row_color), col_order.index(col_color)])


def _choose_clue_cells(
    legend: np.ndarray,
    row_colors: list[int],
    col_colors: list[int],
    full_clues: list[tuple[int, int, int, int, int]],
) -> list[tuple[int, int]]:
    total = len(full_clues)
    min_count = max(3, len(row_colors) + 1)
    max_count = min(total - 1, len(row_colors) + 4)
    if min_count > max_count:
        return []

    for _ in range(CLUE_ATTEMPTS):
        count = random.randint(min_count, max_count)
        picks = random.sample(full_clues, count)
        clue_facts = [(rc, cc, mix) for _r, _c, rc, cc, mix in picks]
        matches = _consistent_orders(legend, row_colors, col_colors, clue_facts, max_solutions=2)
        if len(matches) == 1:
            return [(int(r), int(c)) for r, c, _rc, _cc, _mix in picks]
    return []


def _gen_attempt() -> np.ndarray | None:
    n = random.choice([3, 4])
    marker_colors = random.sample(PALETTE, n)
    remaining = [color for color in PALETTE if color not in marker_colors]
    mix_count = random.randint(3, min(4, len(remaining)))
    mix_colors = random.sample(remaining, mix_count)
    legend = _legend_block(n, mix_colors)
    if legend is None:
        return None

    row_order = tuple(random.sample(marker_colors, n))
    col_order = tuple(random.sample(marker_colors, n))
    h_count = random.randint(n, n + 1)
    v_count = random.randint(n, n + 1)

    for _ in range(LAYOUT_ATTEMPTS):
        h = random.randint(*GRID_SIDE_RANGE)
        w = random.randint(*GRID_SIDE_RANGE)
        corner = random.choice(LEGEND_CORNERS)

        rows = _spaced_positions(h, _reserved_rows(h, n, corner), h_count)
        cols = _spaced_positions(w, _reserved_cols(w, n, corner), v_count)
        if rows is None or cols is None:
            continue

        grid = np.full((h, w), BG, dtype=int)
        lr, lc = _legend_origin(h, w, n, corner)
        grid[lr : lr + n, lc : lc + n] = legend

        row_markers = dict(zip(rows, _assign_colors(h_count, marker_colors)))
        col_markers = dict(zip(cols, _assign_colors(v_count, marker_colors)))
        row_sides = {r: random.choice(EDGE_SIDES) for r in rows}
        col_sides = {c: random.choice(VERT_SIDES) for c in cols}

        for r in rows:
            grid[r, :] = CHANNEL
        for c in cols:
            grid[:, c] = CHANNEL

        for r in rows:
            if row_sides[r] == "left":
                grid[r, 0] = row_markers[r]
            else:
                grid[r, w - 1] = row_markers[r]
        for c in cols:
            if col_sides[c] == "top":
                grid[0, c] = col_markers[c]
            else:
                grid[h - 1, c] = col_markers[c]

        output = grid.copy()
        full_clues: list[tuple[int, int, int, int, int]] = []
        for r in rows:
            for c in cols:
                mix = _mix_lookup(legend, row_order, col_order, row_markers[r], col_markers[c])
                output[r, c] = mix
                full_clues.append((r, c, row_markers[r], col_markers[c], mix))

        clue_cells = _choose_clue_cells(legend, sorted(marker_colors), sorted(marker_colors), full_clues)
        if not clue_cells:
            continue

        inp = grid.copy()
        for r, c in clue_cells:
            inp[r, c] = output[r, c]

        try:
            if np.array_equal(main(inp), output):
                return inp
        except AssertionError:
            continue
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid channel-mixing input")
