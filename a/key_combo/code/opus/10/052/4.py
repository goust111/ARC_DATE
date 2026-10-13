from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 5
DIVIDER = 6
DATA = 7
LEGEND_PALETTE = (1, 2, 3, 4, 8, 9)
GEN_RETRIES = 400
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

BASE_PATTERNS = {
    2: [
        ((0, 0), (0, 1)),
    ],
    3: [
        ((0, 0), (0, 1), (0, 2)),
        ((0, 0), (1, 0), (1, 1)),
    ],
    4: [
        ((0, 0), (0, 1), (1, 0), (1, 1)),
        ((0, 0), (1, 0), (2, 0), (2, 1)),
        ((0, 0), (0, 1), (0, 2), (1, 1)),
        ((0, 0), (0, 1), (0, 2), (0, 3)),
    ],
    5: [
        ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
        ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2)),
        ((0, 0), (0, 1), (0, 2), (1, 1), (2, 1)),
        ((0, 0), (0, 1), (0, 2), (0, 3), (0, 4)),
    ],
}


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in DIRS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _find_divider_axes(grid: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    h, w = grid.shape
    divider_rows = [r for r in range(h) if np.all(grid[r, :] == DIVIDER)]
    divider_cols = [c for c in range(w) if np.all(grid[:, c] == DIVIDER)]
    assert len(divider_rows) == 2 and divider_rows[1] == divider_rows[0] + 1
    assert len(divider_cols) == 2 and divider_cols[1] == divider_cols[0] + 1
    return (divider_rows[0], divider_rows[1]), (divider_cols[0], divider_cols[1])


def _find_legend_corner(grid: np.ndarray) -> tuple[int, int]:
    h, w = grid.shape
    candidates = []
    for r0, c0 in ((0, 0), (0, w - 2), (h - 2, 0), (h - 2, w - 2)):
        block = grid[r0 : r0 + 2, c0 : c0 + 2]
        if block.shape != (2, 2):
            continue
        if np.all((block != BG) & (block != DIVIDER) & (block != DATA)):
            candidates.append((r0, c0))
    assert len(candidates) == 1
    return candidates[0]


def _quadrant_slices(
    row_pair: tuple[int, int], col_pair: tuple[int, int]
) -> dict[tuple[int, int], tuple[slice, slice]]:
    top = slice(0, row_pair[0])
    bottom = slice(row_pair[1] + 1, None)
    left = slice(0, col_pair[0])
    right = slice(col_pair[1] + 1, None)
    return {
        (0, 0): (top, left),
        (0, 1): (top, right),
        (1, 0): (bottom, left),
        (1, 1): (bottom, right),
    }


def _legend_quadrant(legend_corner: tuple[int, int], grid_shape: tuple[int, int]) -> tuple[int, int]:
    h, w = grid_shape
    r0, c0 = legend_corner
    return (0 if r0 == 0 else 1, 0 if c0 == 0 else 1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor quadrant pixels by the 2x2 legend, but erase isolated single-pixel noise first.

    Input structure:
    - The grid has grey background, one 2-pixel-wide magenta horizontal divider and one
      2-pixel-wide magenta vertical divider, splitting the board into four equal quadrants.
    - A 2x2 corner legend contains four distinct colors, one per quadrant key cell.
    - Orange data pixels appear inside the quadrants as 4-connected groups plus isolated
      singleton noise pixels; the legend corner itself stays separate from the data.

    Rule:
    - Use the cross divider to determine the top/bottom and left/right quadrant of every
      orange pixel.
    - Inside each quadrant, compute 4-connected components of orange pixels.
    - Recolor every component with area greater than one to that quadrant's legend color,
      erase every singleton component to grey, and leave legend cells and dividers unchanged.

    Key insight:
    - The output for each orange pixel depends on two independent tests: which quadrant it
      lies in determines the replacement color, and component size determines whether it is
      kept or erased at all.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    row_pair, col_pair = _find_divider_axes(g)
    quadrants = _quadrant_slices(row_pair, col_pair)
    legend_corner = _find_legend_corner(g)
    legend = g[legend_corner[0] : legend_corner[0] + 2, legend_corner[1] : legend_corner[1] + 2]

    out = g.copy()
    for qr in range(2):
        for qc in range(2):
            rs, cs = quadrants[(qr, qc)]
            sub = g[rs, cs]
            for comp in _components4(sub == DATA):
                if len(comp) == 1:
                    rr, cc = comp[0]
                    out[rs.start + rr, cs.start + cc] = BG
                else:
                    color = int(legend[qr, qc])
                    for rr, cc in comp:
                        out[rs.start + rr, cs.start + cc] = color
    return out


def _canon_shape(cells: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def _shape_variants(base: tuple[tuple[int, int], ...]) -> list[tuple[tuple[int, int], ...]]:
    cells = set(base)
    variants = set()
    for rot in range(4):
        rotated = set(cells)
        for _ in range(rot):
            rotated = {(-c, r) for r, c in rotated}
        for flip in (False, True):
            transformed = {(-r, c) for r, c in rotated} if flip else set(rotated)
            variants.add(_canon_shape(transformed))
    return list(variants)


PATTERN_VARIANTS = {
    size: [variant for base in bases for variant in _shape_variants(base)]
    for size, bases in BASE_PATTERNS.items()
}


def _corner_reserved_cells(h: int, w: int) -> set[tuple[int, int]]:
    reserved = set()
    for r0, c0 in ((0, 0), (0, w - 2), (h - 2, 0), (h - 2, w - 2)):
        for dr in range(2):
            for dc in range(2):
                reserved.add((r0 + dr, c0 + dc))
    return reserved


def _grid_quadrants(h: int, w: int) -> dict[tuple[int, int], tuple[slice, slice]]:
    qh = (h - 2) // 2
    qw = (w - 2) // 2
    return {
        (0, 0): (slice(0, qh), slice(0, qw)),
        (0, 1): (slice(0, qh), slice(qw + 2, w)),
        (1, 0): (slice(qh + 2, h), slice(0, qw)),
        (1, 1): (slice(qh + 2, h), slice(qw + 2, w)),
    }


def _place_singleton(
    occupied: set[tuple[int, int]],
    allowed: set[tuple[int, int]],
) -> tuple[int, int] | None:
    candidates = []
    for cell in allowed:
        if cell in occupied:
            continue
        ok = True
        for dr, dc in DIRS4:
            nbr = (cell[0] + dr, cell[1] + dc)
            if nbr in occupied:
                ok = False
                break
        if ok:
            candidates.append(cell)
    return random.choice(candidates) if candidates else None


def _place_pattern(
    occupied: set[tuple[int, int]],
    allowed: set[tuple[int, int]],
    rs: slice,
    cs: slice,
    size: int,
) -> list[tuple[int, int]] | None:
    variant_list = random.sample(PATTERN_VARIANTS[size], len(PATTERN_VARIANTS[size]))
    row_start = rs.start
    row_stop = rs.stop
    col_start = cs.start
    col_stop = cs.stop
    for shape in variant_list:
        local_rows = [r for r, _ in shape]
        local_cols = [c for _, c in shape]
        shape_h = max(local_rows) + 1
        shape_w = max(local_cols) + 1
        anchors = [
            (r0, c0)
            for r0 in range(row_start, row_stop - shape_h + 1)
            for c0 in range(col_start, col_stop - shape_w + 1)
        ]
        random.shuffle(anchors)
        for r0, c0 in anchors:
            cells = [(r0 + r, c0 + c) for r, c in shape]
            if any(cell not in allowed or cell in occupied for cell in cells):
                continue
            cell_set = set(cells)
            ok = True
            for rr, cc in cells:
                for dr, dc in DIRS4:
                    nbr = (rr + dr, cc + dc)
                    if nbr in occupied and nbr not in cell_set:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                return cells
    return None


def _build_expected(
    inp: np.ndarray,
    legend: np.ndarray,
    placed: dict[tuple[int, int], list[list[tuple[int, int]]]],
) -> np.ndarray:
    out = inp.copy()
    for qr in range(2):
        for qc in range(2):
            color = int(legend[qr, qc])
            for comp in placed[(qr, qc)]:
                if len(comp) == 1:
                    r, c = comp[0]
                    out[r, c] = BG
                else:
                    for r, c in comp:
                        out[r, c] = color
    return out


def _validate_input(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert g.ndim == 2 and h <= 30 and w <= 30
    assert set(map(int, np.unique(g))).issubset(set(LEGEND_PALETTE) | {BG, DIVIDER, DATA})
    assert Counter(map(int, g.ravel()))[BG] > Counter(map(int, g.ravel()))[DIVIDER]

    row_pair, col_pair = _find_divider_axes(g)
    assert row_pair[0] > 0 and row_pair[1] < h - 1
    assert col_pair[0] > 0 and col_pair[1] < w - 1
    assert row_pair[0] == (h - 2) // 2 and col_pair[0] == (w - 2) // 2

    legend_corner = _find_legend_corner(g)
    legend = g[legend_corner[0] : legend_corner[0] + 2, legend_corner[1] : legend_corner[1] + 2]
    assert len(set(map(int, legend.ravel()))) == 4

    quads = _quadrant_slices(row_pair, col_pair)
    legend_quad = _legend_quadrant(legend_corner, g.shape)
    fg_total = 0
    comp_summary: dict[tuple[int, int], list[int]] = {}
    for key, (rs, cs) in quads.items():
        sub = g[rs, cs]
        comps = _components4(sub == DATA)
        sizes = sorted(len(comp) for comp in comps)
        assert any(size == 1 for size in sizes)
        assert any(size > 1 for size in sizes)
        assert all(size <= 5 for size in sizes)
        assert sum(size == 1 for size in sizes) in (1, 2)
        assert 1 <= sum(size > 1 for size in sizes) <= 2
        fg_total += sum(sizes)
        comp_summary[key] = sizes
        if key == legend_quad:
            corner_block = sub[:2, :2] if legend_corner == (0, 0) else None
            if legend_corner == (0, w - 2):
                corner_block = sub[:2, -2:]
            elif legend_corner == (h - 2, 0):
                corner_block = sub[-2:, :2]
            elif legend_corner == (h - 2, w - 2):
                corner_block = sub[-2:, -2:]
            assert corner_block is not None
            assert np.all((corner_block != BG) & (corner_block != DIVIDER) & (corner_block != DATA))

    assert fg_total == int(np.count_nonzero(g == DATA))
    return {
        "legend": legend,
        "legend_corner": legend_corner,
        "component_sizes": comp_summary,
    }


def _gen_attempt() -> np.ndarray | None:
    qh = random.randint(5, 9)
    qw = random.randint(5, 9)
    h = 2 * qh + 2
    w = 2 * qw + 2
    legend_corner = random.choice(((0, 0), (0, w - 2), (h - 2, 0), (h - 2, w - 2)))
    legend_quad = _legend_quadrant(legend_corner, (h, w))
    legend = np.array(random.sample(LEGEND_PALETTE, 4), dtype=int).reshape(2, 2)

    grid = np.full((h, w), BG, dtype=int)
    grid[qh : qh + 2, :] = DIVIDER
    grid[:, qw : qw + 2] = DIVIDER
    grid[legend_corner[0] : legend_corner[0] + 2, legend_corner[1] : legend_corner[1] + 2] = legend

    quadrants = _grid_quadrants(h, w)
    reserved = _corner_reserved_cells(h, w)
    placed: dict[tuple[int, int], list[list[tuple[int, int]]]] = {}

    for key, (rs, cs) in quadrants.items():
        allowed = {
            (r, c)
            for r in range(rs.start, rs.stop)
            for c in range(cs.start, cs.stop)
            if (r, c) not in reserved
        }
        if key == legend_quad:
            for r in range(legend_corner[0], legend_corner[0] + 2):
                for c in range(legend_corner[1], legend_corner[1] + 2):
                    allowed.discard((r, c))
        occupied: set[tuple[int, int]] = set()
        comps: list[list[tuple[int, int]]] = []

        multi_count = random.randint(1, 2)
        noise_count = random.randint(1, 2)
        multi_sizes = [random.randint(2, 5) for _ in range(multi_count)]
        if multi_count == 2 and sum(multi_sizes) > max(6, (len(allowed) // 3)):
            return None

        for size in multi_sizes:
            cells = _place_pattern(occupied, allowed, rs, cs, size)
            if cells is None:
                return None
            for cell in cells:
                occupied.add(cell)
            comps.append(cells)

        for _ in range(noise_count):
            cell = _place_singleton(occupied, allowed)
            if cell is None:
                return None
            occupied.add(cell)
            comps.append([cell])

        if not any(len(comp) == 1 for comp in comps) or not any(len(comp) > 1 for comp in comps):
            return None
        placed[key] = comps
        for comp in comps:
            for r, c in comp:
                grid[r, c] = DATA

    expected = _build_expected(grid, legend, placed)
    info = _validate_input(grid)
    if not np.array_equal(info["legend"], legend):
        return None
    if np.array_equal(expected, grid):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
