from __future__ import annotations

import numpy as np


BG = 5
BLACK = 0
RED = 2
YELLOW = 4
ARROW_COLORS = (1, 3, 6, 7, 8, 9)
CARDINALS = ((-1, 0), (1, 0), (0, -1), (0, 1))
ARROW_FROM_TIP = {
    (-1, 0): ((0, 0), (1, 0), (2, 0), (0, -1), (0, 1)),
    (1, 0): ((0, 0), (-1, 0), (-2, 0), (0, -1), (0, 1)),
    (0, -1): ((0, 0), (0, 1), (0, 2), (-1, 0), (1, 0)),
    (0, 1): ((0, 0), (0, -1), (0, -2), (-1, 0), (1, 0)),
}
GENERATE_RETRIES = 400
MIN_GRID_SIZE = 16
MAX_GRID_SIZE = 30
MIN_SPACING = 3


def _components_by_color(grid: np.ndarray, excluded: set[int]) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []

    for r0 in range(h):
        for c0 in range(w):
            color = int(grid[r0, c0])
            if seen[r0, c0] or color in excluded:
                continue

            stack = [(r0, c0)]
            seen[r0, c0] = True
            cells: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for dr, dc in CARDINALS:
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            comps.append({"color": color, "cells": cells})

    return comps


def _detect_arrow(cells: list[tuple[int, int]]) -> dict[str, object] | None:
    cell_set = set(cells)
    if len(cell_set) != 5:
        return None

    degrees = {
        cell: sum((cell[0] + dr, cell[1] + dc) in cell_set for dr, dc in CARDINALS)
        for cell in cell_set
    }
    tips = [cell for cell, degree in degrees.items() if int(degree) == 3]
    mids = [cell for cell, degree in degrees.items() if int(degree) == 2]
    leaves = [cell for cell, degree in degrees.items() if int(degree) == 1]
    if len(tips) != 1 or len(mids) != 1 or len(leaves) != 3:
        return None

    tip = tips[0]
    for dr, dc in CARDINALS:
        mid = (tip[0] - dr, tip[1] - dc)
        tail = (tip[0] - 2 * dr, tip[1] - 2 * dc)
        side_a = (tip[0] + dc, tip[1] - dr)
        side_b = (tip[0] - dc, tip[1] + dr)
        if {mid, tail, side_a, side_b} == cell_set - {tip} and mid == mids[0]:
            return {
                "tip": tip,
                "direction": (dr, dc),
                "cells": tuple(sorted(cell_set)),
            }

    return None


def _extract_arrows(grid: np.ndarray) -> tuple[list[dict[str, object]], np.ndarray]:
    arrows: list[dict[str, object]] = []
    arrow_mask = np.zeros(grid.shape, dtype=bool)

    for comp in _components_by_color(grid, excluded={BG, BLACK}):
        arrow = _detect_arrow(comp["cells"])
        assert arrow is not None, "every non-grey, non-black component must be a 5-cell arrow"
        arrow["color"] = int(comp["color"])
        arrows.append(arrow)
        for r, c in arrow["cells"]:
            arrow_mask[r, c] = True

    assert len(arrows) >= 2, "expected at least two arrows"
    colors = [int(arrow["color"]) for arrow in arrows]
    assert len(colors) == len(set(colors)), "each arrow must use a distinct color"
    return arrows, arrow_mask


def _ray_counts_from_arrows(shape: tuple[int, int], arrows: list[dict[str, object]]) -> np.ndarray:
    h, w = shape
    counts = np.zeros((h, w), dtype=int)
    for arrow in arrows:
        r, c = arrow["tip"]
        dr, dc = arrow["direction"]
        r += dr
        c += dc
        while 0 <= r < h and 0 <= c < w:
            counts[r, c] += 1
            r += dr
            c += dc
    return counts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each arrow's aim line and mark every multi-ray treasure intersection.

    Input structure:
    - The background is Grey.
    - Several disconnected 5-cell arrows of distinct colors are scattered in the grid.
    - Black singleton dots may appear on background cells, including on future ray paths.

    Rule:
    - Detect each arrow from its T-shaped 5-cell geometry and infer its cardinal pointing direction.
    - From the front tip cell of every arrow, continue a straight ray beyond the arrow to the grid edge.
    - Any non-arrow cell crossed by two or more rays becomes Red, except a Black dot at such a cell becomes Yellow.

    Key insight:
    - Solving requires two separate inferences: first identify the tip and direction of each 5-cell arrow, then intersect the extended axis-aligned rays while preserving the original arrow sprites.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30

    arrows, arrow_mask = _extract_arrows(g)
    counts = _ray_counts_from_arrows(g.shape, arrows)

    out = g.copy()
    intersections = (counts >= 2) & ~arrow_mask
    yellow_mask = intersections & (g == BLACK)
    red_mask = intersections & ~yellow_mask
    out[red_mask] = RED
    out[yellow_mask] = YELLOW
    return out


def _arrow_cells_from_tip(tip: tuple[int, int], direction: tuple[int, int]) -> list[tuple[int, int]]:
    tr, tc = tip
    return [(tr + dr, tc + dc) for dr, dc in ARROW_FROM_TIP[direction]]


def _sample_spaced_positions(
    lo: int,
    hi: int,
    count: int,
    gap: int,
    rng: np.random.Generator,
) -> list[int] | None:
    if count <= 0 or lo > hi:
        return None

    span = hi - lo + 1
    if count + (count - 1) * (gap - 1) > span:
        return None

    base_hi = hi - (count - 1) * (gap - 1)
    base = sorted(int(x) for x in rng.choice(np.arange(lo, base_hi + 1), size=count, replace=False))
    return [value + idx * (gap - 1) for idx, value in enumerate(base)]


def _max_spaced_count(lo: int, hi: int, gap: int) -> int:
    if lo > hi:
        return 0
    span = hi - lo + 1
    return (span + gap - 1) // gap


def _place_arrow(grid: np.ndarray, tip: tuple[int, int], direction: tuple[int, int], color: int) -> set[tuple[int, int]]:
    cells = _arrow_cells_from_tip(tip, direction)
    h, w = grid.shape
    for r, c in cells:
        assert 0 <= r < h and 0 <= c < w
        assert int(grid[r, c]) == BG
        grid[r, c] = int(color)
    return set(cells)


def _black_can_use(
    cell: tuple[int, int],
    shape: tuple[int, int],
    arrow_cells: set[tuple[int, int]],
    black_cells: set[tuple[int, int]],
) -> bool:
    r, c = cell
    h, w = shape
    if not (0 <= r < h and 0 <= c < w):
        return False
    if cell in arrow_cells or cell in black_cells:
        return False
    for dr, dc in CARDINALS:
        nbr = (r + dr, c + dc)
        if nbr in arrow_cells or nbr in black_cells:
            return False
    return True


def _pick_isolated_cells(
    candidates: list[tuple[int, int]],
    want: int,
    shape: tuple[int, int],
    arrow_cells: set[tuple[int, int]],
    black_cells: set[tuple[int, int]],
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    picked: list[tuple[int, int]] = []
    for idx in rng.permutation(len(candidates)):
        if len(picked) >= want:
            break
        cell = candidates[int(idx)]
        current_black = black_cells | set(picked)
        if _black_can_use(cell, shape, arrow_cells, current_black):
            picked.append(cell)
    return picked


def _black_intersection_subset(
    intersections: list[tuple[int, int]],
    rng: np.random.Generator,
) -> set[tuple[int, int]]:
    if not intersections:
        return set()

    mode = int(rng.integers(0, 3))
    if mode == 0:
        count = 0
    elif mode == 1:
        if len(intersections) == 1:
            count = 1
        else:
            count = int(rng.integers(1, len(intersections)))
    else:
        count = len(intersections)

    if count == 0:
        return set()
    indices = rng.choice(len(intersections), size=count, replace=False)
    return {intersections[int(idx)] for idx in np.atleast_1d(indices)}


def _attempt_generate(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(MIN_GRID_SIZE, MAX_GRID_SIZE + 1))
    w = int(rng.integers(MIN_GRID_SIZE, MAX_GRID_SIZE + 1))
    h_dir = (0, 1) if int(rng.integers(0, 2)) == 0 else (0, -1)
    v_dir = (1, 0) if int(rng.integers(0, 2)) == 0 else (-1, 0)

    row_lo, row_hi = (6, h - 2) if v_dir == (1, 0) else (1, h - 7)
    col_lo, col_hi = (6, w - 2) if h_dir == (0, 1) else (1, w - 7)
    max_h_arrows = min(3, _max_spaced_count(row_lo, row_hi, MIN_SPACING))
    max_v_arrows = min(3, _max_spaced_count(col_lo, col_hi, MIN_SPACING))
    if max_h_arrows == 0 or max_v_arrows == 0:
        return None

    h_count = int(rng.integers(1, max_h_arrows + 1))
    v_count = int(rng.integers(1, max_v_arrows + 1))
    rows = _sample_spaced_positions(row_lo, row_hi, h_count, MIN_SPACING, rng)
    cols = _sample_spaced_positions(col_lo, col_hi, v_count, MIN_SPACING, rng)
    if rows is None or cols is None:
        return None

    if h_dir == (0, 1):
        h_tips = [int(rng.integers(2, 5)) for _ in range(h_count)]
    else:
        h_tips = [int(rng.integers(w - 5, w - 2)) for _ in range(h_count)]

    if v_dir == (1, 0):
        v_tips = [int(rng.integers(2, 5)) for _ in range(v_count)]
    else:
        v_tips = [int(rng.integers(h - 5, h - 2)) for _ in range(v_count)]

    colors = [int(x) for x in rng.choice(np.array(ARROW_COLORS), size=h_count + v_count, replace=False)]
    grid = np.full((h, w), BG, dtype=int)
    arrow_cells: set[tuple[int, int]] = set()
    specs: list[dict[str, object]] = []

    for idx, row in enumerate(rows):
        spec = {"tip": (int(row), int(h_tips[idx])), "direction": h_dir, "color": colors[idx]}
        cells = _arrow_cells_from_tip(spec["tip"], spec["direction"])
        if any(cell in arrow_cells for cell in cells):
            return None
        arrow_cells |= _place_arrow(grid, spec["tip"], spec["direction"], spec["color"])
        specs.append(spec)

    for idx, col in enumerate(cols):
        spec = {
            "tip": (int(v_tips[idx]), int(col)),
            "direction": v_dir,
            "color": colors[h_count + idx],
        }
        cells = _arrow_cells_from_tip(spec["tip"], spec["direction"])
        if any(cell in arrow_cells for cell in cells):
            return None
        arrow_cells |= _place_arrow(grid, spec["tip"], spec["direction"], spec["color"])
        specs.append(spec)

    counts = _ray_counts_from_arrows(grid.shape, specs)
    intersections = [
        (int(r), int(c))
        for r, c in np.argwhere(counts >= 2)
        if (int(r), int(c)) not in arrow_cells
    ]
    if not intersections:
        return None

    black_cells = _black_intersection_subset(intersections, rng)

    single_ray_candidates = [
        (int(r), int(c))
        for r, c in np.argwhere((counts == 1) & (grid == BG))
    ]
    off_ray_candidates = [
        (int(r), int(c))
        for r, c in np.argwhere((counts == 0) & (grid == BG))
    ]
    rng.shuffle(single_ray_candidates)
    rng.shuffle(off_ray_candidates)

    want_single = int(rng.integers(0, min(4, len(single_ray_candidates)) + 1))
    want_off = int(rng.integers(0, 4))
    black_cells |= set(
        _pick_isolated_cells(single_ray_candidates, want_single, grid.shape, arrow_cells, black_cells, rng)
    )
    black_cells |= set(
        _pick_isolated_cells(off_ray_candidates, want_off, grid.shape, arrow_cells, black_cells, rng)
    )

    if not black_cells:
        fallback = _pick_isolated_cells(off_ray_candidates + single_ray_candidates, 1, grid.shape, arrow_cells, black_cells, rng)
        if not fallback:
            return None
        black_cells |= set(fallback)

    for r, c in black_cells:
        if not _black_can_use((r, c), grid.shape, arrow_cells, black_cells - {(r, c)}):
            return None
        grid[r, c] = BLACK

    out = main(grid)
    if out.shape != grid.shape:
        return None

    if not np.any((out == RED) | (out == YELLOW)):
        return None

    black_positions = np.argwhere(grid == BLACK)
    for r, c in black_positions:
        neighbors = 0
        for dr, dc in CARDINALS:
            nr = int(r) + dr
            nc = int(c) + dc
            if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) == BLACK:
                neighbors += 1
        if neighbors != 0:
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        rng = np.random.default_rng()
        grid = _attempt_generate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid arrow-ray input")
