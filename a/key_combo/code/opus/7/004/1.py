from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, components_mask, in_bounds


def _find_guide_row(grid: np.ndarray) -> int:
    rows = np.flatnonzero(np.any(grid == Color.RED, axis=1))
    assert len(rows) == 1, "expected exactly one horizontal red guide row"
    return int(rows[0])


def _find_pivot(grid: np.ndarray) -> tuple[int, int]:
    coords = np.argwhere(grid == Color.GREY)
    assert coords.shape == (1, 2), "expected exactly one grey pivot pixel"
    return int(coords[0, 0]), int(coords[0, 1])


def _blue_coords(grid: np.ndarray) -> list[tuple[int, int]]:
    return [tuple(map(int, rc)) for rc in np.argwhere(grid == Color.BLUE)]


def _apply_rule(grid: np.ndarray) -> np.ndarray:
    guide_r = _find_guide_row(grid)
    pivot_r, pivot_c = _find_pivot(grid)

    out = np.array(grid, dtype=int, copy=True)
    out[out == Color.BLUE] = Color.BLACK

    for r, c in _blue_coords(grid):
        rr = 2 * pivot_r - r
        cc = 2 * pivot_c - c
        assert in_bounds(grid.shape, rr, cc), "reflected coordinate must stay in bounds"
        assert rr != guide_r, "generator should avoid reflection landing on the guide row"
        if grid[guide_r, cc] != Color.RED:
            continue
        draw_r = guide_r - 1 if rr < guide_r else guide_r + 1
        assert in_bounds(grid.shape, draw_r, cc)
        out[draw_r, cc] = Color.BLUE

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reflect scattered pixels around a pivot, then keep only the reflections whose columns project onto solid guide-bar segments.

    Input structure:
    - Black background contains exactly one red horizontal guide bar with 1 to 3 internal black gaps.
    - A single grey pivot pixel sits 3 to 5 rows above or below the guide bar.
    - Several isolated blue singleton pixels lie on the same side of the guide bar as the pivot.

    Rule:
    - For each blue pixel (r, c), compute its 180-degree reflection through the grey pivot: (2*pr-r, 2*pc-c).
    - Inspect the reflected column on the red guide row. If that column is a red guide cell, draw a blue pixel on the guide-adjacent row on the same side as the reflected point.
    - If the reflected column falls into a black guide-bar gap, erase that blue pixel instead.
    - Preserve the red guide bar and grey pivot, and clear the original blue input pixels.

    Key insight:
    - The reflection decides the candidate column and side of the guide bar, while the guide-bar gaps decide which reflected candidates survive.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    return _apply_rule(g)


def _random_partition(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(total + parts - 1), parts - 1))
    values = []
    prev = -1
    for cut in cuts + [total + parts - 1]:
        values.append(cut - prev - 1)
        prev = cut
    return values


def _make_guide_mask(width: int, solid_need: int, gap_need: int) -> np.ndarray | None:
    for _ in range(200):
        n_gaps = random.randint(1, 3)
        gap_lengths = [random.randint(1, 3) for _ in range(n_gaps)]
        min_solid_total = n_gaps + 1
        extra = width - sum(gap_lengths) - min_solid_total
        if extra < 0:
            continue

        solid_lengths = [1 + x for x in _random_partition(extra, n_gaps + 1)]
        mask = np.zeros(width, dtype=bool)
        c = 0
        for i, solid_len in enumerate(solid_lengths):
            mask[c : c + solid_len] = True
            c += solid_len
            if i < n_gaps:
                c += gap_lengths[i]

        if not (mask[0] and mask[-1]):
            continue
        if int(mask.sum()) < solid_need or int((~mask).sum()) < gap_need:
            continue
        return mask
    return None


def _sample_spaced(candidates: list[int], count: int, taken: list[int]) -> list[int] | None:
    for _ in range(200):
        picks: list[int] = []
        pool = candidates[:]
        random.shuffle(pool)
        for c in pool:
            if any(abs(c - x) <= 1 for x in taken + picks):
                continue
            picks.append(int(c))
            if len(picks) == count:
                return picks
    return None


def _reflect_inverse(pivot_r: int, pivot_c: int, rr: int, cc: int) -> tuple[int, int]:
    return 2 * pivot_r - rr, 2 * pivot_c - cc


def _build_expected(
    shape: tuple[int, int],
    guide_r: int,
    guide_mask: np.ndarray,
    pivot: tuple[int, int],
    blue_inputs: list[tuple[int, int]],
) -> np.ndarray:
    h, w = shape
    out = np.full((h, w), Color.BLACK, dtype=int)
    out[guide_r, guide_mask] = Color.RED
    pivot_r, pivot_c = pivot
    out[pivot_r, pivot_c] = Color.GREY

    for r, c in blue_inputs:
        rr = 2 * pivot_r - r
        cc = 2 * pivot_c - c
        if guide_mask[cc]:
            out[guide_r - 1 if rr < guide_r else guide_r + 1, cc] = Color.BLUE
    return out


def _alt_without_gap_filter(grid: np.ndarray) -> np.ndarray:
    guide_r = _find_guide_row(grid)
    pivot_r, pivot_c = _find_pivot(grid)
    out = np.array(grid, dtype=int, copy=True)
    out[out == Color.BLUE] = Color.BLACK
    for r, c in _blue_coords(grid):
        rr = 2 * pivot_r - r
        cc = 2 * pivot_c - c
        out[guide_r - 1 if rr < guide_r else guide_r + 1, cc] = Color.BLUE
    return out


def _alt_using_input_columns(grid: np.ndarray) -> np.ndarray:
    guide_r = _find_guide_row(grid)
    pivot_r, _pivot_c = _find_pivot(grid)
    out = np.array(grid, dtype=int, copy=True)
    out[out == Color.BLUE] = Color.BLACK
    for r, c in _blue_coords(grid):
        rr = 2 * pivot_r - r
        if grid[guide_r, c] != Color.RED:
            continue
        out[guide_r - 1 if rr < guide_r else guide_r + 1, c] = Color.BLUE
    return out


def _alt_direct_reflection(grid: np.ndarray) -> np.ndarray:
    guide_r = _find_guide_row(grid)
    pivot_r, pivot_c = _find_pivot(grid)
    out = np.array(grid, dtype=int, copy=True)
    out[out == Color.BLUE] = Color.BLACK
    for r, c in _blue_coords(grid):
        rr = 2 * pivot_r - r
        cc = 2 * pivot_c - c
        if grid[guide_r, cc] == Color.RED:
            out[rr, cc] = Color.BLUE
    return out


def _count_gap_runs(row_mask: np.ndarray) -> int:
    runs = 0
    in_gap = False
    for is_solid in row_mask:
        if not is_solid and not in_gap:
            runs += 1
            in_gap = True
        elif is_solid:
            in_gap = False
    return runs


def _semantic_checks(grid: np.ndarray, expected: np.ndarray) -> bool:
    guide_r = _find_guide_row(grid)
    pivot_r, pivot_c = _find_pivot(grid)
    blues = _blue_coords(grid)
    if len(blues) not in (4, 6, 8):
        return False
    if len(components_mask(grid == Color.BLUE, connectivity=4)) != len(blues):
        return False

    guide_mask = grid[guide_r] == Color.RED
    if not (guide_mask[0] and guide_mask[-1]):
        return False
    if not (1 <= _count_gap_runs(guide_mask) <= 3):
        return False
    if not (3 <= abs(pivot_r - guide_r) <= 5):
        return False

    side = -1 if pivot_r < guide_r else 1
    solid_hits = 0
    gap_hits = 0
    solid_above = solid_below = False
    gap_above = gap_below = False

    for r, c in blues:
        if (r - guide_r) * side <= 0:
            return False
        rr = 2 * pivot_r - r
        cc = 2 * pivot_c - c
        if not in_bounds(grid.shape, rr, cc) or rr == guide_r:
            return False
        if c == cc:
            return False
        if guide_mask[cc]:
            solid_hits += 1
            solid_above |= rr < guide_r
            solid_below |= rr > guide_r
        else:
            gap_hits += 1
            gap_above |= rr < guide_r
            gap_below |= rr > guide_r

    if solid_hits != gap_hits or solid_hits == 0:
        return False
    if not (solid_above and solid_below and gap_above and gap_below):
        return False

    true_out = _apply_rule(grid)
    if not np.array_equal(true_out, expected):
        return False
    if np.array_equal(_alt_without_gap_filter(grid), expected):
        return False
    if np.array_equal(_alt_using_input_columns(grid), expected):
        return False
    if np.array_equal(_alt_direct_reflection(grid), expected):
        return False
    return True


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(19, 24)
    w = random.randint(16, 24)
    n_blue = random.choice((4, 6, 8))
    solid_need = gap_need = n_blue // 2

    for _ in range(300):
        distance = random.randint(3, 5)
        feasible_sides = []
        if 2 * distance + 3 <= h - 6:
            feasible_sides.append("above")
        if 5 <= h - 2 * distance - 4:
            feasible_sides.append("below")
        if not feasible_sides:
            continue

        pivot_side = random.choice(feasible_sides)
        if pivot_side == "above":
            guide_r = random.randint(2 * distance + 3, h - 6)
            pivot_r = guide_r - distance
            rr_above_pool = list(range(pivot_r + 1, guide_r - 1))
            rr_below_pool = list(range(guide_r + 2, min(h - 1, 2 * pivot_r - 1) + 1))
        else:
            guide_r = random.randint(5, h - 2 * distance - 4)
            pivot_r = guide_r + distance
            rr_above_pool = list(range(max(0, 2 * pivot_r - (h - 2)), guide_r - 1))
            rr_below_pool = list(range(guide_r + 2, pivot_r))

        if not rr_above_pool or not rr_below_pool:
            continue

        pivot_c = random.randint(max(2, w // 4), min(w - 3, w - 1 - w // 4))
        guide_mask = _make_guide_mask(w, solid_need, gap_need)
        if guide_mask is None:
            continue

        valid_solid = [
            c
            for c in range(w)
            if guide_mask[c] and c != pivot_c and 0 <= 2 * pivot_c - c < w
        ]
        valid_gap = [
            c
            for c in range(w)
            if (not guide_mask[c]) and c != pivot_c and 0 <= 2 * pivot_c - c < w
        ]
        solid_cols = _sample_spaced(valid_solid, solid_need, [])
        if solid_cols is None:
            continue
        gap_cols = _sample_spaced(valid_gap, gap_need, solid_cols)
        if gap_cols is None:
            continue

        categories = [
            ("solid", "above"),
            ("solid", "below"),
            ("gap", "above"),
            ("gap", "below"),
        ]
        for _ in range((n_blue - 4) // 2):
            categories.append(("solid", random.choice(("above", "below"))))
            categories.append(("gap", random.choice(("above", "below"))))
        random.shuffle(categories)

        solid_cols = solid_cols[:]
        gap_cols = gap_cols[:]
        random.shuffle(solid_cols)
        random.shuffle(gap_cols)

        blue_inputs: list[tuple[int, int]] = []
        for kind, vertical_side in categories:
            cc = solid_cols.pop() if kind == "solid" else gap_cols.pop()
            rr_pool = rr_above_pool if vertical_side == "above" else rr_below_pool
            rr = random.choice(rr_pool)
            blue_inputs.append(_reflect_inverse(pivot_r, pivot_c, rr, cc))

        if len(set(blue_inputs)) != len(blue_inputs):
            continue

        grid = np.full((h, w), Color.BLACK, dtype=int)
        grid[guide_r, guide_mask] = Color.RED
        grid[pivot_r, pivot_c] = Color.GREY
        for r, c in blue_inputs:
            if not in_bounds((h, w), r, c):
                break
            grid[r, c] = Color.BLUE
        else:
            expected = _build_expected((h, w), guide_r, guide_mask, (pivot_r, pivot_c), blue_inputs)
            if not _semantic_checks(grid, expected):
                continue
            if not np.array_equal(main(grid), expected):
                continue
            return grid

    return None


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed to produce a valid ARC sample")
