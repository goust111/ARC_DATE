from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, find_components


BG = int(Color.BLACK)
BAR = int(Color.BLUE)
CENTER = int(Color.YELLOW)
BRIDGE = int(Color.RED)

CARDINAL_GAPS = ((-1, 0), (1, 0), (0, -1), (0, 1))
MANDATORY_BARS = {
    "up": ((-3, 0), (-2, 0)),
    "down": ((2, 0), (3, 0)),
    "left": ((0, -3), (0, -2)),
    "right": ((0, 2), (0, 3)),
}
OUTER_BARS = {
    "up": {"gap": (-4, 0), "cells": ((-6, 0), (-5, 0))},
    "down": {"gap": (4, 0), "cells": ((5, 0), (6, 0))},
    "left": {"gap": (0, -4), "cells": ((0, -6), (0, -5))},
    "right": {"gap": (0, 4), "cells": ((0, 5), (0, 6))},
}

TEMPLATES = (
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1), (0, 2)),
    ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def _offset_cells(r: int, c: int, offsets: tuple[tuple[int, int], ...]) -> list[tuple[int, int]]:
    return [(int(r + dr), int(c + dc)) for dr, dc in offsets]


def _bar_orientation(cells: list[tuple[int, int]]) -> str | None:
    (r0, c0), (r1, c1) = cells
    if r0 == r1 and abs(c0 - c1) == 1:
        return "h"
    if c0 == c1 and abs(r0 - r1) == 1:
        return "v"
    return None


def _blue_bar_components(grid: np.ndarray) -> list[dict]:
    comps = find_components(grid, bg=BG, colors=[BAR], connectivity=4, monochromatic=True)
    for comp in comps:
        coords = [tuple(map(int, rc)) for rc in comp["coords"]]
        assert len(coords) == 2, "every blue object must be a two-pixel bar"
        assert _bar_orientation(coords) is not None, "every blue object must be horizontal or vertical"
    return comps


def _mandatory_bar_present(grid: np.ndarray, r: int, c: int, direction: str) -> bool:
    cells = _offset_cells(r, c, MANDATORY_BARS[direction])
    return all(_in_bounds(grid.shape, rr, cc) and int(grid[rr, cc]) == BAR for rr, cc in cells)


def _outer_bar_present(grid: np.ndarray, r: int, c: int, direction: str) -> bool:
    gap = OUTER_BARS[direction]["gap"]
    gap_r, gap_c = int(r + gap[0]), int(c + gap[1])
    cells = _offset_cells(r, c, OUTER_BARS[direction]["cells"])
    return (
        _in_bounds(grid.shape, gap_r, gap_c)
        and int(grid[gap_r, gap_c]) == BG
        and all(_in_bounds(grid.shape, rr, cc) and int(grid[rr, cc]) == BAR for rr, cc in cells)
    )


def _is_hidden_center(grid: np.ndarray, r: int, c: int) -> bool:
    if int(grid[r, c]) != BG:
        return False
    if any(int(grid[r + dr, c + dc]) != BG for dr, dc in CARDINAL_GAPS):
        return False
    return all(_mandatory_bar_present(grid, r, c, direction) for direction in MANDATORY_BARS)


def _detect_centers(grid: np.ndarray) -> list[tuple[int, int]]:
    h, w = grid.shape
    centers = []
    for r in range(3, h - 3):
        for c in range(3, w - 3):
            if _is_hidden_center(grid, r, c):
                centers.append((int(r), int(c)))
    return centers


def _countable_outer_dirs(
    centers: set[tuple[int, int]],
    center: tuple[int, int],
) -> list[str]:
    return _available_extra_dirs(centers, center)


def _count_structure_bars(
    grid: np.ndarray,
    center: tuple[int, int],
    centers: set[tuple[int, int]],
) -> int:
    r, c = center
    count = 4
    for direction in _countable_outer_dirs(centers, center):
        if _outer_bar_present(grid, r, c, direction):
            count += 1
    return count


def _aligned_pairs(centers: list[tuple[int, int]]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    ordered = sorted((int(r), int(c)) for r, c in centers)
    pairs = []
    for i, a in enumerate(ordered):
        for b in ordered[i + 1 :]:
            if a[0] == b[0] or a[1] == b[1]:
                pairs.append((a, b))
    return pairs


def _band_offsets(width: int) -> range:
    return range(-((int(width) - 1) // 2), int(width) // 2 + 1)


def _draw_bridge(
    out: np.ndarray,
    a: tuple[int, int],
    b: tuple[int, int],
    width: int,
) -> None:
    width = int(width)
    (r0, c0), (r1, c1) = a, b
    if r0 == r1:
        left, right = sorted((c0, c1))
        for dc in range(left + 1, right):
            for dr_off in _band_offsets(width):
                rr = int(r0 + dr_off)
                if _in_bounds(out.shape, rr, dc) and int(out[rr, dc]) == BG:
                    out[rr, dc] = BRIDGE
        return
    assert c0 == c1, "bridge endpoints must share a row or a column"
    top, bottom = sorted((r0, r1))
    for dr in range(top + 1, bottom):
        for dc_off in _band_offsets(width):
            cc = int(c0 + dc_off)
            if _in_bounds(out.shape, dr, cc) and int(out[dr, cc]) == BG:
                out[dr, cc] = BRIDGE


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mark hidden plus centers, then connect aligned centers with red bridges whose width comes from local bar counts.

    Input structure:
    - The grid is a black background with blue two-pixel bars only.
    - Some bars form an implied plus around an empty background center: one vertical bar above, one below, one horizontal bar left, and one right, each separated from the center by one background cell.
    - A subset of these implied pluses also has one or two extra two-pixel bars continuing outward along some arms; unrelated blue bars may appear as distractors.

    Rule:
    - Detect every background cell whose four mandatory arm-bars form a complete hidden plus and recolor that center yellow.
    - For each detected plus, count its own bars: the four mandatory bars plus any outward continuation bars on its arms.
    - For every pair of detected centers sharing a row or a column, draw a red bridge between the centers along that row or column.
    - The bridge width is the smaller of the two bar-counts minus three; bridge cells are drawn only onto background and must not overwrite yellow centers or blue bars.

    Key insight:
    - The same local bar motif supplies two different signals: its four mandatory bars reveal a missing center, while its total bar count controls the thickness of long-distance bridges between aligned centers.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D ARC grid"
    _blue_bar_components(g)
    centers = _detect_centers(g)
    assert centers, "expected at least one hidden plus center"
    center_set = set(centers)

    counts = {center: _count_structure_bars(g, center, center_set) for center in centers}
    out = g.copy()
    for r, c in centers:
        out[r, c] = CENTER

    for a, b in _aligned_pairs(centers):
        width = min(counts[a], counts[b]) - 3
        _draw_bridge(out, a, b, width)
    return out


def _sample_axis_positions(rng: random.Random, size: int, count: int) -> list[int] | None:
    lo = 6
    hi = int(size - 7)
    if hi < lo:
        return None
    candidates = list(range(lo, hi + 1))
    for _ in range(200):
        picks = sorted(rng.sample(candidates, count))
        if all(b - a >= 8 for a, b in zip(picks, picks[1:])):
            return picks
    return None


def _choose_template(rng: random.Random) -> tuple[tuple[int, int], ...]:
    return tuple(rng.choice(TEMPLATES))


def _same_line_neighbors(
    centers: set[tuple[int, int]],
    center: tuple[int, int],
) -> dict[str, list[tuple[int, int]]]:
    r, c = center
    return {
        "up": sorted((rr, cc) for rr, cc in centers if cc == c and rr < r),
        "down": sorted((rr, cc) for rr, cc in centers if cc == c and rr > r),
        "left": sorted((rr, cc) for rr, cc in centers if rr == r and cc < c),
        "right": sorted((rr, cc) for rr, cc in centers if rr == r and cc > c),
    }


def _available_extra_dirs(
    centers: set[tuple[int, int]],
    center: tuple[int, int],
) -> list[str]:
    neighbors = _same_line_neighbors(centers, center)
    return [direction for direction, others in neighbors.items() if not others]


def _plan_counts_and_extras(
    rng: random.Random,
    centers: list[tuple[int, int]],
) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], tuple[str, ...]]]:
    center_set = set(centers)
    counts: dict[tuple[int, int], int] = {}
    extras: dict[tuple[int, int], tuple[str, ...]] = {}

    for center in centers:
        available = _countable_outer_dirs(center_set, center)
        feasible_counts = [4 + k for k in range(min(2, len(available)) + 1)]
        weights = [2, 3, 3][: len(feasible_counts)]
        count = int(rng.choices(feasible_counts, weights=weights, k=1)[0])
        chosen_dirs = tuple(sorted(rng.sample(available, count - 4)))
        counts[center] = count
        extras[center] = chosen_dirs

    unique_counts = set(counts.values())
    if len(unique_counts) == 1 and len(centers) >= 2:
        shuffled = centers[:]
        rng.shuffle(shuffled)
        for center in shuffled:
            available = _countable_outer_dirs(center_set, center)
            alternatives = [4 + k for k in range(min(2, len(available)) + 1) if 4 + k != counts[center]]
            if alternatives:
                new_count = int(rng.choice(alternatives))
                counts[center] = new_count
                extras[center] = tuple(sorted(rng.sample(available, new_count - 4)))
                break

    return counts, extras


def _paint_bar(grid: np.ndarray, cells: list[tuple[int, int]]) -> None:
    for r, c in cells:
        grid[r, c] = BAR


def _candidate_bar_cells(center: tuple[int, int], direction: str, outer: bool = False) -> list[tuple[int, int]]:
    r, c = center
    if outer:
        offsets = OUTER_BARS[direction]["cells"]
    else:
        offsets = MANDATORY_BARS[direction]
    return _offset_cells(r, c, offsets)


def _can_place_bar(grid: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    if _bar_orientation(cells) is None:
        return False
    cell_set = set(cells)
    for r, c in cells:
        if not _in_bounds(grid.shape, r, c) or int(grid[r, c]) != BG:
            return False
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (nr, nc) in cell_set:
                continue
            if _in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) == BAR:
                return False
    return True


def _protected_cross_cells(
    centers: list[tuple[int, int]],
) -> set[tuple[int, int]]:
    protected: set[tuple[int, int]] = set()
    for r, c in centers:
        protected.add((int(r), int(c)))
        protected.update(_offset_cells(r, c, tuple(CARDINAL_GAPS)))
        for direction in MANDATORY_BARS:
            protected.update(_candidate_bar_cells((r, c), direction, outer=False))
            protected.update(_candidate_bar_cells((r, c), direction, outer=True))
            gap = OUTER_BARS[direction]["gap"]
            protected.add((int(r + gap[0]), int(c + gap[1])))
    return protected


def _bridge_band_cells(
    centers: list[tuple[int, int]],
    counts: dict[tuple[int, int], int],
) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for a, b in _aligned_pairs(centers):
        width = min(counts[a], counts[b]) - 3
        (r0, c0), (r1, c1) = a, b
        if r0 == r1:
            left, right = sorted((c0, c1))
            for c in range(left + 1, right):
                for off in _band_offsets(width):
                    cells.add((int(r0 + off), int(c)))
        else:
            top, bottom = sorted((r0, r1))
            for r in range(top + 1, bottom):
                for off in _band_offsets(width):
                    cells.add((int(r), int(c0 + off)))
    return cells


def _try_place_corridor_blocker(
    grid: np.ndarray,
    rng: random.Random,
    centers: list[tuple[int, int]],
    counts: dict[tuple[int, int], int],
) -> bool:
    pairs = _aligned_pairs(centers)
    rng.shuffle(pairs)
    for a, b in pairs:
        (r0, c0), (r1, c1) = a, b
        if r0 == r1:
            left, right = sorted((c0, c1))
            gap_cols = list(range(left + 2, right - 1))
            if not gap_cols:
                continue
            mid_c = int(rng.choice(gap_cols))
            cells = [(r0, mid_c), (r0 + 1, mid_c)]
        else:
            top, bottom = sorted((r0, r1))
            gap_rows = list(range(top + 2, bottom - 1))
            if not gap_rows:
                continue
            mid_r = int(rng.choice(gap_rows))
            cells = [(mid_r, c0), (mid_r, c0 + 1)]
        if _can_place_bar(grid, cells):
            _paint_bar(grid, cells)
            return True
    return False


def _place_random_distractors(
    grid: np.ndarray,
    rng: random.Random,
    target_count: int,
    forbidden: set[tuple[int, int]],
) -> int:
    placed = 0
    h, w = grid.shape
    for _ in range(target_count * 60):
        if placed >= target_count:
            break
        horizontal = bool(rng.randint(0, 1))
        r = int(rng.randrange(h))
        c = int(rng.randrange(w))
        cells = [(r, c), (r, c + 1)] if horizontal else [(r, c), (r + 1, c)]
        if any((int(rr), int(cc)) in forbidden for rr, cc in cells):
            continue
        if _can_place_bar(grid, cells):
            _paint_bar(grid, cells)
            placed += 1
    return placed


def _plan_layout(
    rng: random.Random,
) -> tuple[np.ndarray, list[tuple[int, int]], dict[tuple[int, int], int], dict[tuple[int, int], tuple[str, ...]]] | None:
    template = _choose_template(rng)
    row_count = max(r for r, _c in template) + 1
    col_count = max(c for _r, c in template) + 1

    h = rng.randint(29, 30) if row_count == 3 else rng.randint(19, 30)
    w = rng.randint(29, 30) if col_count == 3 else rng.randint(19, 30)
    rows = _sample_axis_positions(rng, h, row_count)
    cols = _sample_axis_positions(rng, w, col_count)
    if rows is None or cols is None:
        return None

    centers = sorted((rows[ri], cols[ci]) for ri, ci in template)
    counts, extras = _plan_counts_and_extras(rng, centers)

    grid = np.full((h, w), BG, dtype=int)
    for center in centers:
        for direction in MANDATORY_BARS:
            cells = _candidate_bar_cells(center, direction, outer=False)
            if not _can_place_bar(grid, cells):
                return None
            _paint_bar(grid, cells)
        for direction in extras[center]:
            cells = _candidate_bar_cells(center, direction, outer=True)
            if not _can_place_bar(grid, cells):
                return None
            _paint_bar(grid, cells)

    return grid, centers, counts, extras


def _expected_output_from_plan(
    grid: np.ndarray,
    centers: list[tuple[int, int]],
    counts: dict[tuple[int, int], int],
) -> np.ndarray:
    out = grid.copy()
    for r, c in centers:
        out[r, c] = CENTER
    for a, b in _aligned_pairs(centers):
        _draw_bridge(out, a, b, min(counts[a], counts[b]) - 3)
    return out


def _validate_generated_instance(
    grid: np.ndarray,
    centers: list[tuple[int, int]],
    counts: dict[tuple[int, int], int],
    extras: dict[tuple[int, int], tuple[str, ...]],
) -> bool:
    detected = sorted(_detect_centers(grid))
    if detected != sorted(centers):
        return False

    center_set = set(detected)
    detected_counts = {_center: _count_structure_bars(grid, _center, center_set) for _center in detected}
    if detected_counts != counts:
        return False

    expected_extra_sizes = {center: len(extra_dirs) for center, extra_dirs in extras.items()}
    if any(counts[center] != 4 + expected_extra_sizes[center] for center in centers):
        return False

    pairs = _aligned_pairs(centers)
    if len(pairs) < 2:
        return False

    comps = _blue_bar_components(grid)
    distractor_count = len(comps) - sum(counts.values())
    if distractor_count < 1:
        return False

    expected = _expected_output_from_plan(grid, centers, counts)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False

    if int((actual == CENTER).sum()) != len(centers):
        return False
    if int((actual == BRIDGE).sum()) == 0:
        return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    planned = _plan_layout(rng)
    if planned is None:
        return None
    grid, centers, counts, extras = planned

    want_blocker = rng.random() < 0.65
    placed_blocker = False
    if want_blocker:
        placed_blocker = _try_place_corridor_blocker(grid, rng, centers, counts)
        if not placed_blocker:
            return None

    forbidden = _protected_cross_cells(centers)
    distractor_target = rng.randint(2, 7)
    placed_distractors = _place_random_distractors(grid, rng, distractor_target, forbidden)
    if placed_distractors + int(placed_blocker) < 1:
        return None

    if not _validate_generated_instance(grid, centers, counts, extras):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid hidden-cross bridge puzzle")
