from __future__ import annotations

import random
import sys
from itertools import permutations
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, coords_where, get_bg, in_bounds


Coord = tuple[int, int]
Pair = tuple[Coord, Coord]


ANGLE_BY_COLOR = {
    Color.RED: "cw90",
    Color.GREEN: "halfturn",
    Color.YELLOW: "ccw90",
}
SOURCE_COLORS = tuple(ANGLE_BY_COLOR)
ROTATION_LABELS = tuple(ANGLE_BY_COLOR[color] for color in SOURCE_COLORS)


GRID_SIDE_RANGE = (14, 19)
PIVOT_COUNT_RANGE = (3, 5)
SOURCE_DISTANCE_RANGE = (1, 5)
MAX_SOURCES_PER_PIVOT = 2
GENERATION_RETRIES = 500
MIN_PIVOT_GAP = 3


def _coords(mask: np.ndarray) -> list[Coord]:
    return [(int(r), int(c)) for r, c in coords_where(mask)]


def _rotate_offset(dr: int, dc: int, angle: str) -> Coord:
    if angle == "cw90":
        return dc, -dr
    if angle == "halfturn":
        return -dr, -dc
    if angle == "ccw90":
        return -dc, dr
    raise ValueError(f"unknown angle {angle}")


def _unique_pivot(source: Coord, pivots: list[Coord]) -> Coord:
    matches = [pivot for pivot in pivots if pivot[0] == source[0] or pivot[1] == source[1]]
    assert len(matches) == 1, "each source must align with exactly one pivot"
    return matches[0]


def _extract_pairs(grid: np.ndarray) -> tuple[list[Coord], list[Pair]]:
    pivots = _coords(grid == Color.BLUE)
    assert 3 <= len(pivots) <= 5, "expected 3-5 blue pivots"

    sources = _coords(np.isin(grid, SOURCE_COLORS))
    assert sources, "expected at least one source pixel"

    counts_by_pivot = {pivot: 0 for pivot in pivots}
    pairs: list[Pair] = []
    for source in sources:
        pivot = _unique_pivot(source, pivots)
        counts_by_pivot[pivot] += 1
        assert counts_by_pivot[pivot] <= MAX_SOURCES_PER_PIVOT, "each pivot supports at most two sources"
        pairs.append((source, pivot))

    return pivots, pairs


def _apply_mapping(grid: np.ndarray, angle_by_color: dict[int, str]) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    out[np.isin(out, SOURCE_COLORS)] = Color.BLACK

    pivots, pairs = _extract_pairs(g)
    _ = pivots
    for source, pivot in pairs:
        sr, sc = source
        pr, pc = pivot
        dr, dc = sr - pr, sc - pc
        tr_off, tc_off = _rotate_offset(dr, dc, angle_by_color[int(g[sr, sc])])
        tr, tc = pr + tr_off, pc + tc_off
        assert in_bounds(g.shape, tr, tc), "rotated target must stay in bounds"
        assert out[tr, tc] == Color.BLACK, "rotated targets must land on empty background"
        out[tr, tc] = Color.ORANGE

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each colored source pixel around its uniquely aligned blue pivot, using the source color as the rotation instruction, then recolor the result orange.

    Input structure:
    - The grid has black background, 3-5 isolated blue pivot pixels, and red/green/yellow source pixels.
    - Every source lies on the same row or same column as exactly one blue pivot.
    - Each pivot is paired with at most two sources, and source/target positions do not overlap.

    Rule:
    - Pair each source with its unique row-or-column-aligned blue pivot.
    - Decode the source color as a rotation angle: red -> clockwise 90 degrees, green -> 180 degrees, yellow -> counterclockwise 90 degrees.
    - Rotate the source's axis-aligned offset vector around the pivot, erase the original source, and paint the rotated in-bounds target orange.
    - Keep all blue pivots unchanged.

    Key insight:
    - The colored source pixels are not output colors to preserve; they are per-pixel rotation instructions whose effect is only visible after matching each source to its pivot.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must fit within 30x30"
    assert int(get_bg(g)) == Color.BLACK, "background must be black"
    assert set(map(int, g.flat)).issubset({Color.BLACK, Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW})

    return _apply_mapping(g, ANGLE_BY_COLOR)


def _sample_spaced_values(length: int, count: int) -> list[int] | None:
    usable_max = length - 2 - (MIN_PIVOT_GAP - 1) * (count - 1)
    if usable_max < 1:
        return None
    base = sorted(random.sample(range(1, usable_max + 1), count))
    return [value + (MIN_PIVOT_GAP - 1) * i for i, value in enumerate(base)]


def _sample_pivots(shape: tuple[int, int], count: int) -> list[Coord] | None:
    rows = _sample_spaced_values(shape[0], count)
    cols = _sample_spaced_values(shape[1], count)
    if rows is None or cols is None:
        return None
    random.shuffle(cols)
    return list(zip(rows, cols))


def _source_slots(pivots: list[Coord]) -> list[Coord]:
    total = random.randint(max(4, len(pivots)), min(len(pivots) * MAX_SOURCES_PER_PIVOT, len(pivots) + 2))
    counts = {pivot: 1 for pivot in pivots}
    extras = total - len(pivots)
    while extras > 0:
        pivot = random.choice([p for p in pivots if counts[p] < MAX_SOURCES_PER_PIVOT])
        counts[pivot] += 1
        extras -= 1

    slots: list[Coord] = []
    for pivot in pivots:
        slots.extend([pivot] * counts[pivot])
    random.shuffle(slots)
    return slots


def _slot_colors(total: int) -> list[int]:
    colors = [Color.RED, Color.GREEN, Color.YELLOW]
    while len(colors) < total:
        colors.append(random.choice(SOURCE_COLORS))
    random.shuffle(colors)
    return colors


def _candidate_placements(
    pivot: Coord,
    color: int,
    shape: tuple[int, int],
    pivots: list[Coord],
) -> list[tuple[Coord, Coord, bool]]:
    pr, pc = pivot
    candidates: list[tuple[Coord, Coord, bool]] = []
    for distance in range(SOURCE_DISTANCE_RANGE[0], SOURCE_DISTANCE_RANGE[1] + 1):
        for dr, dc in ((0, distance), (0, -distance), (distance, 0), (-distance, 0)):
            sr, sc = pr + dr, pc + dc
            if not in_bounds(shape, sr, sc):
                continue
            if (sr, sc) in pivots:
                continue
            if sum(1 for other in pivots if other[0] == sr or other[1] == sc) != 1:
                continue

            rr, rc = _rotate_offset(dr, dc, ANGLE_BY_COLOR[color])
            tr, tc = pr + rr, pc + rc
            if not in_bounds(shape, tr, tc):
                continue
            if (tr, tc) in pivots or (tr, tc) == (sr, sc):
                continue
            candidates.append(((sr, sc), (tr, tc), dr == 0))

    random.shuffle(candidates)
    return candidates


def _search_layout(
    slots: list[tuple[Coord, int]],
    candidates_by_slot: list[list[tuple[Coord, Coord, bool]]],
    idx: int,
    used_sources: set[Coord],
    used_targets: set[Coord],
    chosen: list[tuple[Coord, int, Coord, Coord, bool]],
    row_count: int,
    col_count: int,
) -> list[tuple[Coord, int, Coord, Coord, bool]] | None:
    if idx == len(slots):
        if row_count > 0 and col_count > 0:
            return chosen.copy()
        return None

    remaining = len(slots) - idx
    if row_count == 0 and not any(any(candidate[2] for candidate in candidates_by_slot[j]) for j in range(idx, len(slots))):
        return None
    if col_count == 0 and not any(any(not candidate[2] for candidate in candidates_by_slot[j]) for j in range(idx, len(slots))):
        return None
    if row_count == 0 and remaining == 0:
        return None
    if col_count == 0 and remaining == 0:
        return None

    pivot, color = slots[idx]
    for source, target, same_row in candidates_by_slot[idx]:
        if source in used_sources or source in used_targets:
            continue
        if target in used_targets or target in used_sources:
            continue

        used_sources.add(source)
        used_targets.add(target)
        chosen.append((pivot, color, source, target, same_row))
        result = _search_layout(
            slots,
            candidates_by_slot,
            idx + 1,
            used_sources,
            used_targets,
            chosen,
            row_count + int(same_row),
            col_count + int(not same_row),
        )
        if result is not None:
            return result
        chosen.pop()
        used_sources.remove(source)
        used_targets.remove(target)

    return None


def _build_input(shape: tuple[int, int], pivots: list[Coord], placements: list[tuple[Coord, int, Coord, Coord, bool]]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for r, c in pivots:
        grid[r, c] = Color.BLUE
    for _pivot, color, source, _target, _same_row in placements:
        grid[source] = color
    return grid


def _expected_output(shape: tuple[int, int], pivots: list[Coord], placements: list[tuple[Coord, int, Coord, Coord, bool]]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for r, c in pivots:
        out[r, c] = Color.BLUE
    for _pivot, _color, _source, target, _same_row in placements:
        out[target] = Color.ORANGE
    return out


def _mapping_is_unique(grid: np.ndarray, expected: np.ndarray) -> bool:
    color_orders = permutations(ROTATION_LABELS)
    for labels in color_orders:
        alt_mapping = {color: label for color, label in zip(SOURCE_COLORS, labels)}
        if alt_mapping == ANGLE_BY_COLOR:
            continue
        try:
            alt = _apply_mapping(grid, alt_mapping)
        except AssertionError:
            continue
        if np.array_equal(alt, expected):
            return False
    return True


def _semantic_checks(grid: np.ndarray, expected: np.ndarray) -> bool:
    if not _mapping_is_unique(grid, expected):
        return False

    pivots, pairs = _extract_pairs(grid)
    counts_by_color = {color: 0 for color in SOURCE_COLORS}
    counts_by_pivot = {pivot: 0 for pivot in pivots}
    same_row = 0
    same_col = 0

    for source, pivot in pairs:
        color = int(grid[source])
        counts_by_color[color] += 1
        counts_by_pivot[pivot] += 1
        same_row += int(source[0] == pivot[0])
        same_col += int(source[1] == pivot[1])

    if any(counts_by_color[color] == 0 for color in SOURCE_COLORS):
        return False
    if any(count > MAX_SOURCES_PER_PIVOT for count in counts_by_pivot.values()):
        return False
    if same_row == 0 or same_col == 0:
        return False
    if np.array_equal(grid, expected):
        return False
    if np.count_nonzero(expected == Color.ORANGE) != len(pairs):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    pivot_count = random.randint(PIVOT_COUNT_RANGE[0], PIVOT_COUNT_RANGE[1])
    side_min = max(GRID_SIDE_RANGE[0], 3 * pivot_count)
    if side_min > GRID_SIDE_RANGE[1]:
        return None
    shape = (
        random.randint(side_min, GRID_SIDE_RANGE[1]),
        random.randint(side_min, GRID_SIDE_RANGE[1]),
    )

    pivots = _sample_pivots(shape, pivot_count)
    if pivots is None:
        return None

    slot_pivots = _source_slots(pivots)
    slots = [(pivot, color) for pivot, color in zip(slot_pivots, _slot_colors(len(slot_pivots)))]
    if not slots:
        return None

    candidates_by_slot = []
    for pivot, color in slots:
        candidates = _candidate_placements(pivot, color, shape, pivots)
        if not candidates:
            return None
        candidates_by_slot.append(candidates)

    order = sorted(range(len(slots)), key=lambda i: len(candidates_by_slot[i]))
    ordered_slots = [slots[i] for i in order]
    ordered_candidates = [candidates_by_slot[i] for i in order]
    placements = _search_layout(ordered_slots, ordered_candidates, 0, set(), set(), [], 0, 0)
    if placements is None:
        return None

    grid = _build_input(shape, pivots, placements)
    expected = _expected_output(shape, pivots, placements)

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    if not _semantic_checks(grid, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
