from __future__ import annotations

import random
from typing import NamedTuple

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREY = 5
ORANGE = 7
TEAL = 8

RING_COLORS = (BLUE, RED, ORANGE, TEAL)
MOVING_FAMILIES = ((BLUE, TEAL), (RED, ORANGE))
MOVING_PAIR_SETS = {frozenset(family) for family in MOVING_FAMILIES}
NONMOVING_SIDE_PAIRS = tuple(
    (left, right)
    for left in RING_COLORS
    for right in RING_COLORS
    if frozenset((left, right)) not in MOVING_PAIR_SETS
)
TOP_BOTTOM_PAIR_SIDES = ("left", "right")
MAX_GENERATION_ATTEMPTS = 600


class RowPair(NamedTuple):
    row: int
    left_col: int
    source_col: int
    right_color: int


class RingPlan(NamedTuple):
    top: int
    left: int
    patch: np.ndarray


def _moving_pair_in_row(row: np.ndarray) -> RowPair | None:
    matches: list[RowPair] = []
    for family in MOVING_FAMILIES:
        cols = [c for c, color in enumerate(row) if int(color) in family]
        if len(cols) != 2:
            continue
        c0, c1 = map(int, cols)
        if {int(row[c0]), int(row[c1])} != set(family):
            continue
        matches.append(RowPair(-1, c0, c1, int(row[c1])))

    assert len(matches) <= 1, "each row may contain at most one moving pair"
    return matches[0] if matches else None


def _slide_pairs(grid: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]]]:
    out = np.asarray(grid, dtype=int).copy()
    moved_rows: list[tuple[int, int]] = []
    for r in range(out.shape[0]):
        pair = _moving_pair_in_row(out[r])
        if pair is None:
            continue
        dst_col = pair.left_col + 1
        if pair.source_col == dst_col:
            continue
        assert out[r, dst_col] == BLACK, "moving pair destination must be empty"
        out[r, dst_col] = pair.right_color
        out[r, pair.source_col] = BLACK
        moved_rows.append((r, dst_col))
    return out, moved_rows


def _find_rings(grid: np.ndarray) -> list[tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    rings: list[tuple[int, int]] = []
    for r in range(g.shape[0] - 2):
        for c in range(g.shape[1] - 2):
            patch = g[r : r + 3, c : c + 3]
            if int(patch[1, 1]) != BLACK:
                continue
            border = patch.copy()
            border[1, 1] = 1
            if np.all(border != BLACK):
                rings.append((r, c))
    return rings


def _rotate_ring(out: np.ndarray, source: np.ndarray, r0: int, c0: int) -> None:
    patch = source[r0 : r0 + 3, c0 : c0 + 3]
    out[r0 + 0, c0 + 0] = patch[2, 0]
    out[r0 + 0, c0 + 1] = patch[1, 2]
    out[r0 + 0, c0 + 2] = patch[0, 0]
    out[r0 + 1, c0 + 0] = patch[0, 1]
    out[r0 + 1, c0 + 2] = patch[2, 1]
    out[r0 + 2, c0 + 0] = patch[2, 2]
    out[r0 + 2, c0 + 1] = patch[1, 0]
    out[r0 + 2, c0 + 2] = patch[0, 2]


def _rotate_rings(grid: np.ndarray, rings: list[tuple[int, int]]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for r0, c0 in rings:
        _rotate_ring(out, grid, r0, c0)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide same-hue row pairs together, detect the complete 3x3 rings that newly appear, and cycle ring corner and side colors in opposite directions.

    Input structure:
    - The grid has a black background with scattered blue/teal and red/orange pixels plus grey distractor singles.
    - A row may contain one moving same-hue pair: exactly one blue and one teal, or exactly one red and one orange.
    - The left pixel of a moving pair stays fixed while the right pixel starts farther to the right; grey pixels and non-paired foreground pixels are static.

    Rule:
    - First compact every moving row pair by moving its right pixel leftward to the cell immediately right of the left pixel and clearing the old source cell.
    - After all rows slide, scan every 3x3 window whose center is black and whose eight border cells are all non-black; each such window is a complete ring.
    - For every detected ring, rotate the four corner colors one step clockwise and the four side-center colors one step counterclockwise.

    Key insight:
    - The ring is not an input object: horizontal pair compaction changes which pixels become adjacent, so ring detection must happen only after simulating the slides.
    """
    slid, _ = _slide_pairs(np.asarray(grid, dtype=int))
    rings = _find_rings(slid)
    return _rotate_rings(slid, rings)


def _ring_rotation_changes(patch: np.ndarray) -> bool:
    probe = np.asarray(patch, dtype=int).copy()
    out = probe.copy()
    _rotate_ring(out, probe, 0, 0)
    return not np.array_equal(out, probe)


def _random_ring_patch() -> tuple[np.ndarray, tuple[str, tuple[int, int]], tuple[str, tuple[int, int]]]:
    for _ in range(200):
        patch = np.zeros((3, 3), dtype=int)

        top_family = random.choice(MOVING_FAMILIES)
        top_side = random.choice(TOP_BOTTOM_PAIR_SIDES)
        top_pair = random.sample(top_family, 2)
        top_extra = random.choice([c for c in RING_COLORS if c not in top_family])
        if top_side == "left":
            patch[0] = np.array([top_pair[0], top_pair[1], top_extra], dtype=int)
        else:
            patch[0] = np.array([top_extra, top_pair[0], top_pair[1]], dtype=int)

        mid_left, mid_right = random.choice(NONMOVING_SIDE_PAIRS)
        patch[1, 0] = mid_left
        patch[1, 2] = mid_right

        bottom_family = random.choice(MOVING_FAMILIES)
        bottom_side = random.choice(TOP_BOTTOM_PAIR_SIDES)
        bottom_pair = random.sample(bottom_family, 2)
        bottom_extra = random.choice([c for c in RING_COLORS if c not in bottom_family])
        if bottom_side == "left":
            patch[2] = np.array([bottom_pair[0], bottom_pair[1], bottom_extra], dtype=int)
        else:
            patch[2] = np.array([bottom_extra, bottom_pair[0], bottom_pair[1]], dtype=int)

        if _ring_rotation_changes(patch):
            return patch, (top_side, tuple(top_pair)), (bottom_side, tuple(bottom_pair))
    raise RuntimeError("failed to build a non-degenerate ring patch")


def _choose_source_col(grid: np.ndarray, row: int, min_col: int) -> int | None:
    choices = [c for c in range(min_col, grid.shape[1]) if int(grid[row, c]) == BLACK]
    return random.choice(choices) if choices else None


def _paint_ring_input(grid: np.ndarray, top: int, left: int) -> RingPlan | None:
    patch, top_meta, bottom_meta = _random_ring_patch()

    top_side, top_pair = top_meta
    if top_side == "left":
        grid[top, left] = patch[0, 0]
        grid[top, left + 2] = patch[0, 2]
        src = _choose_source_col(grid, top, left + 4)
        if src is None:
            return None
        grid[top, src] = top_pair[1]
    else:
        grid[top, left] = patch[0, 0]
        grid[top, left + 1] = patch[0, 1]
        src = _choose_source_col(grid, top, left + 4)
        if src is None:
            return None
        grid[top, src] = top_pair[1]

    grid[top + 1, left] = patch[1, 0]
    grid[top + 1, left + 2] = patch[1, 2]

    bottom_side, bottom_pair = bottom_meta
    if bottom_side == "left":
        grid[top + 2, left] = patch[2, 0]
        grid[top + 2, left + 2] = patch[2, 2]
        src = _choose_source_col(grid, top + 2, left + 4)
        if src is None:
            return None
        grid[top + 2, src] = bottom_pair[1]
    else:
        grid[top + 2, left] = patch[2, 0]
        grid[top + 2, left + 1] = patch[2, 1]
        src = _choose_source_col(grid, top + 2, left + 4)
        if src is None:
            return None
        grid[top + 2, src] = bottom_pair[1]

    return RingPlan(top=top, left=left, patch=patch.copy())


def _row_is_free(grid: np.ndarray, row: int) -> bool:
    return bool(np.all(grid[row] == BLACK))


def _add_extra_moving_pair(grid: np.ndarray, forbidden_rows: set[int]) -> bool:
    rows = [r for r in range(grid.shape[0]) if r not in forbidden_rows and _row_is_free(grid, r)]
    if not rows:
        return False
    row = random.choice(rows)
    family = random.choice(MOVING_FAMILIES)
    pair = random.sample(family, 2)
    left_col = random.randint(0, grid.shape[1] - 4)
    src_choices = [c for c in range(left_col + 3, grid.shape[1]) if int(grid[row, c]) == BLACK]
    if not src_choices:
        return False
    src = random.choice(src_choices)
    grid[row, left_col] = pair[0]
    grid[row, src] = pair[1]
    return True


def _add_extra_nonmoving_row(grid: np.ndarray, forbidden_rows: set[int]) -> bool:
    rows = [r for r in range(grid.shape[0]) if r not in forbidden_rows and _row_is_free(grid, r)]
    if not rows:
        return False
    row = random.choice(rows)
    cols = random.sample(range(grid.shape[1]), random.choice((1, 2, 2, 3)))
    cols.sort()
    for _ in range(40):
        values = [random.choice(RING_COLORS) for _ in cols]
        trial = np.zeros(grid.shape[1], dtype=int)
        for c, value in zip(cols, values):
            trial[c] = value
        if _moving_pair_in_row(trial) is not None:
            continue
        for c, value in zip(cols, values):
            grid[row, c] = value
        return True
    return False


def _empty_cells(grid: np.ndarray) -> list[tuple[int, int]]:
    return [tuple(map(int, rc)) for rc in np.argwhere(grid == BLACK)]


def _future_slide_destinations(grid: np.ndarray) -> set[tuple[int, int]]:
    destinations: set[tuple[int, int]] = set()
    for r in range(grid.shape[0]):
        pair = _moving_pair_in_row(grid[r])
        if pair is not None and pair.source_col != pair.left_col + 1:
            destinations.add((r, pair.left_col + 1))
    return destinations


def _add_grey_singles(grid: np.ndarray, count: int) -> None:
    blocked = _future_slide_destinations(grid)
    cells = [cell for cell in _empty_cells(grid) if cell not in blocked]
    random.shuffle(cells)
    for r, c in cells[:count]:
        grid[r, c] = GREY


def _ring_boxes_ok(plans: list[RingPlan]) -> bool:
    for i, a in enumerate(plans):
        for b in plans[i + 1 :]:
            if not (a.top + 2 < b.top or b.top + 2 < a.top or a.left + 2 < b.left or b.left + 2 < a.left):
                return False
    return True


def _candidate_ring_positions(height: int, width: int, count: int) -> list[list[tuple[int, int]]]:
    spots = [(r, c) for r in range(height - 2) for c in range(max(1, width - 5))]
    combos: list[list[tuple[int, int]]] = []
    if count == 1:
        return [[spot] for spot in spots]
    for i, first in enumerate(spots):
        for second in spots[i + 1 :]:
            if abs(first[0] - second[0]) >= 4:
                combos.append([first, second])
    return combos


def _semantic_checks(inp: np.ndarray, expected_rings: set[tuple[int, int]]) -> bool:
    if inp.ndim != 2 or max(inp.shape) > 30:
        return False

    for row in inp:
        matches = 0
        for family in MOVING_FAMILIES:
            cols = [c for c, color in enumerate(row) if int(color) in family]
            if len(cols) == 2 and {int(row[cols[0]]), int(row[cols[1]])} == set(family):
                matches += 1
        if matches > 1:
            return False

    input_rings = set(_find_rings(inp))
    if input_rings:
        return False

    try:
        slid, moved = _slide_pairs(inp)
    except AssertionError:
        return False
    if not moved:
        return False
    slid_rings = set(_find_rings(slid))
    if slid_rings != expected_rings:
        return False

    try:
        out = main(inp)
    except AssertionError:
        return False
    expected_out = _rotate_rings(slid, sorted(expected_rings))
    if not np.array_equal(out, expected_out):
        return False
    if np.array_equal(out, inp) or np.array_equal(out, slid):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(11, 16)
    width = random.randint(11, 17)
    ring_count = random.choice((1, 2))
    position_sets = _candidate_ring_positions(height, width, ring_count)
    if not position_sets:
        return None

    grid = np.zeros((height, width), dtype=int)
    plans: list[RingPlan] = []
    for top, left in random.choice(position_sets):
        plan = _paint_ring_input(grid, top, left)
        if plan is None:
            return None
        plans.append(plan)
    if not _ring_boxes_ok(plans):
        return None

    reserved_rows = {plan.top + offset for plan in plans for offset in range(3)}

    extra_moving_target = random.randint(1, 3)
    extra_nonmoving_target = random.randint(1, 3)
    moving_added = 0
    nonmoving_added = 0
    for _ in range(20):
        if moving_added < extra_moving_target and _add_extra_moving_pair(grid, reserved_rows):
            moving_added += 1
        if nonmoving_added < extra_nonmoving_target and _add_extra_nonmoving_row(grid, reserved_rows):
            nonmoving_added += 1
        if moving_added >= extra_moving_target and nonmoving_added >= extra_nonmoving_target:
            break

    _add_grey_singles(grid, random.randint(2, 6))

    expected_rings = {(plan.top, plan.left) for plan in plans}
    if not _semantic_checks(grid, expected_rings):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        candidate = _gen_attempt()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid emergent-ring input")
