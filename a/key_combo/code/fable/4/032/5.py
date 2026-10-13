"""ARC task 032-5: population-ordered diagonal ray extension."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


BACKGROUND = 0
FIRST_COLOR = 8   # teal
SECOND_COLOR = 2  # red
ORIENTATION_STEPS = ((1, 1), (1, -1))  # down-right, down-left
INPUT_COLORS = (1, 3, 4, 5, 6, 7, 9)
MAX_GENERATION_ATTEMPTS = 4000


Coord = tuple[int, int]


@dataclass(frozen=True)
class Bar:
    orientation: int
    cells: tuple[Coord, ...]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    """Use the common ARC `(row, column)` coordinate convention."""
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _extract_bars(grid: np.ndarray) -> dict[int, list[Bar]]:
    """Decompose foreground into unique maximal diagonal runs."""
    foreground = grid != BACKGROUND
    bars: dict[int, list[Bar]] = {0: [], 1: []}
    claims = np.zeros(grid.shape, dtype=np.int8)

    for orientation, (dr, dc) in enumerate(ORIENTATION_STEPS):
        for raw_r, raw_c in np.argwhere(foreground):
            r, c = int(raw_r), int(raw_c)
            previous = (r - dr, c - dc)
            following = (r + dr, c + dc)
            if _in_bounds(grid.shape, *previous) and foreground[previous]:
                continue
            if not (_in_bounds(grid.shape, *following) and foreground[following]):
                continue

            cells: list[Coord] = []
            rr, cc = r, c
            while _in_bounds(grid.shape, rr, cc) and foreground[rr, cc]:
                cells.append((rr, cc))
                rr += dr
                cc += dc
            if len(cells) >= 2:
                bars[orientation].append(Bar(orientation, tuple(cells)))
                for cell in cells:
                    claims[cell] += 1

    assert bars[0] and bars[1], "expected at least one bar of each diagonal orientation"
    assert np.all(claims[foreground] == 1), (
        "every foreground pixel must belong to exactly one unambiguous diagonal bar"
    )

    # Same-orientation bars must occupy distinct infinite diagonals, as specified.
    for orientation, group in bars.items():
        keys = [
            bar.cells[0][0] - bar.cells[0][1]
            if orientation == 0
            else bar.cells[0][0] + bar.cells[0][1]
            for bar in group
        ]
        assert len(keys) == len(set(keys)), "same-orientation bars may not share a diagonal"
    return bars


def _draw(
    shape: tuple[int, int],
    bars: dict[int, list[Bar]],
    first: int,
) -> tuple[np.ndarray, dict[str, list[int]]]:
    """Draw whole-bar rays in population order and retain semantic trace counts."""
    owner = np.full(shape, -1, dtype=np.int8)
    occupied = np.zeros(shape, dtype=bool)
    for orientation in (0, 1):
        for bar in bars[orientation]:
            for cell in bar.cells:
                assert not occupied[cell], "input bars must not overlap"
                occupied[cell] = True
                owner[cell] = orientation

    stats = {"extensions": [0, 0], "blocked_by_other": [0, 0]}
    for orientation in (first, 1 - first):
        dr, dc = ORIENTATION_STEPS[orientation]
        for bar in bars[orientation]:
            rays = (
                (bar.cells[0], (-dr, -dc)),
                (bar.cells[-1], (dr, dc)),
            )
            for (r, c), (step_r, step_c) in rays:
                rr, cc = r + step_r, c + step_c
                while _in_bounds(shape, rr, cc):
                    if occupied[rr, cc]:
                        if owner[rr, cc] != orientation:
                            stats["blocked_by_other"][orientation] += 1
                        break
                    occupied[rr, cc] = True
                    owner[rr, cc] = orientation
                    stats["extensions"][orientation] += 1
                    rr += step_r
                    cc += step_c

    return owner, stats


def _colored_output(owner: np.ndarray, first: int) -> np.ndarray:
    out = np.zeros(owner.shape, dtype=int)
    out[owner == first] = FIRST_COLOR
    out[owner == 1 - first] = SECOND_COLOR
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend both diagonal-bar populations in majority-first order, then recolor them by that order.

    Input structure:
    - A black background contains one non-black input color forming one-cell-thick short diagonal bars.
    - Every bar has either down-right or down-left orientation; both orientations occur, and bars of the same orientation lie on distinct diagonals.
    - Each foreground pixel belongs to exactly one bar, and the two orientations have unequal total pixel populations.

    Rule:
    - The orientation with more original pixels moves first. Extend every one of its bars from both endpoints along its own grid diagonal.
    - A ray stops at the canvas edge or immediately before the first already occupied cell. Then extend the minority orientation under the same rule, so the first population's original and added cells block it.
    - Color all first-population original and extension cells teal, all second-population cells red, and leave the background black.

    Key insight:
    - Occlusion priority is not tied to a fixed diagonal direction: it must be inferred anew from the two original pixel totals.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert 1 <= g.shape[0] <= 30 and 1 <= g.shape[1] <= 30, "ARC grids are at most 30x30"
    assert np.any(g == BACKGROUND), "expected a black background"

    input_colors = np.unique(g[g != BACKGROUND])
    assert len(input_colors) == 1, "expected exactly one non-black input color"
    bars = _extract_bars(g)
    populations = [sum(len(bar.cells) for bar in bars[o]) for o in (0, 1)]
    assert populations[0] != populations[1], "the two orientation populations must be unequal"
    first = 0 if populations[0] > populations[1] else 1

    owner, _ = _draw(g.shape, bars, first)
    return _colored_output(owner, first)


def _canonical_bar(orientation: int, cells: list[Coord]) -> Bar:
    return Bar(orientation, tuple(sorted(cells)))


def _line_key(bar: Bar) -> int:
    r, c = bar.cells[0]
    return r - c if bar.orientation == 0 else r + c


def _well_separated(candidate: Bar, bars: list[Bar]) -> bool:
    """Keep input objects visually separate and their decomposition unique."""
    if any(
        other.orientation == candidate.orientation and _line_key(other) == _line_key(candidate)
        for other in bars
    ):
        return False
    for r, c in candidate.cells:
        for other in bars:
            for rr, cc in other.cells:
                if max(abs(r - rr), abs(c - cc)) <= 1:
                    return False
    return True


def _crossing_pair(
    rng: np.random.Generator,
    shape: tuple[int, int],
    lengths: tuple[int, int],
) -> tuple[Bar, Bar] | None:
    """Place one bar of each orientation whose empty infinite diagonals cross."""
    h, w = shape
    for _ in range(300):
        cross_r = int(rng.integers(2, h - 2))
        cross_c = int(rng.integers(2, w - 2))
        made: list[Bar] = []
        for orientation, length in enumerate(lengths):
            dr, dc = ORIENTATION_STEPS[orientation]
            side = int(rng.choice((-1, 1)))
            gap = int(rng.integers(1, 3))
            cells = [
                (cross_r + side * (gap + i) * dr, cross_c + side * (gap + i) * dc)
                for i in range(length)
            ]
            if not all(_in_bounds(shape, *cell) for cell in cells):
                break
            made.append(_canonical_bar(orientation, cells))
        if len(made) == 2 and _well_separated(made[1], [made[0]]):
            return made[0], made[1]
    return None


def _random_bar(
    rng: np.random.Generator,
    shape: tuple[int, int],
    orientation: int,
    length: int,
) -> Bar | None:
    h, w = shape
    dr, dc = ORIENTATION_STEPS[orientation]
    for _ in range(250):
        r = int(rng.integers(1, h - 1))
        c = int(rng.integers(1, w - 1))
        cells = [(r + i * dr, c + i * dc) for i in range(length)]
        if all(_in_bounds(shape, *cell) for cell in cells):
            return _canonical_bar(orientation, cells)
    return None


def _choose_plan(rng: np.random.Generator, desired_first: int) -> tuple[list[int], list[int]]:
    """Choose multiple varied bars with the requested unique population winner."""
    for _ in range(500):
        counts = (int(rng.integers(1, 4)), int(rng.integers(1, 4)))
        if sum(counts) < 3:
            continue
        lengths = [
            [int(x) for x in rng.integers(2, 6, size=counts[orientation])]
            for orientation in (0, 1)
        ]
        totals = [sum(group) for group in lengths]
        winner = 0 if totals[0] > totals[1] else 1 if totals[1] > totals[0] else -1
        if winner == desired_first and abs(totals[0] - totals[1]) <= 6:
            return lengths[0], lengths[1]
    raise RuntimeError("could not choose unequal diagonal populations")


def _independent_input_checks(grid: np.ndarray, intended: list[Bar]) -> bool:
    """Check construction facts without trusting the final colored output."""
    try:
        parsed = _extract_bars(grid)
    except AssertionError:
        return False
    expected = {
        (bar.orientation, frozenset(bar.cells))
        for bar in intended
    }
    actual = {
        (bar.orientation, frozenset(bar.cells))
        for orientation in (0, 1)
        for bar in parsed[orientation]
    }
    return expected == actual


def _generate_attempt(rng: np.random.Generator, desired_first: int) -> np.ndarray | None:
    shape = (int(rng.integers(15, 26)), int(rng.integers(15, 26)))
    lengths = _choose_plan(rng, desired_first)
    core = _crossing_pair(rng, shape, (lengths[0][0], lengths[1][0]))
    if core is None:
        return None
    bars: list[Bar] = [core[0], core[1]]

    for orientation in (0, 1):
        for length in lengths[orientation][1:]:
            placed = None
            for _ in range(300):
                candidate = _random_bar(rng, shape, orientation, length)
                if candidate is not None and _well_separated(candidate, bars):
                    placed = candidate
                    break
            if placed is None:
                return None
            bars.append(placed)

    input_color = int(rng.choice(INPUT_COLORS))
    grid = np.zeros(shape, dtype=int)
    for bar in bars:
        for cell in bar.cells:
            grid[cell] = input_color

    if not _independent_input_checks(grid, bars):
        return None
    parsed = _extract_bars(grid)
    totals = [sum(len(bar.cells) for bar in parsed[o]) for o in (0, 1)]
    inferred_first = 0 if totals[0] > totals[1] else 1
    if totals[0] == totals[1] or inferred_first != desired_first:
        return None

    correct_owner, stats = _draw(shape, parsed, inferred_first)
    reversed_owner, _ = _draw(shape, parsed, 1 - inferred_first)
    correct = _colored_output(correct_owner, inferred_first)

    # Semantic anti-shortcut checks: both populations really extend, the second
    # is visibly blocked by the first, and reversing priority changes the geometry.
    if min(stats["extensions"]) <= 0:
        return None
    if stats["blocked_by_other"][1 - inferred_first] <= 0:
        return None
    if np.array_equal(correct_owner, reversed_owner):
        return None
    sensitive = int(np.count_nonzero(correct_owner != reversed_owner))
    if sensitive > max(12, grid.size // 10):
        return None
    if np.count_nonzero(correct) > int(grid.size * 0.42):
        return None
    if np.count_nonzero(correct == FIRST_COLOR) <= totals[inferred_first]:
        return None
    if np.count_nonzero(correct == SECOND_COLOR) <= totals[1 - inferred_first]:
        return None
    for bar in bars:
        expected_color = FIRST_COLOR if bar.orientation == inferred_first else SECOND_COLOR
        if any(correct[cell] != expected_color for cell in bar.cells):
            return None

    # Solver-oracle closure is the final check, not the only validation.
    try:
        if not np.array_equal(main(grid), correct):
            return None
    except AssertionError:
        return None
    return grid


_generator_rng = np.random.default_rng()
_next_majority = int(_generator_rng.integers(0, 2))
_emitted_outputs: set[tuple[tuple[int, int], bytes]] = set()


def generate_input() -> np.ndarray:
    """Generate a varied, non-degenerate input; consecutive calls alternate the majority orientation."""
    global _next_majority
    desired_first = _next_majority
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt(_generator_rng, desired_first)
        if grid is None:
            continue
        output = main(grid)
        signature = (output.shape, output.tobytes())
        if signature in _emitted_outputs:
            continue
        _emitted_outputs.add(signature)
        _next_majority = 1 - _next_majority
        return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
