from __future__ import annotations

from collections import Counter
import random

import numpy as np


Coord = tuple[int, int]


def _orbit(r: int, c: int, side: int) -> tuple[Coord, Coord, Coord, Coord]:
    """Return one orbit in TL, TR, BL, BR order in full-canvas coordinates."""
    return (
        (r, c),
        (r, side - 1 - c),
        (side - 1 - r, c),
        (side - 1 - r, side - 1 - c),
    )


def _unique_background(grid: np.ndarray) -> int:
    counts = Counter(map(int, grid.flat))
    largest = max(counts.values())
    modes = [color for color, count in counts.items() if count == largest]
    assert len(modes) == 1, "expected one structurally dominant background color"
    return int(modes[0])


def main(grid: np.ndarray) -> np.ndarray:
    """Extract scattered minority residuals and fold them into the upper-left quadrant.

    Input structure:
    - The input is an even-sided square canvas with one dominant background color.
    - Horizontal and vertical reflections partition it into four-cell mirror orbits.
    - Every orbit is either uniform or has exactly one minority-colored cell against three equal cells.

    Rule:
    - Make a background-filled canvas whose side is half the input side.
    - For each mirror orbit, ignore uniform orbits; in a nonuniform orbit, find the color occurring once.
    - Paint that minority cell's own color at the orbit's upper-left folded coordinate.

    Key insight:
    - Two-axis symmetry is a filter for the underlying texture: only three-to-one orbit residuals are signal,
      and folding residuals from all four quadrants reconstructs the otherwise incomplete hidden figure.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    height, width = map(int, g.shape)
    assert height == width and height % 2 == 0, "expected an even-sided square canvas"

    half = height // 2
    background = _unique_background(g)
    out = np.full((half, half), background, dtype=int)

    for r in range(half):
        for c in range(half):
            values = [int(g[rr, cc]) for rr, cc in _orbit(r, c, height)]
            counts = Counter(values)
            if len(counts) == 1:
                continue
            assert sorted(counts.values()) == [1, 3], (
                "each nonuniform mirror orbit must contain one minority and three equal cells"
            )
            minority = next(color for color, count in counts.items() if count == 1)
            out[r, c] = int(minority)

    return out


# Generator constraints.  Every generated target uses all four source quadrants,
# while its colors also occur in ordinary symmetric texture orbits.
_HALF_SIDE_RANGE = (5, 10)
_MIN_TARGET_AREA = 8
_MAX_TARGET_AREA = 18
_MAX_GENERATION_ATTEMPTS = 400
_PALETTE = tuple(range(10))
_SEEN_OUTPUTS: set[tuple[tuple[int, int], bytes]] = set()


def _neighbors4(cell: Coord, side: int) -> list[Coord]:
    r, c = cell
    return [
        (rr, cc)
        for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
        if 0 <= rr < side and 0 <= cc < side
    ]


def _axis_asymmetric(cells: set[Coord]) -> bool:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    horizontal = {(r0 + r1 - r, c) for r, c in cells}
    vertical = {(r, c0 + c1 - c) for r, c in cells}
    return cells != horizontal and cells != vertical


def _make_target_cells(side: int) -> set[Coord] | None:
    upper = min(_MAX_TARGET_AREA, side * side // 2)
    if upper < _MIN_TARGET_AREA:
        return None
    target_area = random.randint(_MIN_TARGET_AREA, upper)
    cells: set[Coord] = {(random.randrange(side), random.randrange(side))}

    while len(cells) < target_area:
        frontier = {
            neighbor
            for cell in cells
            for neighbor in _neighbors4(cell, side)
            if neighbor not in cells
        }
        if not frontier:
            return None
        cells.add(random.choice(tuple(frontier)))

    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    bbox_area = (max(rows) - min(rows) + 1) * (max(cols) - min(cols) + 1)
    if len(set(rows)) < 3 or len(set(cols)) < 3:
        return None
    if len(cells) == bbox_area or not _axis_asymmetric(cells):
        return None
    return cells


def _color_target(cells: set[Coord], signal_colors: tuple[int, int]) -> dict[Coord, int]:
    ordered = list(cells)
    random.shuffle(ordered)
    colors = [signal_colors[0], signal_colors[0], signal_colors[1], signal_colors[1]]
    colors.extend(random.choice(signal_colors) for _ in range(len(ordered) - len(colors)))
    random.shuffle(colors)
    return dict(zip(ordered, colors))


def _make_base(
    side: int,
    background: int,
    palette: tuple[int, int, int, int],
    target: dict[Coord, int],
) -> np.ndarray:
    base = np.full((side, side), background, dtype=int)
    non_background = list(palette[1:])
    outside = [(r, c) for r in range(side) for c in range(side) if (r, c) not in target]
    random.shuffle(outside)

    texture_count = max(len(non_background), random.randint(side, max(side, side * side // 4)))
    texture_cells = outside[:texture_count]
    for cell, color in zip(texture_cells, non_background):
        base[cell] = color
    for cell in texture_cells[len(non_background) :]:
        base[cell] = random.choice(non_background)

    # A target residual must differ from its three-cell symmetric baseline.
    for cell, signal in target.items():
        choices = [color for color in palette if color != signal]
        weights = [5 if color == background else 1 for color in choices]
        base[cell] = random.choices(choices, weights=weights, k=1)[0]

    # Make the baseline under the target visibly nonuniform, so the signal is not
    # explainable as painting over one fixed-color substrate.
    target_base_colors = {int(base[cell]) for cell in target}
    if len(target_base_colors) < 2:
        cell = next(iter(target))
        alternatives = [color for color in palette if color not in (target[cell], base[cell])]
        base[cell] = random.choice(alternatives)
    return base


def _reflect_base(base: np.ndarray) -> np.ndarray:
    top = np.concatenate((base, np.fliplr(base)), axis=1)
    return np.concatenate((top, np.flipud(top)), axis=0)


def _quadrant_assignment(cells: set[Coord]) -> dict[Coord, int]:
    ordered = list(cells)
    random.shuffle(ordered)
    quadrants = [0, 1, 2, 3, 0, 1, 2, 3]
    quadrants.extend(random.randrange(4) for _ in range(len(ordered) - len(quadrants)))
    random.shuffle(quadrants)
    return dict(zip(ordered, quadrants))


def _expected_output(side: int, background: int, target: dict[Coord, int]) -> np.ndarray:
    expected = np.full((side, side), background, dtype=int)
    for cell, color in target.items():
        expected[cell] = int(color)
    return expected


def _semantically_valid(
    grid: np.ndarray,
    expected: np.ndarray,
    base: np.ndarray,
    target: dict[Coord, int],
    assignments: dict[Coord, int],
    background: int,
    signal_colors: tuple[int, int],
) -> bool:
    """Validate the stated mechanism independently of the solver implementation."""
    side = expected.shape[0]
    full_side = grid.shape[0]
    if grid.shape != (2 * side, 2 * side) or full_side > 30:
        return False
    if _unique_background(grid) != background:
        return False
    if set(np.unique(expected)) != {background, *signal_colors}:
        return False

    quadrant_counts = Counter(assignments.values())
    if any(quadrant_counts[q] < 2 for q in range(4)):
        return False
    if any(quadrant_counts[q] == len(target) for q in range(4)):
        return False

    seen_residuals: dict[Coord, int] = {}
    majority_colors: set[int] = set()
    for r in range(side):
        for c in range(side):
            orbit = _orbit(r, c, full_side)
            values = [int(grid[rr, cc]) for rr, cc in orbit]
            counts = Counter(values)
            baseline = int(base[r, c])
            if (r, c) in target:
                signal = int(target[(r, c)])
                if counts != Counter({baseline: 3, signal: 1}):
                    return False
                if values[assignments[(r, c)]] != signal:
                    return False
                seen_residuals[(r, c)] = signal
                majority_colors.add(baseline)
            elif counts != Counter({baseline: 4}):
                return False

    # Both signal colors must also be ordinary majority/texture colors somewhere;
    # hence color identity alone cannot reveal which cells belong in the answer.
    clean_base_colors = {
        int(base[r, c])
        for r in range(side)
        for c in range(side)
        if (r, c) not in target
    }
    if not set(signal_colors) <= clean_base_colors:
        return False
    if len(majority_colors) < 2 or seen_residuals != target:
        return False
    if not np.array_equal(_expected_output(side, background, seen_residuals), expected):
        return False

    try:
        actual = main(grid)
    except (AssertionError, ValueError):
        return False
    return np.array_equal(actual, expected)


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    half_side = random.randint(*_HALF_SIDE_RANGE)
    target_cells = _make_target_cells(half_side)
    if target_cells is None:
        return None

    chosen = random.sample(_PALETTE, 4)
    background = chosen[0]
    signal_colors = (chosen[1], chosen[2])
    palette = tuple(chosen)
    target = _color_target(target_cells, signal_colors)
    base = _make_base(half_side, background, palette, target)
    grid = _reflect_base(base)

    assignments = _quadrant_assignment(target_cells)
    full_side = 2 * half_side
    for cell, signal in target.items():
        r, c = cell
        rr, cc = _orbit(r, c, full_side)[assignments[cell]]
        grid[rr, cc] = int(signal)

    expected = _expected_output(half_side, background, target)
    if not _semantically_valid(
        grid, expected, base, target, assignments, background, signal_colors
    ):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        generated = _gen_attempt()
        if generated is None:
            continue
        grid, expected = generated
        output_key = (tuple(map(int, expected.shape)), expected.tobytes())
        if output_key in _SEEN_OUTPUTS:
            continue
        _SEEN_OUTPUTS.add(output_key)
        return grid
    raise RuntimeError("failed to generate a valid symmetric-residual folding input")
