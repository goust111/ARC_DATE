from __future__ import annotations

from collections import Counter
from itertools import combinations
import random

import numpy as np


BLACK = 0
ARC_COLORS = tuple(range(1, 10))
DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _background_color(grid: np.ndarray) -> int:
    """Return the unique most frequent color, which is the field color."""
    counts = Counter(map(int, grid.ravel()))
    largest = max(counts.values())
    winners = [color for color, count in counts.items() if count == largest]
    assert len(winners) == 1, "expected a unique dominant field color"
    bg = winners[0]
    assert bg != BLACK, "the dominant field must differ from the black marker"
    return bg


def _ray(
    shape: tuple[int, int], origin: tuple[int, int], direction: tuple[int, int]
) -> list[tuple[int, int]]:
    """Coordinates strictly beyond origin, ordered from near to far."""
    height, width = shape
    r, c = origin
    dr, dc = direction
    cells: list[tuple[int, int]] = []
    r += dr
    c += dc
    while 0 <= r < height and 0 <= c < width:
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Color four marker-centered arms by their outward breakpoint order, then erase each remaining tail.

    Input structure:
    - A 2-D grid is dominated by one nonblack field color and contains exactly one black marker pixel.
    - Colored singleton breakpoints may lie on the marker's row or column; other colored pixels lie off that cross as distractors.

    Rule:
    - On each of the four cardinal rays, read non-field pixels from the marker outward as ordered breakpoints.
    - Color every cell from the previous breakpoint (exclusive) through the next breakpoint (inclusive) with that next breakpoint's color.
    - Turn every cell beyond the last breakpoint black; if a ray has no breakpoint, turn its whole length black. Preserve the marker and all off-cross cells.

    Key insight:
    - Row/column membership selects the controls, while distance from the black marker determines their segment order independently on all four arms.
    """
    g = np.asarray(grid)
    assert g.ndim == 2, "expected a 2-D grid"
    assert g.size > 1, "expected a nontrivial grid"

    marker_coords = np.argwhere(g == BLACK)
    assert marker_coords.shape == (1, 2), "expected exactly one black marker pixel"
    marker = tuple(map(int, marker_coords[0]))
    bg = _background_color(g)

    out = g.copy()
    for direction in DIRECTIONS:
        cells = _ray(g.shape, marker, direction)
        breakpoints = [
            (index, int(g[r, c]))
            for index, (r, c) in enumerate(cells)
            if int(g[r, c]) != bg
        ]

        segment_start = 0
        for endpoint, color in breakpoints:
            assert color != BLACK, "black may occur only at the origin marker"
            for r, c in cells[segment_start : endpoint + 1]:
                out[r, c] = color
            segment_start = endpoint + 1

        for r, c in cells[segment_start:]:
            out[r, c] = BLACK

    out[marker] = BLACK
    return out


def _spaced_breakpoint_distances(length: int, count: int) -> list[int]:
    """Choose endpoints with visible segments and a nonempty tail."""
    if count == 0:
        return []
    choices = [
        option
        for option in combinations(range(2, length), count)
        if all(b - a >= 2 for a, b in zip(option, option[1:]))
    ]
    assert choices, "arm is too short for the requested breakpoint count"
    return list(random.choice(choices))


def _arm_color_sequence(count: int, palette: list[int]) -> list[int]:
    """Choose colors without giving adjacent segments the same visible color."""
    sequence: list[int] = []
    for _ in range(count):
        options = [color for color in palette if not sequence or color != sequence[-1]]
        sequence.append(random.choice(options))
    return sequence


def _semantic_validation(
    inp: np.ndarray,
    expected: np.ndarray,
    bg: int,
    marker: tuple[int, int],
    specs: dict[tuple[int, int], tuple[list[int], list[int]]],
    distractors: list[tuple[int, int]],
) -> None:
    """Check the stated semantics and reject shortcut-friendly generations."""
    assert inp.ndim == 2 and max(inp.shape) <= 30
    assert expected.shape == inp.shape
    assert np.count_nonzero(inp == BLACK) == 1 and int(inp[marker]) == BLACK
    assert _background_color(inp) == bg
    assert sorted(len(distances) for distances, _ in specs.values()) == [0, 1, 2, 3]

    cross = np.zeros(inp.shape, dtype=bool)
    cross[marker[0], :] = True
    cross[:, marker[1]] = True
    cross[marker] = False
    assert np.array_equal(inp[~cross], expected[~cross])

    all_breakpoints: list[tuple[int, int]] = []
    for direction in DIRECTIONS:
        cells = _ray(inp.shape, marker, direction)
        distances, colors = specs[direction]
        assert len(distances) == len(colors)
        assert all(b - a >= 2 for a, b in zip(distances, distances[1:]))
        assert all(a != b for a, b in zip(colors, colors[1:]))

        visible = [
            (distance, int(inp[cells[distance - 1]]))
            for distance in range(1, len(cells) + 1)
            if int(inp[cells[distance - 1]]) != bg
        ]
        assert visible == list(zip(distances, colors))

        for distance, cell in enumerate(cells, start=1):
            later = [
                color
                for endpoint, color in zip(distances, colors)
                if distance <= endpoint
            ]
            wanted = later[0] if later else BLACK
            assert int(expected[cell]) == wanted

        if distances:
            assert distances[0] >= 2
            assert distances[-1] < len(cells)
            assert all(int(expected[cell]) == BLACK for cell in cells[distances[-1] :])
            all_breakpoints.extend(cells[distance - 1] for distance in distances)
        else:
            assert all(int(expected[cell]) == BLACK for cell in cells)

    assert distractors
    assert all(r != marker[0] and c != marker[1] for r, c in distractors)
    assert all(int(expected[r, c]) == int(inp[r, c]) != bg for r, c in distractors)
    assert any(int(inp[r, c]) in {int(inp[p]) for p in all_breakpoints} for r, c in distractors)

    solved = main(inp)
    assert np.array_equal(solved, expected)
    assert not np.array_equal(solved, inp)

    # Every breakpoint must affect the result: removing any one changes at least
    # one segment, so the ordered multi-breakpoint mechanism cannot collapse to
    # merely using the nearest or farthest colored pixel.
    for cell in all_breakpoints:
        without_one = inp.copy()
        without_one[cell] = bg
        assert not np.array_equal(main(without_one), expected)


def _generate_attempt() -> np.ndarray:
    height = random.randint(18, 28)
    width = random.randint(18, 28)
    marker = (
        random.randint(7, height - 8),
        random.randint(7, width - 8),
    )

    bg = random.choice(ARC_COLORS)
    palette = random.sample([color for color in ARC_COLORS if color != bg], 4)
    inp = np.full((height, width), bg, dtype=int)
    expected = inp.copy()
    inp[marker] = BLACK
    expected[marker] = BLACK

    counts = [0, 1, 2, 3]
    random.shuffle(counts)
    specs: dict[tuple[int, int], tuple[list[int], list[int]]] = {}

    for direction, count in zip(DIRECTIONS, counts):
        cells = _ray(inp.shape, marker, direction)
        distances = _spaced_breakpoint_distances(len(cells), count)
        colors = _arm_color_sequence(count, palette)
        specs[direction] = (distances, colors)

        for distance, color in zip(distances, colors):
            inp[cells[distance - 1]] = color

        start = 0
        for distance, color in zip(distances, colors):
            for r, c in cells[start:distance]:
                expected[r, c] = color
            start = distance
        for r, c in cells[start:]:
            expected[r, c] = BLACK

    # Keep distractors well off both cross lines and separated from one another.
    candidates = [
        (r, c)
        for r in range(1, height - 1)
        for c in range(1, width - 1)
        if abs(r - marker[0]) >= 2 and abs(c - marker[1]) >= 2
    ]
    random.shuffle(candidates)
    distractors: list[tuple[int, int]] = []
    distractor_count = random.randint(4, 8)
    for cell in candidates:
        if all(max(abs(cell[0] - r), abs(cell[1] - c)) >= 2 for r, c in distractors):
            distractors.append(cell)
            if len(distractors) == distractor_count:
                break
    assert len(distractors) == distractor_count

    breakpoint_colors = [color for _distances, colors in specs.values() for color in colors]
    distractor_colors = [random.choice(breakpoint_colors)]
    distractor_colors.extend(random.choice(palette) for _ in range(distractor_count - 1))
    random.shuffle(distractor_colors)
    for (r, c), color in zip(distractors, distractor_colors):
        inp[r, c] = color
        expected[r, c] = color

    _semantic_validation(inp, expected, bg, marker, specs, distractors)
    return inp


def generate_input() -> np.ndarray:
    """Generate a diverse, nondegenerate instance satisfying all solver assumptions."""
    for _ in range(100):
        try:
            return _generate_attempt()
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid four-way color-ruler input")
