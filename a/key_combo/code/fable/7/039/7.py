"""ARC task 7: restore a shelf inferred from the water trail above it."""

from __future__ import annotations

from collections import Counter, deque

import numpy as np


def _horizontal_runs(mask: np.ndarray, min_length: int = 1) -> list[tuple[int, int, int]]:
    """Return maximal True runs as inclusive ``(row, first_col, last_col)`` tuples."""
    height, width = mask.shape
    runs: list[tuple[int, int, int]] = []
    for row in range(height):
        col = 0
        while col < width:
            if not bool(mask[row, col]):
                col += 1
                continue
            first = col
            while col + 1 < width and bool(mask[row, col + 1]):
                col += 1
            if col - first + 1 >= min_length:
                runs.append((row, first, col))
            col += 1
    return runs


def _run_containing(
    runs_by_row: dict[int, list[tuple[int, int]]], row: int, first: int, last: int
) -> tuple[int, int] | None:
    """Find the unique horizontal run on ``row`` containing an interval."""
    matches = [
        (a, b)
        for a, b in runs_by_row.get(row, [])
        if a <= first and last <= b
    ]
    assert len(matches) <= 1, "maximal runs cannot overlap"
    return matches[0] if matches else None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore the one erased horizontal shelf by causally inverting its unsupported water spread.

    Input structure:
    - A dominant background surrounds one water color and one shelf color.
    - Water starts at isolated top-edge sources, falls vertically, and forms horizontal
      spreads immediately above several horizontal shelf segments.
    - Exactly one spread has background, rather than a shelf, directly beneath it.

    Rule:
    - Infer from every visible shelf-water pair how far the water spread extends past
      the shelf at its left and right ends.
    - Find the unique horizontal water run lacking that inferred support, then paint
      shelf-colored cells one row below it after removing the learned end extensions.
    - Preserve every other input cell.

    Key insight:
    - The unsupported water run is evidence of an erased cause: its two overhanging
      end cells are water overflow, so only the inset interval is the missing shelf.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"
    height, _width = map(int, g.shape)

    counts = Counter(map(int, g.ravel()))
    bg = int(counts.most_common(1)[0][0])
    foreground = [int(color) for color in counts if int(color) != bg]
    assert len(foreground) == 2, "expected exactly water and shelf colors"

    top_colors = {int(color) for color in g[0] if int(color) != bg}
    assert len(top_colors) == 1, "top edge must structurally identify the water color"
    water = top_colors.pop()
    assert water in foreground
    shelf_colors = [color for color in foreground if color != water]
    assert len(shelf_colors) == 1
    shelf = shelf_colors[0]

    water_runs = _horizontal_runs(g == water)
    water_by_row: dict[int, list[tuple[int, int]]] = {}
    for row, first, last in water_runs:
        water_by_row.setdefault(row, []).append((first, last))

    # The complete shelves are the visible demonstrations of the spread geometry.
    extensions: list[tuple[int, int]] = []
    shelf_runs = _horizontal_runs(g == shelf)
    for row, first, last in shelf_runs:
        assert row > 0, "a shelf needs a water row above it"
        covering = _run_containing(water_by_row, row - 1, first, last)
        assert covering is not None, "every visible shelf must support a water spread"
        water_first, water_last = covering
        left_extension = first - water_first
        right_extension = water_last - last
        assert left_extension > 0 and right_extension > 0
        extensions.append((left_extension, right_extension))

    assert len(extensions) >= 2, "need multiple visible shelf-water examples"
    assert len(set(extensions)) == 1, "visible pairs must teach one unambiguous offset"
    left_extension, right_extension = extensions[0]

    missing: list[tuple[int, int, int]] = []
    supported = 0
    minimum_spread = left_extension + right_extension + 1
    for row, water_first, water_last in water_runs:
        if row + 1 >= height or water_last - water_first + 1 < minimum_spread:
            continue
        first = water_first + left_extension
        last = water_last - right_extension
        if first > last:
            continue
        below = g[row + 1, first : last + 1]
        if np.all(below == shelf):
            supported += 1
        elif np.all(below == bg):
            missing.append((row + 1, first, last))

    assert supported == len(shelf_runs), "each visible shelf must have exactly one spread"
    assert len(missing) == 1, "expected a unique unsupported horizontal water spread"

    out = g.copy()
    row, first, last = missing[0]
    out[row, first : last + 1] = shelf
    return out


def _simulate_water(
    height: int,
    width: int,
    sources: list[int],
    shelves: list[tuple[int, int, int]],
) -> np.ndarray:
    """Independently simulate the stated forward waterfall physics."""
    shelf_mask = np.zeros((height, width), dtype=bool)
    by_row: dict[int, list[tuple[int, int]]] = {}
    for row, first, last in shelves:
        shelf_mask[row, first : last + 1] = True
        by_row.setdefault(row, []).append((first, last))

    water = np.zeros((height, width), dtype=bool)
    queue = deque((0, col) for col in sources)
    seen: set[tuple[int, int]] = set()
    while queue:
        row, col = queue.popleft()
        if (row, col) in seen:
            continue
        seen.add((row, col))
        water[row, col] = True

        while row + 1 < height:
            if shelf_mask[row + 1, col]:
                segment = [
                    (first, last)
                    for first, last in by_row[row + 1]
                    if first <= col <= last
                ]
                assert len(segment) == 1
                first, last = segment[0]
                spread_first, spread_last = first - 1, last + 1
                assert 0 <= spread_first and spread_last < width
                water[row, spread_first : spread_last + 1] = True
                queue.append((row, spread_first))
                queue.append((row, spread_last))
                break
            row += 1
            water[row, col] = True
    return water


def _random_partition(rng: np.random.Generator, total: int, count: int) -> list[int]:
    """Randomly distribute ``total`` indistinguishable padding cells among slots."""
    parts = [0] * count
    for _ in range(total):
        parts[int(rng.integers(0, count))] += 1
    return parts


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    shelf_count = int(rng.integers(3, 5))
    if shelf_count == 4:
        # Four independent waterfall bands only fit within 30 columns when the
        # shelves are compact.  Vary one or two of them without biasing this
        # object-count branch toward near-certain rejection.
        lengths = [3] * shelf_count
        for index in rng.choice(shelf_count, size=int(rng.integers(1, 3)), replace=False):
            lengths[int(index)] = 4
        rng.shuffle(lengths)
    else:
        lengths = [int(rng.integers(3, 7)) for _ in range(shelf_count)]
    if len(set(lengths)) < 2:
        return None

    # Each shelf owns a disjoint column band: one overflow column on either side,
    # at least one blank outer margin, and at least two blank columns between bands.
    minimum_width = sum(length + 2 for length in lengths) + 2 * (shelf_count - 1) + 2
    if minimum_width > 30:
        return None
    width = int(rng.integers(minimum_width, 31))
    height = int(rng.integers(14, 31))

    mandatory_gaps = [1] + [2] * (shelf_count - 1) + [1]
    extras = _random_partition(rng, width - minimum_width, shelf_count + 1)
    gaps = [base + extra for base, extra in zip(mandatory_gaps, extras)]

    shelves: list[tuple[int, int, int]] = []
    sources: list[int] = []
    cursor = gaps[0]
    for index, length in enumerate(lengths):
        spread_first = cursor
        first = spread_first + 1
        last = first + length - 1
        shelf_row = int(rng.integers(5, height - 2))
        shelves.append((shelf_row, first, last))
        sources.append(int(rng.integers(first, last + 1)))
        cursor = last + 2 + gaps[index + 1]

    if len({row for row, _first, _last in shelves}) < 2:
        return None

    hidden_index = int(rng.integers(0, shelf_count))
    hidden_length = lengths[hidden_index]
    if all(length == hidden_length for i, length in enumerate(lengths) if i != hidden_index):
        return None

    bg, water_color, shelf_color = map(int, rng.choice(10, size=3, replace=False))
    water_mask = _simulate_water(height, width, sources, shelves)
    shelf_mask = np.zeros((height, width), dtype=bool)
    for row, first, last in shelves:
        shelf_mask[row, first : last + 1] = True
    assert not np.any(water_mask & shelf_mask)

    complete = np.full((height, width), bg, dtype=int)
    complete[water_mask] = water_color
    complete[shelf_mask] = shelf_color

    inp = complete.copy()
    hidden_row, hidden_first, hidden_last = shelves[hidden_index]
    inp[hidden_row, hidden_first : hidden_last + 1] = bg

    # Semantic checks are deliberately independent of merely calling main(inp).
    visible_shelves = [s for i, s in enumerate(shelves) if i != hidden_index]
    visible_mask = np.zeros_like(shelf_mask)
    for row, first, last in visible_shelves:
        visible_mask[row, first : last + 1] = True
    if not np.array_equal(inp == shelf_color, visible_mask):
        return None
    unsupported = [
        (row, first, last)
        for i, (row, first, last) in enumerate(shelves)
        if np.all(inp[row, first : last + 1] == bg)
    ]
    if unsupported != [(hidden_row, hidden_first, hidden_last)]:
        return None
    if not np.array_equal(_simulate_water(height, width, sources, shelves), inp == water_color):
        return None

    try:
        solved = main(inp)
    except (AssertionError, ValueError):
        return None
    if not np.array_equal(solved, complete):
        return None
    changed = solved != inp
    if int(changed.sum()) != hidden_last - hidden_first + 1:
        return None
    if not np.all(changed[hidden_row, hidden_first : hidden_last + 1]):
        return None
    return inp


def generate_input() -> np.ndarray:
    """Generate a varied, nondegenerate instance satisfying the inverse-causality rule."""
    rng = np.random.default_rng()
    for _ in range(400):
        result = _generate_attempt(rng)
        if result is not None:
            return result
    raise RuntimeError("failed to generate a valid invisible-shelf instance")
