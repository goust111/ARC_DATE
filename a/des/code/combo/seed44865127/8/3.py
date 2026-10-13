import random
from collections import Counter

import numpy as np


PALETTE = tuple(range(10))
MIN_COLORS = 2
MAX_COLORS = 5
MIN_WIDTH = 7
MAX_WIDTH = 18
MIN_DATA_ROWS = 2
MAX_DATA_ROWS = 6
MAX_ATTEMPTS = 400
SCRAMBLE_ATTEMPTS = 100


def _row_runs(row: np.ndarray) -> list[tuple[int, int]]:
    values = [int(v) for v in np.asarray(row, dtype=int).tolist()]
    assert values, "rows must be non-empty"

    runs: list[tuple[int, int]] = []
    current = values[0]
    length = 1
    for value in values[1:]:
        if value == current:
            length += 1
        else:
            runs.append((current, length))
            current = value
            length = 1
    runs.append((current, length))
    return runs


def _build_grouped_row(colors: list[int], counts: list[int]) -> np.ndarray:
    pieces = [np.full(count, color, dtype=int) for color, count in zip(colors, counts)]
    return np.concatenate(pieces)


def _run_count(row: np.ndarray) -> int:
    return len(_row_runs(row))


def _greedy_scramble(colors: list[int], counts: list[int]) -> np.ndarray:
    remaining = list(counts)
    row: list[int] = []
    last_index = -1
    total = sum(remaining)

    for _ in range(total):
        candidates = [i for i, count in enumerate(remaining) if count > 0 and i != last_index]
        if not candidates:
            candidates = [i for i, count in enumerate(remaining) if count > 0]
        max_remaining = max(remaining[i] for i in candidates)
        tied = [i for i in candidates if remaining[i] == max_remaining]
        index = random.choice(tied)
        row.append(colors[index])
        remaining[index] -= 1
        last_index = index

    return np.array(row, dtype=int)


def _scramble_row(colors: list[int], counts: list[int]) -> np.ndarray:
    grouped = _build_grouped_row(colors, counts)
    items = grouped.tolist()
    min_runs = min(len(grouped), len(colors) + 1)

    for _ in range(SCRAMBLE_ATTEMPTS):
        shuffled = items[:]
        random.shuffle(shuffled)
        candidate = np.array(shuffled, dtype=int)
        if not np.array_equal(candidate, grouped) and _run_count(candidate) >= min_runs:
            return candidate

    for _ in range(SCRAMBLE_ATTEMPTS):
        candidate = _greedy_scramble(colors, counts)
        if not np.array_equal(candidate, grouped) and _run_count(candidate) >= min_runs:
            return candidate

    raise AssertionError("failed to scramble a data row")


def _positive_composition(total: int, parts: int) -> list[int]:
    assert total >= parts >= 1
    cuts = sorted(random.sample(range(1, total), parts - 1))
    bounds = [0, *cuts, total]
    return [bounds[i + 1] - bounds[i] for i in range(parts)]


def _mutate_counts(template_counts: list[int]) -> list[int]:
    donors = [i for i, count in enumerate(template_counts) if count > 1]
    assert donors, "need at least one donor color to create a varied row"

    donor = random.choice(donors)
    receivers = [i for i in range(len(template_counts)) if i != donor]
    receiver = random.choice(receivers)
    amount = random.randint(1, template_counts[donor] - 1)

    counts = list(template_counts)
    counts[donor] -= amount
    counts[receiver] += amount
    assert counts != template_counts
    assert all(count >= 1 for count in counts)
    assert any(new > old for new, old in zip(counts, template_counts))
    return counts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder each data row by the visible left-to-right color order from the template row.

    Input structure:
    - The first row is a template made of contiguous color blocks.
    - The second row is a uniform background separator.
    - Remaining rows contain the same color set in a different arrangement.

    Rule:
    - Read the template colors from left to right.
    - For each data row, count how many pixels it has of each template color.
    - Rebuild the row by concatenating those colors in template order.
    - Drop the template row and the separator row.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height >= 3 and width >= 2, "expected template, separator, and data rows"

    separator = g[1]
    assert np.all(separator == separator[0]), "expected a solid separator row"

    template_runs = _row_runs(g[0])
    template_colors = [color for color, _length in template_runs]
    assert len(set(template_colors)) == len(template_colors), "template colors must appear in one block each"
    valid_colors = set(template_colors)

    out = np.empty((height - 2, width), dtype=int)
    for row_index, row in enumerate(g[2:]):
        row_colors = set(map(int, np.unique(row)))
        assert row_colors == valid_colors, "each data row must use the same color set as the template"

        counts = Counter(map(int, row.tolist()))
        rebuilt = _build_grouped_row(template_colors, [counts[color] for color in template_colors])
        assert rebuilt.shape == (width,)
        out[row_index] = rebuilt

    return out


def _gen_attempt() -> np.ndarray | None:
    num_colors = random.randint(MIN_COLORS, MAX_COLORS)
    width = random.randint(max(MIN_WIDTH, num_colors + 2), MAX_WIDTH)
    data_rows = random.randint(MIN_DATA_ROWS, MAX_DATA_ROWS)

    bg = random.choice(PALETTE)
    colors = random.sample([color for color in PALETTE if color != bg], num_colors)

    template_counts = _positive_composition(width, num_colors)
    if max(template_counts) == 1:
        return None

    template = _build_grouped_row(colors, template_counts)
    separator = np.full(width, bg, dtype=int)

    branch_types = ["exact", "varied"]
    branch_types.extend(random.choice(("exact", "varied")) for _ in range(data_rows - 2))
    random.shuffle(branch_types)

    rows: list[np.ndarray] = []
    expected_rows: list[np.ndarray] = []
    saw_exact = False
    saw_varied = False

    for branch in branch_types:
        if branch == "exact":
            counts = list(template_counts)
            saw_exact = True
        else:
            counts = _mutate_counts(template_counts)
            saw_varied = True

        row = _scramble_row(colors, counts)
        rows.append(row)
        expected_rows.append(_build_grouped_row(colors, counts))

    if not (saw_exact and saw_varied):
        return None

    grid = np.vstack([template, separator, *rows]).astype(int)
    expected = np.vstack(expected_rows).astype(int)

    if np.array_equal(expected, grid[2:]):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
