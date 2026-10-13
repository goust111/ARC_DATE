from __future__ import annotations

import random

import numpy as np


GRAY = 5
PALETTE = tuple(range(10))


def _is_consecutive(values: np.ndarray) -> bool:
    values = np.sort(values)
    return bool(values.size <= 1 or np.all(np.diff(values) == 1))


def _is_straight_line(coords: np.ndarray) -> bool:
    """Recognize a contiguous horizontal, vertical, or 45-degree line."""
    if coords.ndim != 2 or coords.shape[0] == 0:
        return False
    rows, cols = coords[:, 0], coords[:, 1]
    if np.unique(rows).size == 1:
        return _is_consecutive(cols)
    if np.unique(cols).size == 1:
        return _is_consecutive(rows)
    if np.unique(rows - cols).size == 1 or np.unique(rows + cols).size == 1:
        return (
            np.unique(rows).size == len(coords)
            and np.unique(cols).size == len(coords)
            and _is_consecutive(rows)
        )
    return False


def _parse(grid: np.ndarray) -> tuple[int, int, list[tuple[int, int]]]:
    """Return background, separator row, and (top row, color) roster."""
    g = np.asarray(grid)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"

    separator_rows = np.flatnonzero(np.all(g == GRAY, axis=1))
    assert len(separator_rows) == 1, "expected exactly one all-gray separator row"
    separator = int(separator_rows[0])
    assert 0 < separator < g.shape[0] - 1, "separator must have regions above and below"

    lower = g[separator + 1 :]
    bg = int(lower[0, 0])
    assert bg != GRAY and np.all(lower == bg), "lower region must be uniform background"

    upper = g[:separator]
    colors = [int(c) for c in np.unique(upper) if int(c) not in (bg, GRAY)]
    assert len(colors) >= 2, "expected at least two differently colored line objects"
    assert not np.any(upper == GRAY), "gray is reserved for the separator"

    roster: list[tuple[int, int]] = []
    for color in colors:
        coords = np.argwhere(upper == color)
        assert _is_straight_line(coords), "each color must form one straight line segment"
        roster.append((int(coords[:, 0].min()), color))

    tops = [top for top, _ in roster]
    assert len(tops) == len(set(tops)), "line segments must have distinct topmost rows"
    roster.sort(key=lambda item: item[0])
    return bg, separator, roster


def main(grid: np.ndarray) -> np.ndarray:
    """
    Turn the colored line roster above a gray separator into cyclic weighted rows below it.

    Input structure:
    A single all-gray row separates the grid. Below it is a uniform background;
    above it, each non-background color forms one horizontal, vertical, or 45-degree
    one-cell-wide line segment, and the segments have distinct topmost rows.

    Rule:
    Sort the line colors by their topmost row. Expand each color into as many
    consecutive schedule entries as its segment has cells. Copy the input, then
    fill every row below the separator with the next schedule color, cycling back
    to the first entry whenever the weighted schedule ends.

    Key insight:
    The upper lines are a weighted roster: vertical position supplies order while
    segment length supplies duration; horizontal placement and orientation do not.
    """
    g = np.asarray(grid)
    bg, separator, roster = _parse(g)

    schedule: list[int] = []
    for _, color in roster:
        schedule.extend([color] * int(np.count_nonzero(g[:separator] == color)))
    assert schedule

    out = g.copy()
    for offset, row in enumerate(range(separator + 1, g.shape[0])):
        out[row, :] = schedule[offset % len(schedule)]
    assert bg not in schedule
    return out


def _line_coords(top: int, center_col: int, length: int, kind: str) -> list[tuple[int, int]]:
    if kind == "H":
        start = center_col - (length - 1) // 2
        return [(top, start + i) for i in range(length)]
    if kind == "V":
        return [(top + i, center_col) for i in range(length)]
    if kind == "DR":
        start = center_col - (length - 1) // 2
        return [(top + i, start + i) for i in range(length)]
    if kind == "DL":
        start = center_col + (length - 1) // 2
        return [(top + i, start - i) for i in range(length)]
    raise ValueError(kind)


def _independent_expected(grid: np.ndarray) -> np.ndarray:
    """A deliberately separate oracle used to reject semantically bad samples."""
    sep = int(np.flatnonzero(np.all(grid == GRAY, axis=1))[0])
    bg = int(grid[sep + 1, 0])
    records = []
    for color in np.unique(grid[:sep]):
        color = int(color)
        if color != bg:
            cells = np.argwhere(grid[:sep] == color)
            records.append((int(cells[:, 0].min()), color, len(cells)))
    records.sort()
    agenda = [color for _, color, size in records for _ in range(size)]
    expected = grid.copy()
    for i, r in enumerate(range(sep + 1, grid.shape[0])):
        expected[r] = agenda[i % len(agenda)]
    return expected


def generate_input() -> np.ndarray:
    """Generate a varied, nondegenerate weighted-roster instance."""
    for _ in range(300):
        n = random.choice((3, 3, 4))
        lengths = [random.randint(2, 4) for _ in range(n)]
        if len(set(lengths)) == 1:
            lengths[random.randrange(n)] = 2 if lengths[0] != 2 else 4
        period = sum(lengths)

        upper_h = random.randint(9, 12)
        extra_max = min(5, 29 - upper_h - period)
        if extra_max < 2:
            continue
        lower_h = period + random.randint(2, extra_max)
        height = upper_h + 1 + lower_h
        width = random.randint(6 * n + 1, min(30, 6 * n + 5))

        bg = random.choice(tuple(c for c in PALETTE if c != GRAY))
        colors = random.sample(tuple(c for c in PALETTE if c not in (bg, GRAY)), n)
        grid = np.full((height, width), bg, dtype=int)
        grid[upper_h, :] = GRAY

        # Every sample contains horizontal, vertical, and diagonal examples.
        kinds = ["H", "V", random.choice(("DR", "DL"))]
        if n == 4:
            kinds.append("DL" if kinds[2] == "DR" else "DR")
        random.shuffle(kinds)

        top_pool = list(range(1, upper_h - max(lengths)))
        if len(top_pool) < n:
            continue
        tops = sorted(random.sample(top_pool, n))

        # Assign top-to-bottom objects to a non-identity permutation of column slots.
        slot_centers = [3 + 6 * i for i in range(n)]
        slot_order = list(range(n))
        while slot_order == list(range(n)):
            random.shuffle(slot_order)

        occupied: set[tuple[int, int]] = set()
        valid = True
        for rank in range(n):
            coords = _line_coords(tops[rank], slot_centers[slot_order[rank]], lengths[rank], kinds[rank])
            if any(not (0 <= r < upper_h and 0 <= c < width) for r, c in coords):
                valid = False
                break
            halo = {(r + dr, c + dc) for r, c in occupied for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
            if any(cell in halo for cell in coords):
                valid = False
                break
            for r, c in coords:
                grid[r, c] = colors[rank]
            occupied.update(coords)
        if not valid:
            continue

        # Semantic guards: left-to-right is a false shortcut, lengths matter, and
        # the bottom displays a complete period plus the beginning of a repeat.
        left_order = sorted(
            range(n), key=lambda i: int(np.argwhere(grid[:upper_h] == colors[i])[:, 1].min())
        )
        if left_order == list(range(n)) or lower_h <= period:
            continue
        expected = _independent_expected(grid)
        if not np.array_equal(main(grid), expected):
            continue
        return grid

    raise RuntimeError("failed to generate a valid weighted line-roster input")
