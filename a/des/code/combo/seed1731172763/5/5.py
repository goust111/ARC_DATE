from __future__ import annotations

import random
from collections import Counter

import numpy as np


BACKGROUND = 8  # azure
SEPARATOR = 5  # grey
ARROW_COLORS = (0, 1, 2, 3, 4, 6, 7, 9)
DIRECTIONS = ("U", "R", "D", "L")
PRIORITY = ("U", "R", "D", "L")
SHIFT_BY_DIRECTION = {
    "U": (-1, 0),
    "R": (0, 1),
    "D": (1, 0),
    "L": (0, -1),
}
ARROW_SHAPES = {
    "U": frozenset({(0, 1), (1, 0), (1, 1), (1, 2)}),
    "R": frozenset({(0, 0), (1, 0), (1, 1), (2, 0)}),
    "D": frozenset({(0, 0), (0, 1), (0, 2), (1, 1)}),
    "L": frozenset({(0, 1), (1, 0), (1, 1), (2, 1)}),
}
ARROW_SIZE = {
    direction: (
        max(r for r, _c in cells) + 1,
        max(c for _r, c in cells) + 1,
    )
    for direction, cells in ARROW_SHAPES.items()
}

PANEL_HEIGHT_RANGE = (5, 8)
PANEL_WIDTH_RANGE = (5, 8)
PANEL_MARGIN = 1
MAX_GENERATION_ATTEMPTS = 300


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _separator_indices(grid: np.ndarray, bg: int) -> tuple[list[int], list[int], int]:
    rows = [
        r
        for r in range(grid.shape[0])
        if int(grid[r, 0]) != bg and bool(np.all(grid[r] == int(grid[r, 0])))
    ]
    cols = [
        c
        for c in range(grid.shape[1])
        if int(grid[0, c]) != bg and bool(np.all(grid[:, c] == int(grid[0, c])))
    ]
    assert len(rows) == 2 and len(cols) == 2, "expected two separator rows and two separator columns"

    sep_color = int(grid[rows[0], 0])
    assert all(int(grid[r, 0]) == sep_color for r in rows), "separator rows must share one color"
    assert all(int(grid[0, c]) == sep_color for c in cols), "separator columns must share one color"
    return rows, cols, sep_color


def _segments(length: int, cuts: list[int]) -> list[tuple[int, int]]:
    assert len(cuts) == 2
    a, b = cuts
    parts = [(0, a), (a + 1, b), (b + 1, length)]
    sizes = [end - start for start, end in parts]
    assert min(sizes) > 0 and len(set(sizes)) == 1, "panel sizes must be equal"
    return parts


def _direction_from_panel(panel: np.ndarray, bg: int) -> str:
    coords = np.argwhere(panel != bg)
    assert coords.shape == (4, 2), "each panel must contain exactly four arrow pixels"

    colors = {int(panel[r, c]) for r, c in coords}
    assert len(colors) == 1, "arrow pixels in one panel must share one color"

    norm = coords - coords.min(axis=0)
    shape = frozenset((int(r), int(c)) for r, c in norm)
    for direction, template in ARROW_SHAPES.items():
        if shape == template:
            return direction
    raise AssertionError("panel arrow must match one of the four supported directions")


def _winning_direction(votes: Counter[str]) -> str:
    best = max(votes.values())
    for direction in PRIORITY:
        if votes[direction] == best:
            return direction
    raise AssertionError("unreachable")


def _shift_grid(grid: np.ndarray, direction: str) -> np.ndarray:
    dr, dc = SHIFT_BY_DIRECTION[direction]
    return np.roll(grid, shift=(dr, dc), axis=(0, 1)).copy()


def main(grid: np.ndarray) -> np.ndarray:
    """
    In the 3x3 panel grid, each panel's arrow votes for a direction.
    Count the votes and cyclically shift the whole grid by one pixel toward the winning direction.
    Ties are resolved in priority order: up, right, down, left.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _dominant_color(g)
    sep_rows, sep_cols, sep_color = _separator_indices(g, bg)
    assert sep_color != bg, "separator color must differ from the background"

    row_parts = _segments(g.shape[0], sep_rows)
    col_parts = _segments(g.shape[1], sep_cols)

    votes: Counter[str] = Counter({direction: 0 for direction in DIRECTIONS})
    for r0, r1 in row_parts:
        for c0, c1 in col_parts:
            direction = _direction_from_panel(g[r0:r1, c0:c1], bg)
            votes[direction] += 1

    return _shift_grid(g, _winning_direction(votes))


def _profile_groups() -> dict[tuple[str, str], list[tuple[int, int, int, int]]]:
    groups = {(direction, mode): [] for direction in DIRECTIONS for mode in ("unique", "tie")}
    for up in range(10):
        for right in range(10 - up):
            for down in range(10 - up - right):
                left = 9 - up - right - down
                counts = (up, right, down, left)
                if sum(count > 0 for count in counts) < 2:
                    continue
                vote_map = Counter(dict(zip(DIRECTIONS, counts)))
                winner = _winning_direction(vote_map)
                best = max(counts)
                mode = "tie" if sum(count == best for count in counts) > 1 else "unique"
                groups[(winner, mode)].append(counts)
    return groups


COUNT_PROFILES = _profile_groups()


def _sample_vote_list() -> tuple[list[str], str]:
    winner = random.choice(DIRECTIONS)
    modes = ["unique"]
    if COUNT_PROFILES[(winner, "tie")]:
        modes.append("tie")
    mode = random.choice(modes)
    counts = random.choice(COUNT_PROFILES[(winner, mode)])

    votes: list[str] = []
    for direction, count in zip(DIRECTIONS, counts):
        votes.extend([direction] * count)
    random.shuffle(votes)
    return votes, winner


def _sample_arrow_colors() -> list[int]:
    unique_count = random.randint(2, min(6, len(ARROW_COLORS)))
    chosen = random.sample(list(ARROW_COLORS), unique_count)
    colors = chosen + [random.choice(chosen) for _ in range(9 - unique_count)]
    random.shuffle(colors)
    return colors


def _panel_origin(index: int, panel_size: int) -> int:
    return index * (panel_size + 1)


def _draw_arrow(grid: np.ndarray, top: int, left: int, direction: str, color: int) -> None:
    for dr, dc in ARROW_SHAPES[direction]:
        rr = top + dr
        cc = left + dc
        assert int(grid[rr, cc]) == BACKGROUND
        grid[rr, cc] = int(color)


def _build_input(votes: list[str]) -> tuple[np.ndarray, str]:
    panel_h = random.randint(*PANEL_HEIGHT_RANGE)
    panel_w = random.randint(*PANEL_WIDTH_RANGE)
    grid_h = 3 * panel_h + 2
    grid_w = 3 * panel_w + 2

    grid = np.full((grid_h, grid_w), BACKGROUND, dtype=int)
    grid[panel_h, :] = SEPARATOR
    grid[2 * panel_h + 1, :] = SEPARATOR
    grid[:, panel_w] = SEPARATOR
    grid[:, 2 * panel_w + 1] = SEPARATOR

    colors = _sample_arrow_colors()
    winner = _winning_direction(Counter(votes))
    for idx, (direction, color) in enumerate(zip(votes, colors)):
        panel_r = idx // 3
        panel_c = idx % 3
        base_r = _panel_origin(panel_r, panel_h)
        base_c = _panel_origin(panel_c, panel_w)
        arrow_h, arrow_w = ARROW_SIZE[direction]
        top = base_r + random.randint(PANEL_MARGIN, panel_h - arrow_h - PANEL_MARGIN)
        left = base_c + random.randint(PANEL_MARGIN, panel_w - arrow_w - PANEL_MARGIN)
        _draw_arrow(grid, top, left, direction, color)

    return grid, winner


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        votes, winner = _sample_vote_list()
        grid, expected_winner = _build_input(votes)
        assert winner == expected_winner
        expected = _shift_grid(grid, winner)
        try:
            actual = main(grid)
        except AssertionError:
            continue
        if np.array_equal(actual, expected):
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
