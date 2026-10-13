from __future__ import annotations

import random

import numpy as np


_ARC_COLORS = tuple(range(10))
_HEIGHTS = tuple(range(7, 11))
_PANEL_WIDTHS = tuple(range(7, 10))


def _split_panels(grid: np.ndarray) -> tuple[list[np.ndarray], int, list[int]]:
    """Validate the divider layout and return panels, background, and panel colors."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a two-dimensional ARC grid"
    height, total_width = map(int, g.shape)
    assert height > 0 and total_width >= 5, "grid is too small for three panels"
    assert (total_width - 2) % 3 == 0, "two columns must separate three equal-width panels"

    panel_width = (total_width - 2) // 3
    first_separator = panel_width
    second_separator = 2 * panel_width + 1
    assert np.all(g[:, first_separator] == g[0, first_separator]), "first separator must be solid"
    assert np.all(g[:, second_separator] == g[0, second_separator]), "second separator must be solid"

    panels = [
        g[:, :panel_width],
        g[:, panel_width + 1 : 2 * panel_width + 1],
        g[:, 2 * panel_width + 2 :],
    ]
    assert all(panel.shape == (height, panel_width) for panel in panels)

    color_sets = [set(map(int, np.unique(panel))) for panel in panels]
    assert all(len(colors) == 2 for colors in color_sets), (
        "each panel must contain one background and one foreground color"
    )
    common = set.intersection(*color_sets)
    assert len(common) == 1, "the shared panel color must uniquely identify the background"
    background = common.pop()

    foregrounds: list[int] = []
    for colors in color_sets:
        foreground = colors - {background}
        assert len(foreground) == 1
        foregrounds.append(foreground.pop())
    assert len(set(foregrounds)) == 3, "the three foreground colors must be different"
    return panels, int(background), foregrounds


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combine three aligned masks by coloring cells from their foreground coverage count.

    Input structure:
    - Two solid separator columns divide the grid into three equal-size aligned panels.
    - All panels share one background color and each has exactly one distinct foreground color.

    Rule:
    - Remove the separators and compare corresponding panel cells. A cell covered by exactly
      one foreground keeps that foreground color; a cell covered by exactly two foregrounds
      takes the third, absent panel's foreground color; zero or three covers become background.

    Key insight:
    - A double overlap is colored by the panel that is missing there, so coverage count and
      panel membership matter rather than layer order or a conventional transparent overlay.
    """
    panels, background, foregrounds = _split_panels(grid)
    masks = np.stack([panel == color for panel, color in zip(panels, foregrounds)])
    coverage = masks.sum(axis=0)

    out = np.full(panels[0].shape, background, dtype=int)
    for panel_index, color in enumerate(foregrounds):
        use_color = ((coverage == 1) & masks[panel_index]) | (
            (coverage == 2) & ~masks[panel_index]
        )
        out[use_color] = color
    return out


def _landmark_positions(height: int, width: int) -> list[tuple[int, int]]:
    """Choose seven separated interior locations for coverage states 001 through 111."""
    lattice = [(r, c) for r in range(1, height - 1, 2) for c in range(1, width - 1, 2)]
    assert len(lattice) >= 7
    random.shuffle(lattice)
    return lattice[:7]


def _grow_state_patch(states: np.ndarray, code: int, seed: tuple[int, int], extra: int) -> None:
    """Grow a small orthogonal patch without overwriting another semantic state."""
    height, width = states.shape
    cells = {seed}
    for _ in range(extra):
        frontier: set[tuple[int, int]] = set()
        for r, c in cells:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = r + dr, c + dc
                if 0 < rr < height - 1 and 0 < cc < width - 1 and states[rr, cc] == 0:
                    frontier.add((rr, cc))
        if not frontier:
            return
        cell = random.choice(tuple(frontier))
        states[cell] = code
        cells.add(cell)


def _sample_coverage_states(height: int, width: int) -> np.ndarray:
    """Build a sparse map containing every possible three-panel coverage state."""
    states = np.zeros((height, width), dtype=np.uint8)
    positions = _landmark_positions(height, width)
    random.shuffle(positions)
    seeds: dict[int, tuple[int, int]] = {}
    for code, position in zip(range(1, 8), positions):
        states[position] = code
        seeds[code] = position

    # Make single-cover regions visually prominent while keeping overlap cells sparse.
    for code in (1, 2, 4):
        _grow_state_patch(states, code, seeds[code], random.randint(2, 5))
    for code in (3, 5, 6):
        _grow_state_patch(states, code, seeds[code], random.randint(0, 1))
    return states


def _render_input(
    states: np.ndarray, background: int, foregrounds: list[int], separator: int
) -> np.ndarray:
    height, panel_width = states.shape
    total_width = 3 * panel_width + 2
    grid = np.full((height, total_width), background, dtype=int)
    grid[:, panel_width] = separator
    grid[:, 2 * panel_width + 1] = separator
    starts = (0, panel_width + 1, 2 * panel_width + 2)
    for panel_index, (start, color) in enumerate(zip(starts, foregrounds)):
        panel = grid[:, start : start + panel_width]
        panel[(states & (1 << panel_index)) != 0] = color
    return grid


def _expected_from_states(states: np.ndarray, background: int, foregrounds: list[int]) -> np.ndarray:
    """Independent truth table: 011, 101, and 110 select their absent color."""
    lookup = np.array(
        [
            background,
            foregrounds[0],
            foregrounds[1],
            foregrounds[2],
            foregrounds[2],
            foregrounds[1],
            foregrounds[0],
            background,
        ],
        dtype=int,
    )
    return lookup[states]


def _semantics_are_explicit(
    grid: np.ndarray,
    states: np.ndarray,
    background: int,
    foregrounds: list[int],
) -> bool:
    """Reject cases that hide a branch or admit ordinary overlay/priority shortcuts."""
    counts = np.bincount(states.ravel(), minlength=8)
    if np.any(counts[:8] == 0):
        return False
    if int(counts[3] + counts[5] + counts[6] + counts[7]) * 4 > states.size:
        return False

    expected = _expected_from_states(states, background, foregrounds)
    try:
        actual = main(grid)
    except (AssertionError, ValueError):
        return False
    if not np.array_equal(actual, expected):
        return False

    # Each double-cover state must visibly choose the color of its absent panel,
    # which differs from both colors any fixed layer priority could select.
    for state, absent_index, present_indices in (
        (3, 2, (0, 1)),
        (5, 1, (0, 2)),
        (6, 0, (1, 2)),
    ):
        values = set(map(int, actual[states == state]))
        if values != {foregrounds[absent_index]}:
            return False
        if foregrounds[absent_index] in {foregrounds[i] for i in present_indices}:
            return False

    return bool(
        np.all(actual[states == 0] == background)
        and np.all(actual[states == 7] == background)
    )


def generate_input() -> np.ndarray:
    """Generate a varied, non-degenerate three-panel coverage-count puzzle input."""
    for _ in range(100):
        height = random.choice(_HEIGHTS)
        panel_width = random.choice(_PANEL_WIDTHS)
        background, *others = random.sample(_ARC_COLORS, 5)
        foregrounds = list(others[:3])
        separator = others[3]

        states = _sample_coverage_states(height, panel_width)
        grid = _render_input(states, background, foregrounds, separator)
        if max(grid.shape) <= 30 and _semantics_are_explicit(
            grid, states, background, foregrounds
        ):
            return grid
    raise RuntimeError("failed to generate a valid coverage-count overlay input")
