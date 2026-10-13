from __future__ import annotations

from collections import Counter

import numpy as np
import random


def _control_marker(top: np.ndarray, left: np.ndarray) -> tuple[int, int]:
    top_counts = Counter(map(int, top))
    left_counts = Counter(map(int, left))
    markers = [
        color
        for color, count in top_counts.items()
        if count == 1 and left_counts.get(color) == 1
    ]
    assert len(markers) == 1, "expected one shared single-pixel control marker"
    marker = int(markers[0])

    filler_colors = (set(map(int, top)) | set(map(int, left))) - {marker}
    assert len(filler_colors) == 1, "expected one shared control filler color"
    filler = int(next(iter(filler_colors)))
    assert np.all(top[top != marker] == filler), "top control band must be marker plus filler"
    assert np.all(left[left != marker] == filler), "left control band must be marker plus filler"
    return marker, filler


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop off the control row and column, then cyclically roll the payload by the two marker offsets.

    Input structure
    - The grid is square, with a control row on top and a control column on the left.
    - The control row and column share one filler color and one single-pixel marker color.
    - The lower-right square is the payload and contains none of the control colors.

    Rule
    - Remove the top control row and left control column.
    - Roll the remaining payload down by the marker's zero-based position in the left control column.
    - Roll the payload right by the marker's zero-based position in the top control row.

    Key insight
    - The two border markers are cyclic shift amounts for the payload, not colors to preserve.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be a 2D grid"
    h, w = g.shape
    assert h == w and h >= 3, "input must be a square grid with controls"

    top = g[0, 1:]
    left = g[1:, 0]
    payload = g[1:, 1:]
    marker, filler = _control_marker(top, left)

    corner = int(g[0, 0])
    assert corner not in {marker, filler}, "control corner must be a third color"
    payload_colors = set(map(int, payload.ravel()))
    assert not ({corner, marker, filler} & payload_colors), "payload must exclude control colors"

    row_hits = np.flatnonzero(left == marker)
    col_hits = np.flatnonzero(top == marker)
    assert row_hits.size == 1 and col_hits.size == 1, "expected one marker in each control band"
    row_shift = int(row_hits[0])
    col_shift = int(col_hits[0])
    return np.roll(payload, shift=(row_shift, col_shift), axis=(0, 1))

# --------------- generator (att2) ---------------

from itertools import product

# Palette and canvas dimensions.
PAYLOAD_SIZE = 6
GRID_SIZE = PAYLOAD_SIZE + 1
COLORS = tuple(range(1, 10))

# Payload growth constraints.
PAYLOAD_COLOR_COUNT = 6
PAYLOAD_CELLS = PAYLOAD_SIZE * PAYLOAD_SIZE
MIN_COLOR_CELLS = 4
MAX_COLOR_CELLS = 9
MAX_COMPONENTS_PER_COLOR = 2
EMPTY_LABEL = -1

# Sampling choices and retry policy.
ORTHOGONAL_STEPS = ((1, 0), (-1, 0), (0, 1), (0, -1))
VALID_SHIFTS = tuple(
    (r, c) for r in range(PAYLOAD_SIZE) for c in range(PAYLOAD_SIZE) if (r, c) != (0, 0)
)
COUNT_TARGET_CHOICES = [
    counts for counts in product(range(MIN_COLOR_CELLS, MAX_COLOR_CELLS + 1), repeat=PAYLOAD_COLOR_COUNT)
    if sum(counts) == PAYLOAD_CELLS
]
MAX_GENERATION_ATTEMPTS = 300


def _frontier(labels: np.ndarray, color_i: int) -> list[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for r, c in np.argwhere(labels == color_i):
        rr = int(r)
        cc = int(c)
        for dr, dc in ORTHOGONAL_STEPS:
            nr = rr + dr
            nc = cc + dc
            in_bounds = 0 <= nr < PAYLOAD_SIZE and 0 <= nc < PAYLOAD_SIZE
            if in_bounds and labels[nr, nc] == EMPTY_LABEL:
                cells.add((nr, nc))
    return list(cells)


def _choose_growth_cell(labels, active, components) -> tuple[int, int, int] | None:
    """Pick the next label/cell, counting a new component when growth must split."""
    frontiers = {i: _frontier(labels, i) for i in active}
    expandable = [i for i in active if frontiers[i]]
    if expandable:
        min_frontier = min(len(frontiers[i]) for i in expandable)
        at_risk = [i for i in expandable if len(frontiers[i]) == min_frontier]
        color_i = random.choice(at_risk)
        r, c = random.choice(frontiers[color_i])
        return color_i, r, c

    splitable = [i for i in active if components[i] < MAX_COMPONENTS_PER_COLOR]
    if not splitable:
        return None
    color_i = random.choice(splitable)
    r, c = random.choice([tuple(map(int, rc)) for rc in np.argwhere(labels == EMPTY_LABEL)])
    components[color_i] += 1
    return color_i, r, c


def _make_payload(payload_colors: list[int]) -> np.ndarray | None:
    targets = random.choice(COUNT_TARGET_CHOICES)
    labels = np.full((PAYLOAD_SIZE, PAYLOAD_SIZE), EMPTY_LABEL, dtype=int)
    seeds = random.sample(range(PAYLOAD_CELLS), PAYLOAD_COLOR_COUNT)
    have = [1] * PAYLOAD_COLOR_COUNT
    components = [1] * PAYLOAD_COLOR_COUNT

    for color_i, pos in enumerate(seeds):
        labels[pos // PAYLOAD_SIZE, pos % PAYLOAD_SIZE] = color_i

    while np.any(labels == EMPTY_LABEL):
        active = [i for i in range(PAYLOAD_COLOR_COUNT) if have[i] < targets[i]]
        if not active:
            return None

        picked = _choose_growth_cell(labels, active, components)
        if picked is None:
            return None
        color_i, r, c = picked
        labels[r, c] = color_i
        have[color_i] += 1

    palette = np.array(payload_colors, dtype=int)
    return palette[labels]


def _sample_colors() -> tuple[int, int, int, list[int]]:
    colors = list(COLORS)
    random.shuffle(colors)
    return colors[0], colors[1], colors[2], colors[3:]


def _assemble_grid(corner, marker, filler, payload, shift) -> np.ndarray:
    row_shift, col_shift = shift
    grid = np.empty((GRID_SIZE, GRID_SIZE), dtype=int)
    grid[0, 0] = corner
    grid[0, 1:] = filler
    grid[1:, 0] = filler
    grid[0, 1 + col_shift] = marker
    grid[1 + row_shift, 0] = marker
    grid[1:, 1:] = payload
    return grid


def _gen_attempt() -> np.ndarray | None:
    corner, marker, filler, payload_colors = _sample_colors()
    payload = _make_payload(payload_colors)
    if payload is None:
        return None

    shift = random.choice(VALID_SHIFTS)
    grid = _assemble_grid(corner, marker, filler, payload, shift)
    expected = np.roll(payload, shift=shift, axis=(0, 1))
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 79cce52d input")
