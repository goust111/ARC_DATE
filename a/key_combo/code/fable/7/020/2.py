"""ARC task: use a concentric-ring color cycle as a gate for endpoint rows."""

from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
DIVIDER = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)


def _divider_row(grid: np.ndarray) -> int:
    """Return the unique all-gray row separating the key and data fields."""
    rows = np.flatnonzero(np.all(grid == DIVIDER, axis=1))
    assert rows.size == 1, "expected one full gray divider row"
    return int(rows[0])


def _ring_order(grid: np.ndarray, divider_row: int) -> list[int]:
    """Read layer colors from the square key, from its outside to its center."""
    upper = grid[:divider_row]
    coords = np.argwhere((upper != BACKGROUND) & (upper != DIVIDER))
    assert coords.size, "missing concentric-ring key"
    r0, c0 = map(int, coords.min(axis=0))
    r1, c1 = map(int, coords.max(axis=0))
    height, width = r1 - r0 + 1, c1 - c0 + 1
    assert height == width and height % 2 == 1, "key must have an odd square bbox"

    key = upper[r0 : r1 + 1, c0 : c1 + 1]
    layer_count = (height + 1) // 2
    assert layer_count >= 4, "the key must contain at least four layers"
    order: list[int] = []
    for depth in range(layer_count):
        layer_mask = np.fromfunction(
            lambda r, c: np.minimum.reduce((r, c, height - 1 - r, width - 1 - c)) == depth,
            (height, width),
            dtype=int,
        )
        colors = np.unique(key[layer_mask])
        assert colors.size == 1 and int(colors[0]) not in (BACKGROUND, DIVIDER), (
            "each square layer must be complete and monochromatic"
        )
        order.append(int(colors[0]))

    assert len(set(order)) == len(order), "ring layer colors must define an unambiguous cycle"
    assert np.count_nonzero((upper != BACKGROUND) & (upper != DIVIDER)) == key.size, (
        "the upper field must contain only one solid concentric key"
    )
    return order


def main(grid: np.ndarray) -> np.ndarray:
    """Use the concentric-ring successor cycle to gate and recolor endpoint-marked rows.

    Input structure:
    - A unique full gray divider row separates an upper key field from a lower data field.
    - The upper field contains one odd square of single-pixel-thick concentric, monochromatic
      layers; their distinct colors are read from the outer layer toward the center.
    - Every active lower row has colored markers only in its leftmost and rightmost cells,
      on an otherwise black background.

    Rule:
    - Treat the outer-to-inner layer colors as a cyclic order, with the center followed by
      the outer layer. If a row's right marker is the immediate successor of its left marker,
      fill that entire row with the successor of the right marker. Otherwise erase both
      markers to black. Preserve the key, divider, and blank lower rows.

    Key insight:
    - The square is a color-order key rather than an object to transform: authorization uses
      one successor step, while the paint color requires a second successor step and therefore
      does not occur among that input row's two markers.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"
    divider_row = _divider_row(g)
    order = _ring_order(g, divider_row)
    successor = {color: order[(i + 1) % len(order)] for i, color in enumerate(order)}

    out = g.copy()
    width = g.shape[1]
    for r in range(divider_row + 1, g.shape[0]):
        occupied = np.flatnonzero(g[r] != BACKGROUND)
        if occupied.size == 0:
            continue
        assert np.array_equal(occupied, np.array([0, width - 1])), (
            "an active data row must contain only two endpoint markers"
        )
        left, right = int(g[r, 0]), int(g[r, -1])
        assert left in successor and right in successor, "markers must use key colors"
        if successor[left] == right:
            out[r, :] = successor[right]
        else:
            out[r, :] = BACKGROUND
    return out


def _draw_key(grid: np.ndarray, top: int, left: int, order: list[int]) -> None:
    side = 2 * len(order) - 1
    for depth, color in enumerate(order):
        lo_r, hi_r = top + depth, top + side - 1 - depth
        lo_c, hi_c = left + depth, left + side - 1 - depth
        grid[lo_r, lo_c : hi_c + 1] = color
        grid[hi_r, lo_c : hi_c + 1] = color
        grid[lo_r : hi_r + 1, lo_c] = color
        grid[lo_r : hi_r + 1, hi_c] = color


def _semantic_check(grid: np.ndarray, order: list[int], row_specs: list[tuple[int, int]]) -> None:
    """Independently check structure, both branches, wraparound, and two-step dependence."""
    divider_row = _divider_row(grid)
    assert _ring_order(grid, divider_row) == order
    assert len(row_specs) == len(order) + 2
    succ = {color: order[(i + 1) % len(order)] for i, color in enumerate(order)}
    valid = [(a, b) for a, b in row_specs if succ[a] == b]
    invalid = [(a, b) for a, b in row_specs if succ[a] != b]
    assert len(valid) == len(order) and len(invalid) == 2
    assert (order[-1], order[0]) in valid, "wraparound successor must be visible"
    assert any(a == b for a, b in invalid), "same-color rejection must be visible"

    out = main(grid)
    data_rows = list(range(divider_row + 1, grid.shape[0] - 1))
    for r, (left, right) in zip(data_rows, row_specs):
        if succ[left] == right:
            expected = succ[right]
            assert expected not in (left, right)
            assert np.all(out[r] == expected)
        else:
            assert np.all(out[r] == BACKGROUND)
    assert np.array_equal(out[: divider_row + 1], grid[: divider_row + 1])


def generate_input() -> np.ndarray:
    """Generate a varied, non-degenerate ring key with exhaustive cycle-row coverage."""
    rng = random.SystemRandom()
    layer_count = rng.randint(4, 7)
    order = rng.sample(PALETTE, layer_count)
    side = 2 * layer_count - 1

    width = rng.randint(max(side + 2, 15), min(27, side + 10))
    upper_height = side + rng.randint(1, 4)
    row_count = layer_count + 2
    height = upper_height + 1 + row_count + 1
    assert height <= 30 and width <= 30

    grid = np.zeros((height, width), dtype=int)
    top = rng.randint(0, upper_height - side)
    left = rng.randint(1, width - side - 1)
    _draw_key(grid, top, left, order)
    divider_row = upper_height
    grid[divider_row, :] = DIVIDER

    # Every directed successor relation appears once, including the center-to-outer wrap.
    specs = [(order[i], order[(i + 1) % layer_count]) for i in range(layer_count)]
    same_index = rng.randrange(layer_count)
    specs.append((order[same_index], order[same_index]))
    # A two-step jump is a particularly tempting but invalid shortcut (n >= 4).
    jump_index = rng.randrange(layer_count)
    specs.append((order[jump_index], order[(jump_index + 2) % layer_count]))
    rng.shuffle(specs)

    for offset, (left_color, right_color) in enumerate(specs, start=1):
        r = divider_row + offset
        grid[r, 0] = left_color
        grid[r, -1] = right_color

    _semantic_check(grid, order, specs)
    return grid
