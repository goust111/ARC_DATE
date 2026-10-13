"""ARC task: grey pixels select rotated whole-grid stamps."""

from __future__ import annotations

import numpy as np


BLACK = 0
GREY = 5
NON_CONTROL_COLORS = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)


def _quadrant_turns(r: int, c: int, n: int) -> int:
    """Return clockwise quarter-turns using the centre of the full n*n grid."""
    half = n // 2
    if r < half:
        return 0 if c < half else 1
    return 3 if c < half else 2


def main(grid: np.ndarray) -> np.ndarray:
    """Stamp a quadrant-oriented copy of the whole input at every grey-selected output block.

    Input structure:
    - The input is a non-empty even-sided square ARC grid; black is empty space and
      grey pixels are control pixels embedded in a multicolored source pattern.
    - Each control pixel lies in one of the four quadrants defined by the centre
      of the full input grid.

    Rule:
    - Build an n-squared by n-squared black output, divided into n by n blocks,
      each block having the same n by n shape as the input.
    - For every grey pixel (r,c), stamp the entire input into block (r,c), rotated
      clockwise by 0, 1, 2, or 3 quarter-turns for the upper-left, upper-right,
      lower-right, or lower-left input quadrant respectively. Leave every block
      selected by a non-grey pixel black.

    Key insight:
    - A grey pixel has two inseparable roles: its full-grid coordinate selects a
      destination block, while its quadrant relative to the full-grid centre
      determines that block's copy orientation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    height, width = map(int, g.shape)
    assert height == width and height % 2 == 0, "expected an even-sided square"
    assert height * height <= 30, "expanded output exceeds the ARC 30x30 limit"
    assert np.all((0 <= g) & (g <= 9)), "expected ARC color ids 0 through 9"

    controls = np.argwhere(g == GREY)
    assert controls.size > 0, "expected at least one grey control pixel"

    n = height
    out = np.full((n * n, n * n), BLACK, dtype=int)
    for r, c in controls:
        r, c = int(r), int(c)
        turns = _quadrant_turns(r, c, n)
        stamp = np.rot90(g, -turns)
        out[r * n : (r + 1) * n, c * n : (c + 1) * n] = stamp
    return out


def _manual_rotate_clockwise(grid: np.ndarray, turns: int) -> np.ndarray:
    """Coordinate-form rotation used only as an independent generator oracle."""
    n = int(grid.shape[0])
    out = np.empty_like(grid)
    for r in range(n):
        for c in range(n):
            if turns == 0:
                source_r, source_c = r, c
            elif turns == 1:
                source_r, source_c = n - 1 - c, r
            elif turns == 2:
                source_r, source_c = n - 1 - r, n - 1 - c
            else:
                source_r, source_c = c, n - 1 - r
            out[r, c] = grid[source_r, source_c]
    return out


def _manual_transform(grid: np.ndarray) -> np.ndarray:
    """Independent, explicit block-and-coordinate specification of the rule."""
    n = int(grid.shape[0])
    out = np.zeros((n * n, n * n), dtype=int)
    for block_r in range(n):
        for block_c in range(n):
            if int(grid[block_r, block_c]) != GREY:
                continue
            turns = _quadrant_turns(block_r, block_c, n)
            block = _manual_rotate_clockwise(grid, turns)
            for r in range(n):
                for c in range(n):
                    out[block_r * n + r, block_c * n + c] = block[r, c]
    return out


def _semantically_valid(grid: np.ndarray) -> bool:
    """Reject ambiguous, shortcut-friendly, or rule-degenerate generated cases."""
    n = int(grid.shape[0])
    controls = [tuple(map(int, p)) for p in np.argwhere(grid == GREY)]
    colored = np.argwhere((grid != BLACK) & (grid != GREY))
    if len(controls) < 2 or len(colored) < 2:
        return False

    # The whole visible source must expose every non-zero quarter-turn.
    if any(np.array_equal(grid, np.rot90(grid, -k)) for k in (1, 2, 3)):
        return False

    turns_present = {_quadrant_turns(r, c, n) for r, c in controls}
    if n == 4 and turns_present != {0, 1, 2, 3}:
        return False
    if n == 2 and len(turns_present) < 2:
        return False

    expected = _manual_transform(grid)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False

    # Every selected block must be the correct oriented source and every other
    # block must be entirely black, so neither the stamp nor placement step can
    # be replaced by grey-pixel upscaling or unconditional tiling.
    for block_r in range(n):
        for block_c in range(n):
            block = actual[
                block_r * n : (block_r + 1) * n,
                block_c * n : (block_c + 1) * n,
            ]
            if int(grid[block_r, block_c]) == GREY:
                turns = _quadrant_turns(block_r, block_c, n)
                if not np.array_equal(block, _manual_rotate_clockwise(grid, turns)):
                    return False
            elif np.any(block != BLACK):
                return False

    # Removing one control must remove its destination stamp, while changing a
    # source color must alter every surviving stamp. These are dependency checks,
    # not merely a main(generate_input()) round trip.
    removed = grid.copy()
    removed[controls[0]] = BLACK
    if np.array_equal(expected, _manual_transform(removed)):
        return False

    changed = grid.copy()
    source_r, source_c = map(int, colored[0])
    old_color = int(changed[source_r, source_c])
    new_color = next(color for color in NON_CONTROL_COLORS.tolist() if color != old_color)
    changed[source_r, source_c] = new_color
    if np.array_equal(expected, _manual_transform(changed)):
        return False
    return True


def _generate_size_two() -> np.ndarray:
    """Make a tiny but asymmetric two-control, two-color instance."""
    grid = np.zeros((2, 2), dtype=int)
    order = np.random.permutation(4)
    control_indices = order[:2]
    source_indices = order[2:]
    grid.flat[control_indices] = GREY
    colors = np.random.choice(NON_CONTROL_COLORS, size=2, replace=False)
    grid.flat[source_indices] = colors
    return grid


def _generate_size_four() -> np.ndarray:
    """Make a spacious instance with at least one control in every quadrant."""
    n = 4
    grid = np.zeros((n, n), dtype=int)

    controls: list[tuple[int, int]] = []
    for row_start, col_start in ((0, 0), (0, 2), (2, 2), (2, 0)):
        controls.append(
            (
                row_start + int(np.random.randint(0, 2)),
                col_start + int(np.random.randint(0, 2)),
            )
        )

    free = [(r, c) for r in range(n) for c in range(n) if (r, c) not in controls]
    np.random.shuffle(free)
    extra_control_count = int(np.random.randint(0, 2))
    controls.extend(free[:extra_control_count])
    for r, c in controls:
        grid[r, c] = GREY

    free = [(r, c) for r in range(n) for c in range(n) if int(grid[r, c]) == BLACK]
    np.random.shuffle(free)
    colored_count = int(np.random.randint(4, min(8, len(free) - 1) + 1))
    color_count = int(np.random.randint(2, min(4, colored_count) + 1))
    palette = np.random.choice(NON_CONTROL_COLORS, size=color_count, replace=False)
    values = np.empty(colored_count, dtype=int)
    values[:color_count] = palette
    if colored_count > color_count:
        values[color_count:] = np.random.choice(
            palette, size=colored_count - color_count, replace=True
        )
    np.random.shuffle(values)
    for (r, c), color in zip(free[:colored_count], values):
        grid[r, c] = int(color)
    return grid


def generate_input() -> np.ndarray:
    """Generate varied valid inputs while explicitly filtering semantic shortcuts."""
    for _ in range(200):
        # Four-by-four cases dominate because they display all four orientation
        # branches together; two-by-two cases provide the only other legal even
        # input size under the n^2 <= 30 ARC output constraint.
        grid = _generate_size_four() if np.random.random() < 0.8 else _generate_size_two()
        if _semantically_valid(grid):
            return grid
    raise RuntimeError("failed to generate a valid grey-quadrant stamp input")
