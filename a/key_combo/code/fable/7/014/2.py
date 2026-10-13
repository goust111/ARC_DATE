from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)


def _split_panels(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the panels separated by the unique all-grey row."""
    divider_rows = np.flatnonzero(np.all(grid == GREY, axis=1))
    assert divider_rows.size == 1, "expected exactly one full grey divider row"
    divider = int(divider_rows[0])
    assert 0 < divider < grid.shape[0] - 1, "both panels must be nonempty"
    return grid[:divider], grid[divider + 1 :]


def _thumbnail(upper: np.ndarray, panel_shape: tuple[int, int]) -> np.ndarray:
    """Compute the true foreground color of every coarse block."""
    ph, pw = panel_shape
    uh, uw = upper.shape
    assert uh % ph == 0 and uw % pw == 0, "panel resolutions must align by integer factors"
    bh, bw = uh // ph, uw // pw
    result = np.zeros((ph, pw), dtype=int)
    for r in range(ph):
        for c in range(pw):
            block = upper[r * bh : (r + 1) * bh, c * bw : (c + 1) * bw]
            colors = set(map(int, np.unique(block))) - {BLACK}
            assert len(colors) <= 1, "each coarse block may use at most one foreground color"
            if colors:
                result[r, c] = colors.pop()
    return result


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use discrepancies in a thumbnail panel to recolor or erase aligned coarse blocks.

    Input structure:
    - A unique full grey row separates an upper black-background pixel picture from a lower thumbnail panel.
    - The upper panel dimensions are integer multiples of the thumbnail dimensions, and each aligned coarse block has at most one non-black foreground color.

    Rule:
    - Compute the upper panel's true thumbnail and compare it cell by cell with the lower panel.
    - Leave matching blocks unchanged; at a mismatch, recolor all existing foreground pixels to the thumbnail cell's non-black color, or clear the whole block when that cell is black.
    - Return only the modified upper panel, preserving every retained block's fine-resolution shape.

    Key insight:
    - The lower panel is mostly an accurate compression; only its differences from the independently computed true thumbnail are full-resolution editing commands.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"
    upper, commands = _split_panels(g)
    truth = _thumbnail(upper, commands.shape)
    bh = upper.shape[0] // commands.shape[0]
    bw = upper.shape[1] // commands.shape[1]

    out = upper.copy()
    for r, c in np.argwhere(commands != truth):
        r, c = int(r), int(c)
        block = out[r * bh : (r + 1) * bh, c * bw : (c + 1) * bw]
        command = int(commands[r, c])
        if command == BLACK:
            block[:] = BLACK
        else:
            block[block != BLACK] = command
    return out


def _expected_from_roles(
    upper: np.ndarray,
    commands: np.ndarray,
    truth: np.ndarray,
) -> np.ndarray:
    """Independent generator oracle for the deliberately assigned roles."""
    bh = upper.shape[0] // commands.shape[0]
    bw = upper.shape[1] // commands.shape[1]
    expected = upper.copy()
    for r, c in np.argwhere(commands != truth):
        r, c = int(r), int(c)
        sl = np.s_[r * bh : (r + 1) * bh, c * bw : (c + 1) * bw]
        if commands[r, c] == BLACK:
            expected[sl] = BLACK
        else:
            mask = expected[sl] != BLACK
            expected[sl][mask] = int(commands[r, c])
    return expected


def _generate_once() -> np.ndarray:
    # An unpadded vertical layout makes both panel widths intrinsic.  The
    # horizontal integer scale is one; the vertical scale varies from 2 to 4.
    ph = random.randint(3, 5)
    pw = random.randint(5, 9)
    bh = random.randint(2, 4)
    upper = np.zeros((ph * bh, pw), dtype=int)
    truth = np.zeros((ph, pw), dtype=int)

    cells = [(r, c) for r in range(ph) for c in range(pw)]
    random.shuffle(cells)
    occupied_count = random.randint(max(8, len(cells) // 2), len(cells) - 3)
    occupied = cells[:occupied_count]
    empty = cells[occupied_count:]

    # Every occupied block has a non-solid fine shape, so a naive expansion of
    # the thumbnail destroys visible information.
    for r, c in occupied:
        color = random.choice(PALETTE)
        count = random.randint(1, bh - 1)
        rows = random.sample(range(bh), count)
        upper[r * bh + np.asarray(rows), c] = color
        truth[r, c] = color

    commands = truth.copy()
    # Force all meaningful roles in every generated puzzle: two recolors, two
    # erasures, faithful occupied cells, and faithful empty cells.
    recolor_cells = occupied[:2]
    erase_cells = occupied[2:4]
    for r, c in recolor_cells:
        commands[r, c] = random.choice([x for x in PALETTE if x != truth[r, c]])
    for r, c in erase_cells:
        commands[r, c] = BLACK

    # Add occasional extra commands without weakening the guaranteed branches.
    stable_occupied = occupied[4]
    for r, c in occupied[5:]:
        roll = random.random()
        if roll < 0.12:
            commands[r, c] = BLACK
        elif roll < 0.28:
            commands[r, c] = random.choice([x for x in PALETTE if x != truth[r, c]])

    divider = np.full((1, pw), GREY, dtype=int)
    grid = np.vstack((upper, divider, commands))
    expected = _expected_from_roles(upper, commands, truth)

    # Semantic checks: each branch exists, each command changes output, fine
    # shapes survive recoloring, and the computed thumbnail is the intended one.
    assert np.array_equal(_thumbnail(upper, commands.shape), truth)
    mismatch = commands != truth
    assert np.count_nonzero(mismatch) >= 4
    assert all(truth[r, c] != BLACK for r, c in np.argwhere(mismatch))
    assert empty and any(commands[r, c] == BLACK for r, c in empty)
    assert commands[stable_occupied] == truth[stable_occupied] != BLACK
    actual = main(grid)
    assert np.array_equal(actual, expected)
    assert not np.array_equal(actual, upper)
    for r, c in recolor_cells:
        before = upper[r * bh : (r + 1) * bh, c]
        after = actual[r * bh : (r + 1) * bh, c]
        assert np.array_equal(before != BLACK, after != BLACK)
        assert np.all(after[after != BLACK] == commands[r, c])
    for r, c in erase_cells:
        assert np.all(actual[r * bh : (r + 1) * bh, c] == BLACK)
    r, c = stable_occupied
    assert np.array_equal(
        actual[r * bh : (r + 1) * bh, c],
        upper[r * bh : (r + 1) * bh, c],
    )
    assert grid.shape[0] <= 30 and grid.shape[1] <= 30
    return grid


def generate_input() -> np.ndarray:
    """Generate a varied, non-degenerate valid instance."""
    for _ in range(100):
        try:
            return _generate_once()
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid distorted-thumbnail puzzle")
