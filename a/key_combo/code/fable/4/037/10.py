"""ARC task 10: complete one missing cell in a coordinate-rolled 3 x 3 array."""

from __future__ import annotations

import random

import numpy as np


_PALETTE = tuple(range(1, 10))
_MAX_GENERATION_ATTEMPTS = 300
_generation_count = 0
_seen_outputs: set[tuple[tuple[int, int], bytes]] = set()


def _layout(grid: np.ndarray) -> tuple[int, int, list[slice]]:
    """Return (cell size, divider color, cell slices) for the 3 x 3 lattice."""
    h, w = map(int, grid.shape)
    assert h == w, "the 3 x 3 array must be square"
    assert h >= 5 and (h - 2) % 3 == 0, "expected three equal cells and two divider lines"

    size = (h - 2) // 3
    cuts = (size, 2 * size + 1)
    line_colors = [
        int(grid[cuts[0], 0]),
        int(grid[cuts[1], 0]),
        int(grid[0, cuts[0]]),
        int(grid[0, cuts[1]]),
    ]
    assert len(set(line_colors)) == 1, "all divider lines must use one color"
    divider = line_colors[0]
    assert all(np.all(grid[r, :] == divider) for r in cuts), "invalid horizontal dividers"
    assert all(np.all(grid[:, c] == divider) for c in cuts), "invalid vertical dividers"

    starts = (0, size + 1, 2 * size + 2)
    bands = [slice(start, start + size) for start in starts]
    return size, divider, bands


def _rolled(template: np.ndarray, row: int, col: int) -> np.ndarray:
    """Roll in the puzzle's (row, column) reference frame."""
    return np.roll(template, shift=(int(row), int(col)), axis=(0, 1))


def main(grid: np.ndarray) -> np.ndarray:
    """Complete the black cell of a 3 x 3 array with its coordinate-dependent cyclic roll.

    Input structure:
    - One-color full divider lines split a square grid into a 3 x 3 array of equal square cells.
    - The top-left cell is a template, and every cell at array coordinate (i, j) is that
      template cyclically rolled down i rows and right j columns.
    - Exactly one cell interior is entirely black, hiding the roll that belongs there.

    Rule:
    - Recover the template by rolling every visible cell up by its array-row coordinate and
      left by its array-column coordinate; all recovered copies must agree.
    - Roll that template down and right by the missing cell's own (i, j), and replace only
      the black cell interior. If (0, 0) is missing, the same reverse recovery supplies it.

    Key insight:
    - The nine cells are not plain copies: each cell's position is itself the two-component
      cyclic-roll parameter, so both array coordinates are necessary for reconstruction.
    """
    g = np.asarray(grid)
    assert g.ndim == 2, "expected a 2D grid"
    _size, _divider, bands = _layout(g)

    missing: list[tuple[int, int]] = []
    recovered: list[np.ndarray] = []
    for i, rows in enumerate(bands):
        for j, cols in enumerate(bands):
            cell = g[rows, cols]
            if np.all(cell == 0):
                missing.append((i, j))
            else:
                recovered.append(_rolled(cell, -i, -j))

    assert len(missing) == 1, "expected exactly one entirely black cell"
    assert recovered, "at least one rolled template copy must be visible"
    template = recovered[0]
    assert np.any(template != 0), "the template must contain a visible pattern"
    assert all(np.array_equal(candidate, template) for candidate in recovered[1:]), (
        "visible cells do not recover one common template"
    )

    mi, mj = missing[0]
    out = g.copy()
    out[bands[mi], bands[mj]] = _rolled(template, mi, mj)
    return out


def _random_template(size: int) -> np.ndarray | None:
    """Make a sparse asymmetric, toroidally connected multicolor template."""
    color_count = random.choice((2, 2, 3))
    pattern_colors = random.sample(_PALETTE, color_count)

    target_area = random.randint(max(color_count + 2, size), max(color_count + 2, size * size // 2))
    seed = (random.randrange(size), random.randrange(size))
    occupied = {seed}
    while len(occupied) < target_area:
        frontier = {
            ((r + dr) % size, (c + dc) % size)
            for r, c in occupied
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
        } - occupied
        if not frontier:
            return None
        occupied.add(random.choice(tuple(frontier)))

    template = np.zeros((size, size), dtype=int)
    cells = list(occupied)
    random.shuffle(cells)
    for index, (r, c) in enumerate(cells):
        # Seed every selected color visibly, then vary the remaining pattern pixels.
        color = pattern_colors[index] if index < color_count else random.choice(pattern_colors)
        template[r, c] = color

    variants = [_rolled(template, i, j) for i in range(3) for j in range(3)]
    if len({variant.tobytes() for variant in variants}) != 9:
        return None
    return template


def _render(template: np.ndarray, divider: int, missing: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Construct the independently known complete output and its one-cell-erased input."""
    size = int(template.shape[0])
    extent = 3 * size + 2
    complete = np.zeros((extent, extent), dtype=int)
    starts = (0, size + 1, 2 * size + 2)
    for i, r0 in enumerate(starts):
        for j, c0 in enumerate(starts):
            complete[r0 : r0 + size, c0 : c0 + size] = _rolled(template, i, j)

    cuts = (size, 2 * size + 1)
    complete[list(cuts), :] = divider
    complete[:, list(cuts)] = divider

    inp = complete.copy()
    mi, mj = missing
    r0, c0 = starts[mi], starts[mj]
    inp[r0 : r0 + size, c0 : c0 + size] = 0
    return inp, complete


def _semantically_valid(
    inp: np.ndarray,
    expected: np.ndarray,
    template: np.ndarray,
    missing: tuple[int, int],
) -> bool:
    """Check the stated semantics independently, including anti-shortcut cases."""
    if inp.shape != expected.shape or max(inp.shape) > 30:
        return False
    size, _divider, bands = _layout(inp)
    if size != template.shape[0]:
        return False

    black_cells = [
        (i, j)
        for i, rows in enumerate(bands)
        for j, cols in enumerate(bands)
        if np.all(inp[rows, cols] == 0)
    ]
    if black_cells != [missing]:
        return False

    variants = [_rolled(template, i, j) for i in range(3) for j in range(3)]
    # This excludes copying another cell, ignoring either coordinate, and periodic templates.
    if len({variant.tobytes() for variant in variants}) != 9:
        return False
    for i, rows in enumerate(bands):
        for j, cols in enumerate(bands):
            if (i, j) == missing:
                continue
            cell = inp[rows, cols]
            if not np.array_equal(cell, variants[3 * i + j]):
                return False
            if not np.array_equal(_rolled(cell, -i, -j), template):
                return False

    changed = inp != expected
    allowed = np.zeros_like(changed, dtype=bool)
    mi, mj = missing
    allowed[bands[mi], bands[mj]] = True
    if not changed.any() or np.any(changed & ~allowed):
        return False
    if not np.array_equal(expected[bands[mi], bands[mj]], variants[3 * mi + mj]):
        return False

    try:
        solved = main(inp)
    except (AssertionError, ValueError):
        return False
    return bool(np.array_equal(solved, expected))


def generate_input() -> np.ndarray:
    """Generate a varied valid lattice, guaranteeing the top-left branch once per three calls."""
    global _generation_count

    for _ in range(_MAX_GENERATION_ATTEMPTS):
        size = random.randint(4, 9)  # Complete grid ranges from 14 x 14 through 29 x 29.
        divider = random.choice(_PALETTE)
        template = _random_template(size)
        if template is None or np.any(template == divider):
            continue

        if _generation_count % 3 == 0:
            missing = (0, 0)
        else:
            missing = random.choice([(i, j) for i in range(3) for j in range(3) if (i, j) != (0, 0)])

        inp, expected = _render(template, divider, missing)
        signature = (tuple(map(int, expected.shape)), expected.tobytes())
        if signature in _seen_outputs:
            continue
        if not _semantically_valid(inp, expected, template, missing):
            continue

        _seen_outputs.add(signature)
        _generation_count += 1
        return inp

    raise RuntimeError("failed to generate a valid coordinate-roll completion input")
