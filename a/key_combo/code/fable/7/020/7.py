"""ARC task 7: extend a bracket-delimited motif with two coupled sequences."""

from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
MARKER = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_ATTEMPTS = 300


def _rotate_block(block: np.ndarray, colors: list[int], shift: int) -> np.ndarray:
    """Rotate motif colors by ``shift`` positions in visible first-occurrence order."""
    rotated = block.copy()
    for i, color in enumerate(colors):
        rotated[block == color] = colors[(i + shift) % len(colors)]
    return rotated


def _copy_start(height: int, copy_index: int) -> int:
    """Top row of copy i; gaps before copies 1, 2, ... have sizes 1, 2, ... ."""
    return copy_index * height + copy_index * (copy_index + 1) // 2


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend a bracket-delimited row motif downward with growing gaps and cyclic colors.

    Input structure:
    - The background is black.  A continuous gray run from the top of the leftmost
      column marks the true height of the first motif block, including an internal
      completely blank row.
    - Three copies start at the top, separated by one and then two blank rows; their
      colors are successive rotations.  Everything below the third copy is blank.

    Rule:
    - Erase the gray marker and read the motif colors in row-major first-occurrence
      order.  Continue copying the entire marked-height block downward.
    - Use gaps of 3, 4, ... blank rows and advance every motif color one position in
      that visible cyclic order per copy.  Preserve existing pixels and clip the last
      copy at the bottom edge.

    Key insight:
    - The gray run, rather than blank-row segmentation, fixes the motif height; both
      the increasing spatial gaps and the cyclic color phase must be extrapolated.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    rows, cols = map(int, g.shape)
    assert rows <= 30 and cols <= 30, "ARC grids must be at most 30x30"

    marker_rows = np.flatnonzero(g[:, 0] == MARKER)
    assert marker_rows.size > 0, "expected a gray height marker in the left column"
    motif_h = int(marker_rows.size)
    assert np.array_equal(marker_rows, np.arange(motif_h)), "marker must be a top-anchored run"
    assert np.count_nonzero(g == MARKER) == motif_h, "gray is reserved for the height marker"

    motif = g[:motif_h].copy()
    motif[:, 0] = BACKGROUND
    colors: list[int] = []
    for value in motif.ravel():
        color = int(value)
        if color not in (BACKGROUND, MARKER) and color not in colors:
            colors.append(color)
    assert len(colors) >= 2, "motif needs a visible color cycle"

    # Confirm that the visible prefix really supplies the two stated sequences.
    expected_prefix = np.full_like(g, BACKGROUND)
    expected_prefix[:motif_h, 0] = MARKER
    for copy_index in range(3):
        start = _copy_start(motif_h, copy_index)
        end = start + motif_h
        assert end <= rows, "the three complete clue copies must fit"
        expected_prefix[start:end] = _rotate_block(motif, colors, copy_index)
    expected_prefix[:motif_h, 0] = MARKER
    assert np.array_equal(g, expected_prefix), "expected exactly three clue copies and a blank suffix"

    out = g.copy()
    out[:motif_h, 0] = BACKGROUND
    copy_index = 3
    while True:
        start = _copy_start(motif_h, copy_index)
        if start >= rows:
            break
        block = _rotate_block(motif, colors, copy_index)
        visible_h = min(motif_h, rows - start)
        visible = block[:visible_h]
        target = out[start : start + visible_h]
        mask = visible != BACKGROUND
        target[mask] = visible[mask]
        copy_index += 1
    return out


def _make_motif(height: int, width: int, colors: list[int]) -> np.ndarray:
    """Make a sparse, readable motif with one genuinely internal blank row."""
    motif = np.zeros((height, width), dtype=int)
    blank_row = random.randint(1, height - 2)
    usable_cols = list(range(random.randint(2, 3), width - 1))
    nonblank_rows = [r for r in range(height) if r != blank_row]

    cells: set[tuple[int, int]] = set()
    center = random.choice(usable_cols)
    for r in nonblank_rows:
        center = min(max(center + random.choice((-1, 0, 1)), usable_cols[0]), usable_cols[-1])
        row_width = random.randint(2, min(5, len(usable_cols)))
        left = min(max(center - row_width // 2, usable_cols[0]), usable_cols[-1] - row_width + 1)
        cells.update((r, c) for c in range(left, left + row_width))

    ordered = sorted(cells)
    if len(ordered) < len(colors):
        raise ValueError("not enough motif cells for all colors")
    # These anchors make the color order a visible structural cue, never an id ordering.
    for cell, color in zip(ordered[: len(colors)], colors):
        motif[cell] = color
    for cell in ordered[len(colors) :]:
        motif[cell] = random.choice(colors)
    return motif


def _oracle(motif: np.ndarray, colors: list[int], rows: int, *, include_marker: bool) -> np.ndarray:
    """Independent constructor used by the generator's semantic checks."""
    height, width = motif.shape
    out = np.zeros((rows, width), dtype=int)
    index = 0
    while True:
        start = index * height + sum(range(1, index + 1))
        if start >= rows:
            break
        take = min(height, rows - start)
        # Deliberately avoid _rotate_block: this is a separate cell-level oracle.
        for r in range(take):
            for c in range(width):
                value = int(motif[r, c])
                if value != BACKGROUND:
                    color_index = colors.index(value)
                    out[start + r, c] = colors[(color_index + index) % len(colors)]
        index += 1
    if include_marker:
        out[:height, 0] = MARKER
    return out


def _wrong_constant_gap(motif: np.ndarray, colors: list[int], rows: int) -> np.ndarray:
    out = _oracle(motif, colors, rows, include_marker=False)
    # Replace only the continuation with copies separated forever by two rows.
    third_end = _copy_start(motif.shape[0], 2) + motif.shape[0]
    out[third_end:] = BACKGROUND
    index = 3
    start = third_end + 2
    while start < rows:
        block = _rotate_block(motif, colors, index)
        take = min(motif.shape[0], rows - start)
        out[start : start + take] = block[:take]
        start += motif.shape[0] + 2
        index += 1
    return out


def _valid_candidate(inp: np.ndarray, motif: np.ndarray, colors: list[int]) -> bool:
    rows = inp.shape[0]
    expected = _oracle(motif, colors, rows, include_marker=False)
    try:
        actual = main(inp)
    except (AssertionError, ValueError):
        return False
    if not np.array_equal(actual, expected):
        return False
    # Semantic anti-shortcuts: both continuation laws must visibly affect the result.
    no_rotation = expected.copy()
    start = _copy_start(motif.shape[0], 3)
    for index in range(3, 8):
        start = _copy_start(motif.shape[0], index)
        if start >= rows:
            break
        take = min(motif.shape[0], rows - start)
        plain = motif[:take]
        mask = plain != BACKGROUND
        region = no_rotation[start : start + take]
        region[mask] = plain[mask]
    if np.array_equal(expected, no_rotation):
        return False
    if np.array_equal(expected, _wrong_constant_gap(motif, colors, rows)):
        return False
    internal_blank = np.all(motif == BACKGROUND, axis=1)
    return bool(internal_blank[1:-1].any() and np.any(actual != inp))


def generate_input() -> np.ndarray:
    """Generate varied, non-degenerate examples satisfying every solver assumption."""
    for _ in range(MAX_ATTEMPTS):
        motif_h = random.randint(4, 6)
        width = random.randint(9, 17)
        color_count = random.randint(3, min(5, motif_h + 1))
        colors = random.sample(PALETTE, color_count)
        try:
            motif = _make_motif(motif_h, width, colors)
        except ValueError:
            continue

        minimum_rows = 3 * motif_h + 7  # at least one visible row of copy four
        if random.random() < 0.5:
            # Bias toward bottom clipping while still varying the clipped amount.
            maximum_partial = min(30, 4 * motif_h + 5)
            rows = random.randint(minimum_rows, maximum_partial)
        else:
            rows = random.randint(minimum_rows, 30)

        full = _oracle(motif, colors, rows, include_marker=True)
        third_end = _copy_start(motif_h, 2) + motif_h
        inp = full.copy()
        inp[third_end:] = BACKGROUND
        if _valid_candidate(inp, motif, colors):
            return inp
    raise RuntimeError("failed to generate a valid bracket-motif input")
