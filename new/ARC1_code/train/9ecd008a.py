import numpy as np

from arc_utils import Color, bbox


def main(grid: np.ndarray) -> np.ndarray:
    """
    Return the 3x3 patch that completes the grid's left-right mirror symmetry.

    Input structure
    - The input is a horizontally mirrored pattern with one compact 3x3 black missing patch.
    - The missing patch's reflected counterpart across the vertical axis is fully colored.

    Rule
    - Find the black 3x3 patch.
    - For each patch cell, copy the color from the same row and mirror-reflected column.
    - Output only the recovered 3x3 patch, not the full repaired input grid.

    Key insight
    - The black cells are not background; they mark a cropped hole whose values are determined by left-right reflection.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    missing = g == Color.BLACK
    b = bbox(missing)
    assert b is not None, "expected black missing-patch cells"
    r0, c0, r1, c1 = b
    assert (r1 - r0 + 1, c1 - c0 + 1) == (3, 3), "expected a 3x3 missing patch"
    assert int(missing.sum()) == 9, "expected exactly one 3x3 black patch"
    assert np.all(missing[r0 : r1 + 1, c0 : c1 + 1]), "missing patch must be compact"

    _, width = g.shape
    src_c0 = width - 1 - c1
    src_c1 = width - 1 - c0
    assert 0 <= src_c0 <= src_c1 < width, "reflected patch must be in bounds"

    patch = g[r0 : r1 + 1, src_c0 : src_c1 + 1][:, ::-1].copy()
    assert patch.shape == (3, 3), "expected reflected source to be 3x3"
    assert not np.any(patch == Color.BLACK), "reflected source patch must be fully colored"
    return patch

# --------------- generator (o5x) ---------------
# CFG: palette, geometry, hole placement, and retry-budget controls.
_GEN_COLOR_VALUES = np.arange(1, 10)  # non-black palette
_GEN_COLOR_COUNT_RANGE = (7, 9)
_GEN_GRID_SIZE, _GEN_PATCH_SIZE = 16, 3
_GEN_QUADRANT_SIZE = _GEN_GRID_SIZE // 2
_GEN_UPPER_TRIANGLE = [(r, c) for r in range(_GEN_QUADRANT_SIZE) for c in range(r, _GEN_QUADRANT_SIZE)]
_GEN_HOLE_ROW_RANGE = (0, _GEN_GRID_SIZE - _GEN_PATCH_SIZE + 1)  # RNG upper exclusive
_GEN_HOLE_COL_RANGES = ((0, _GEN_QUADRANT_SIZE - _GEN_PATCH_SIZE + 1), (_GEN_QUADRANT_SIZE, _GEN_GRID_SIZE - _GEN_PATCH_SIZE + 1))
_GEN_HOLE_LEFT_PROB = 0.5
_GEN_MAX_ATTEMPTS = 300
_GEN_RNG = np.random.default_rng()


def _gen_full_pattern() -> np.ndarray:
    color_count = int(_GEN_RNG.integers(*_GEN_COLOR_COUNT_RANGE))
    colors = _GEN_RNG.choice(_GEN_COLOR_VALUES, size=color_count, replace=False)

    vals = np.empty(len(_GEN_UPPER_TRIANGLE), dtype=int)
    vals[:color_count] = colors
    vals[color_count:] = _GEN_RNG.choice(colors, size=len(vals) - color_count)
    _GEN_RNG.shuffle(vals)

    q = np.zeros((_GEN_QUADRANT_SIZE, _GEN_QUADRANT_SIZE), dtype=int)
    for (r, c), value in zip(_GEN_UPPER_TRIANGLE, vals):
        q[r, c] = int(value)
        q[c, r] = int(value)

    top = np.concatenate([q, np.fliplr(q)], axis=1)
    return np.concatenate([top, np.flipud(top)], axis=0)


def _gen_attempt() -> np.ndarray | None:
    full = _gen_full_pattern()

    r0 = int(_GEN_RNG.integers(*_GEN_HOLE_ROW_RANGE))
    col_range = _GEN_HOLE_COL_RANGES[0 if _GEN_RNG.random() < _GEN_HOLE_LEFT_PROB else 1]
    c0 = int(_GEN_RNG.integers(*col_range))
    expected = full[r0 : r0 + _GEN_PATCH_SIZE, c0 : c0 + _GEN_PATCH_SIZE].copy()

    inp = full.copy()
    inp[r0 : r0 + _GEN_PATCH_SIZE, c0 : c0 + _GEN_PATCH_SIZE] = 0

    if np.array_equal(main(inp), expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {_GEN_MAX_ATTEMPTS} attempts")
