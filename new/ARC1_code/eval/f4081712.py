import numpy as np

from arc_utils import bbox, components_mask


def _symmetry_coords(r: int, c: int, n: int) -> tuple[tuple[int, int], ...]:
    last = n - 1
    return (
        (r, c),
        (r, last - c),
        (last - r, c),
        (last - r, last - c),
        (c, r),
        (last - c, last - r),
        (c, last - r),
        (last - c, r),
    )


def main(grid: np.ndarray) -> np.ndarray:
    """Recover the rectangular green mask from the unique visible symmetric copies.

    Input structure
    - The input is a square grid whose non-mask pattern is symmetric under the
      rotations and reflections of the square.
    - Color 3 forms one solid rectangular mask over the hidden part of the
      pattern.

    Rule
    - Return a grid with the same shape as the mask's bounding box.
    - For each masked cell, inspect every rotation/reflection-equivalent
      coordinate and ignore any masked cells; the remaining visible colors must
      agree, and that unique color is emitted at the same relative position.

    Key insight
    - The green rectangle is damage in an otherwise dihedrally symmetric pattern,
      so each hidden pixel can be reconstructed from its visible symmetric twins.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h == w

    mask = g == 3
    b = bbox(mask)
    assert b is not None
    r0, c0, r1, c1 = b
    rect = mask[r0 : r1 + 1, c0 : c1 + 1]
    assert bool(np.all(rect))
    assert int(mask.sum()) == int(rect.size)
    assert len(components_mask(mask, connectivity=4)) == 1

    out = np.empty((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            visible = []
            for rr, cc in _symmetry_coords(r, c, h):
                color = int(g[rr, cc])
                if color != 3:
                    visible.append(color)
            assert visible
            unique = set(visible)
            assert len(unique) == 1
            out[r - r0, c - c0] = unique.pop()

    return out

# --------------- generator (o5x) ---------------

# Canvas/palette.
GRID_SIZE = 24
MASK_COLOR = 3
PALETTE_COLORS = np.array([1, 2, 4, 5, 6, 7, 8, 9], dtype=int)
PALETTE_SIZE_MIN = 5
PALETTE_SIZE_MAX_EXCLUSIVE = 9
# Mask shape constraints.
MASK_HEIGHTS = range(2, 9)
MASK_WIDTHS = range(3, 9)
MASK_AREA_MIN = 9
MASK_AREA_MAX = 56
# Generation retry budget.
GENERATION_RETRIES = 300
VALID_MASK_SHAPES = tuple(
    (h, w) for h in MASK_HEIGHTS for w in MASK_WIDTHS
    if MASK_AREA_MIN <= h * w <= MASK_AREA_MAX
)

def _gen_palette() -> np.ndarray:
    k = int(np.random.randint(PALETTE_SIZE_MIN, PALETTE_SIZE_MAX_EXCLUSIVE))
    palette = np.random.choice(PALETTE_COLORS, size=k, replace=False)
    return palette[np.random.permutation(k)]

def _gen_d4_orbits(n: int) -> list[tuple[tuple[int, int], ...]]:
    seen = np.zeros((n, n), dtype=bool)
    orbits = []
    for r in range(n):
        for c in range(n):
            if seen[r, c]:
                continue
            coords = tuple(sorted(set(_symmetry_coords(r, c, n))))
            for rr, cc in coords:
                seen[rr, cc] = True
            orbits.append(coords)
    return orbits

GRID_ORBITS = _gen_d4_orbits(GRID_SIZE)

def _gen_symmetric_pattern() -> np.ndarray:
    palette = _gen_palette()
    k = len(palette)

    grid = np.empty((GRID_SIZE, GRID_SIZE), dtype=int)
    orbit_order = np.random.permutation(len(GRID_ORBITS))
    for i, orbit_i in enumerate(orbit_order):
        color = int(palette[i]) if i < k else int(np.random.choice(palette))
        for r, c in GRID_ORBITS[int(orbit_i)]:
            grid[r, c] = color
    return grid

def _gen_mask_has_visible_copy(r0: int, c0: int, h: int, w: int) -> bool:
    for r in range(r0, r0 + h):
        for c in range(c0, c0 + w):
            visible = False
            for rr, cc in _symmetry_coords(r, c, GRID_SIZE):
                if not (r0 <= rr < r0 + h and c0 <= cc < c0 + w):
                    visible = True
                    break
            if not visible:
                return False
    return True

def _gen_valid_mask_positions(h: int, w: int) -> list[tuple[int, int]]:
    return [
        (r0, c0)
        for r0 in range(GRID_SIZE - h + 1)
        for c0 in range(GRID_SIZE - w + 1)
        if _gen_mask_has_visible_copy(r0, c0, h, w)
    ]

def _gen_choose_mask_shape() -> tuple[int, int]:
    return VALID_MASK_SHAPES[int(np.random.randint(len(VALID_MASK_SHAPES)))]

GRID_MASK_POSITIONS = {
    shape: tuple(_gen_valid_mask_positions(*shape))
    for shape in VALID_MASK_SHAPES
}

def _gen_attempt() -> np.ndarray | None:
    full = _gen_symmetric_pattern()
    h, w = _gen_choose_mask_shape()
    positions = GRID_MASK_POSITIONS[(h, w)]
    if not positions:
        return None

    r0, c0 = positions[int(np.random.randint(len(positions)))]
    expected = full[r0 : r0 + h, c0 : c0 + w].copy()
    inp = full.copy()
    inp[r0 : r0 + h, c0 : c0 + w] = MASK_COLOR

    try:
        if np.array_equal(main(inp), expected):
            return inp
    except AssertionError:
        return None
    return None

def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid f4081712 input")
