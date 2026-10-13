from __future__ import annotations

import numpy as np


BACKGROUND = 0
CROSS = 5
KEY_COLORS = (1, 2, 3, 4)
SIDE_CHOICES = tuple(range(15, 26, 2))
CENTER_MARGIN = 4
KEY_DISTANCE_CHOICES = (2, 3, 4)
MAX_ATTEMPTS = 400
QUADRANTS = (
    ("tl", -1, -1),
    ("tr", -1, 1),
    ("bl", 1, -1),
    ("br", 1, 1),
)


def _find_cross(grid: np.ndarray) -> tuple[int, int]:
    full_rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == CROSS)]
    full_cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == CROSS)]
    assert len(full_rows) == 1, "expected exactly one full grey row"
    assert len(full_cols) == 1, "expected exactly one full grey column"
    cr, cc = full_rows[0], full_cols[0]

    rows, cols = np.indices(grid.shape)
    cross_mask = (rows == cr) | (cols == cc)
    assert np.array_equal(grid == CROSS, cross_mask), "grey pixels must form exactly the cross"
    return int(cr), int(cc)


def _quadrant_mask(shape: tuple[int, int], cr: int, cc: int, name: str) -> np.ndarray:
    rows, cols = np.indices(shape)
    if name == "tl":
        return (rows < cr) & (cols < cc)
    if name == "tr":
        return (rows < cr) & (cols > cc)
    if name == "bl":
        return (rows > cr) & (cols < cc)
    if name == "br":
        return (rows > cr) & (cols > cc)
    raise AssertionError(f"unknown quadrant: {name}")


def _extract_keys(grid: np.ndarray, cr: int, cc: int) -> dict[str, tuple[tuple[int, int], int]]:
    keys: dict[str, tuple[tuple[int, int], int]] = {}
    seen_colors: set[int] = set()

    for name, _sr, _sc in QUADRANTS:
        mask = _quadrant_mask(grid.shape, cr, cc, name)
        coords = np.argwhere(mask & (grid != BACKGROUND))
        assert coords.shape[0] == 1, "each quadrant must contain exactly one key pixel"
        r, c = map(int, coords[0])
        color = int(grid[r, c])
        assert color not in (BACKGROUND, CROSS), "key must not use background or cross color"
        seen_colors.add(color)
        keys[name] = ((r, c), color)

    assert len(seen_colors) == 4, "quadrant key colors must be distinct"
    return keys


def _render_output(shape: tuple[int, int], cr: int, cc: int, keys: dict[str, tuple[tuple[int, int], int]]) -> np.ndarray:
    rows, cols = np.indices(shape)
    distance = np.maximum(np.abs(rows - cr), np.abs(cols - cc))
    out = np.full(shape, BACKGROUND, dtype=int)
    out[(rows == cr) | (cols == cc)] = CROSS

    for name, (_pos, color) in keys.items():
        mask = _quadrant_mask(shape, cr, cc, name)
        out[mask & (distance % 2 == 0)] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Color even Chebyshev-distance rings from the cross center, using the quadrant key color in each sector.

    Input structure:
    - The grid is a square black background with one full grey row and one full grey column forming a cross.
    - The cross partitions the non-cross area into four quadrants.
    - Each quadrant contains exactly one non-grey key pixel, and the four key colors are distinct.

    Rule:
    - Find the unique intersection of the full grey row and full grey column.
    - Read the single key color inside each quadrant.
    - For every non-cross cell, compute its Chebyshev distance to the cross intersection; if the distance is even, paint that cell with its quadrant's key color, otherwise leave it black.
    - Replace the key pixels by that same parity rule instead of preserving them, and keep the grey cross unchanged.

    Key insight:
    - The distance field is global and centered on the cross, but the color of each even-distance ring segment is inherited locally from the quadrant's key pixel.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape[0] == g.shape[1], "expected a square grid"

    cr, cc = _find_cross(g)
    keys = _extract_keys(g, cr, cc)
    return _render_output(g.shape, cr, cc, keys)


def _key_candidates(cr: int, cc: int, n: int, sr: int, sc: int, distance: int) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    for ar in range(1, distance + 1):
        for ac in range(1, distance + 1):
            if max(ar, ac) != distance:
                continue
            r = cr + sr * ar
            c = cc + sc * ac
            if 0 <= r < n and 0 <= c < n:
                candidates.append((r, c))
    return candidates


def _sample_key_distances(rng: np.random.Generator) -> list[int]:
    while True:
        distances = [int(rng.choice(KEY_DISTANCE_CHOICES)) for _ in QUADRANTS]
        parities = {d % 2 for d in distances}
        if len(parities) == 2 and len(set(distances)) >= 2:
            return distances


def _build_candidate_input(rng: np.random.Generator) -> np.ndarray:
    n = int(rng.choice(SIDE_CHOICES))
    cr = int(rng.integers(CENTER_MARGIN, n - CENTER_MARGIN))
    cc = int(rng.integers(CENTER_MARGIN, n - CENTER_MARGIN))

    grid = np.full((n, n), BACKGROUND, dtype=int)
    grid[cr, :] = CROSS
    grid[:, cc] = CROSS

    colors = rng.permutation(np.array(KEY_COLORS, dtype=int)).tolist()
    distances = _sample_key_distances(rng)

    for (name, sr, sc), color, distance in zip(QUADRANTS, colors, distances):
        candidates = _key_candidates(cr, cc, n, sr, sc, distance)
        assert candidates, "center margin should guarantee a valid key position"
        r, c = candidates[int(rng.integers(len(candidates)))]
        grid[r, c] = int(color)

    return grid


def _validate_input(grid: np.ndarray) -> tuple[int, int, dict[str, tuple[tuple[int, int], int]]] | None:
    try:
        if grid.ndim != 2 or grid.shape[0] != grid.shape[1]:
            return None
        if grid.shape[0] > 30:
            return None
        if not set(np.unique(grid)).issubset({BACKGROUND, CROSS, *KEY_COLORS}):
            return None

        cr, cc = _find_cross(grid)
        keys = _extract_keys(grid, cr, cc)

        distances = []
        for pos, _color in keys.values():
            r, c = pos
            d = max(abs(r - cr), abs(c - cc))
            if d not in KEY_DISTANCE_CHOICES:
                return None
            distances.append(d)

        if len({d % 2 for d in distances}) != 2:
            return None
        if len(set(distances)) < 2:
            return None
        return cr, cc, keys
    except AssertionError:
        return None


def _oracle_matches(grid: np.ndarray) -> bool:
    info = _validate_input(grid)
    if info is None:
        return False

    cr, cc, keys = info
    expected = _render_output(grid.shape, cr, cc, keys)

    try:
        actual = main(grid)
    except AssertionError:
        return False

    if not np.array_equal(actual, expected):
        return False
    if np.array_equal(actual, grid):
        return False
    if set(np.unique(actual)) != {BACKGROUND, CROSS, *KEY_COLORS}:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        rng = np.random.default_rng()
        grid = _build_candidate_input(rng)
        if _oracle_matches(grid):
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
