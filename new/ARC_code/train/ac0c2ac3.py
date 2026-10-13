import numpy as np

from arc_utils import get_bg


def main(grid: np.ndarray) -> np.ndarray:
    """Fill each centered square ring with the color of the marker at that ring distance.

    Input structure
    The input is an odd square grid with a dominant background color and one
    non-background marker for every Chebyshev distance from the grid center.
    Rule
    The output replaces the background with concentric square rings centered on
    the middle cell; every cell at distance d from the center takes the color of
    the unique marker whose distance from the center is d.
    Key insight
    The scattered markers are not objects to move; their Chebyshev distances
    form a palette for the centered square rings.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert h == w and h % 2 == 1

    bg = get_bg(g)
    _, counts = np.unique(g, return_counts=True)
    max_count = int(counts.max())
    assert int(np.count_nonzero(counts == max_count)) == 1
    assert int(np.count_nonzero(g == bg)) == max_count

    center_r = h // 2
    center_c = w // 2
    radius = h // 2
    markers = np.argwhere(g != bg)
    assert len(markers) == radius + 1

    ring_colors: dict[int, int] = {}
    for r, c in markers:
        dist = max(abs(int(r) - center_r), abs(int(c) - center_c))
        assert dist not in ring_colors
        ring_colors[dist] = int(g[r, c])
    assert set(ring_colors) == set(range(radius + 1))

    rr = np.arange(h)[:, None]
    cc = np.arange(w)[None, :]
    distances = np.maximum(np.abs(rr - center_r), np.abs(cc - center_c))
    out = np.empty_like(g)
    for dist in range(radius + 1):
        out[distances == dist] = ring_colors[dist]
    return out

# --------------- generator (att2) ---------------

# Canvas sizes are odd squares; radius is derived as side // 2.
GEN_GRID_SIZES = (7, 9, 11, 13)
# ARC colors: one background plus distinct marker/ring colors.
GEN_COLORS = tuple(range(10))
# Retry budget for the solver-oracle guard.
GEN_MAX_ATTEMPTS = 300

def _gen_ring_coords(n: int, dist: int) -> np.ndarray:
    center = n // 2
    rr = np.arange(n)[:, None]
    cc = np.arange(n)[None, :]
    mask = np.maximum(np.abs(rr - center), np.abs(cc - center)) == dist
    return np.argwhere(mask)


def _gen_grids(
    n: int, bg: int, ring_colors: list[int], rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    coords_by_dist = [_gen_ring_coords(n, dist) for dist in range(n // 2 + 1)]
    out = np.empty((n, n), dtype=int)
    inp = np.full((n, n), bg, dtype=int)
    for color, coords in zip(ring_colors, coords_by_dist):
        out[coords[:, 0], coords[:, 1]] = color
        marker_idx = int(rng.integers(0, len(coords)))
        r, c = coords[marker_idx]
        inp[int(r), int(c)] = color
    return inp, out


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    n = int(rng.choice(GEN_GRID_SIZES))
    radius = n // 2
    bg = int(rng.integers(0, len(GEN_COLORS)))
    palette = [color for color in GEN_COLORS if color != bg]
    rng.shuffle(palette)
    ring_colors = palette[: radius + 1]
    inp, out = _gen_grids(n, bg, ring_colors, rng)
    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid ac0c2ac3 input")
