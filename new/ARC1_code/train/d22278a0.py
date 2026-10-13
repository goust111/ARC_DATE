import numpy as np

from arc_utils import get_bg


def main(grid: np.ndarray) -> np.ndarray:
    """Paint alternating square rings from each colored corner marker, clipped to cells uniquely nearest to that marker.

    Input structure
    - The grid is background except for colored marker pixels in two or more corners.

    Rule
    - For every cell, compare squared Euclidean distance to each corner marker.
    - Cells tied between markers remain background.
    - A uniquely owned cell takes its marker color exactly when its Chebyshev distance
      from that marker is even.

    Key insight
    - The output is not made from lines between corners; it is the set of even
      square-distance rings around each active corner, cut by nearest-marker
      boundaries.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    bg = get_bg(g)

    corners = {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}
    marker_coords = np.argwhere(g != bg)
    assert len(marker_coords) >= 2

    markers: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int]] = set()
    for coord in marker_coords:
        r, c = map(int, coord)
        assert (r, c) in corners
        assert (r, c) not in seen
        seen.add((r, c))
        markers.append((r, c, int(g[r, c])))

    out = np.full_like(g, bg)
    for r in range(h):
        for c in range(w):
            distances = [
                (r - r0) * (r - r0) + (c - c0) * (c - c0)
                for r0, c0, _color in markers
            ]
            best = min(distances)
            if distances.count(best) != 1:
                continue

            r0, c0, color = markers[distances.index(best)]
            if max(abs(r - r0), abs(c - c0)) % 2 == 0:
                out[r, c] = color

    return out

# --------------- generator (o5x) ---------------

# Sampling bounds, marker/color choices, and retry validation.
GRID_SIZE_RANGE, MARKER_COUNT_RANGE = (7, 18), (2, 4)
COLOR_VALUES = np.arange(1, 10)
MAX_ATTEMPTS = 300
MIN_OUTPUT_COLORS = 2

def _gen_attempt() -> np.ndarray | None:
    n = int(np.random.randint(*GRID_SIZE_RANGE))
    grid = np.zeros((n, n), dtype=int)

    corners = [(0, 0), (0, n - 1), (n - 1, 0), (n - 1, n - 1)]
    marker_count = int(np.random.randint(*MARKER_COUNT_RANGE))
    chosen = np.random.permutation(len(corners))[:marker_count]
    colors = np.random.choice(COLOR_VALUES, size=marker_count, replace=False)

    for corner_i, color in zip(chosen, colors):
        r, c = corners[int(corner_i)]
        grid[r, c] = int(color)

    out = main(grid)
    if len([color for color in np.unique(out) if color != 0]) < MIN_OUTPUT_COLORS:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        inp = _gen_attempt()
        if inp is None:
            continue
        out = main(inp)
        if out.shape == inp.shape and len([color for color in np.unique(out) if color != 0]) >= MIN_OUTPUT_COLORS:
            return inp
    raise RuntimeError("failed to generate valid d22278a0 input")
