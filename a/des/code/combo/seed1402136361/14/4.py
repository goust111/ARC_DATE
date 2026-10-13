import random
import numpy as np


def _dominant_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    max_count = int(counts.max())
    assert int(np.count_nonzero(counts == max_count)) == 1
    return int(vals[counts.argmax()])


def _centered_chebyshev_ring(n: int, dist: int) -> list[tuple[int, int]]:
    center = n // 2
    coords: list[tuple[int, int]] = []
    for r in range(n):
        for c in range(n):
            if max(abs(r - center), abs(c - center)) == dist:
                coords.append((r, c))
    return coords


def _render_ranked_diamond_rings(n: int, bg: int, ordered_colors: list[int]) -> np.ndarray:
    center = n // 2
    rr = np.arange(n)[:, None]
    cc = np.arange(n)[None, :]
    manhattan = np.abs(rr - center) + np.abs(cc - center)

    out = np.full((n, n), bg, dtype=int)
    max_ring = len(ordered_colors) - 1
    for index, color in enumerate(ordered_colors):
        ring_radius = max_ring - index
        out[manhattan == ring_radius] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Rank scattered markers by Chebyshev distance to the center and redraw them as concentric diamond rings.

    Input structure:
    - The grid is an odd-sized square with a unique dominant background color.
    - Each non-background color appears on exactly one pixel.
    - All marker pixels have different Chebyshev distances to the grid center.

    Rule:
    - Compute each marker's distance to the center using max(row offset, col offset).
    - Sort markers from farthest to nearest.
    - Redraw their colors as one-pixel-thick concentric diamond rings centered on the grid center:
      farthest -> outermost drawn ring, nearest -> center pixel.
    - Original marker locations are discarded.

    Cells outside the outermost drawn diamond stay as background when the input contains
    fewer markers than the maximum possible number of layers for that canvas size.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert h == w and h % 2 == 1

    bg = _dominant_color(g)
    center = h // 2
    markers = np.argwhere(g != bg)
    assert len(markers) > 0
    assert len(markers) <= center + 1

    colors = [int(g[r, c]) for r, c in markers]
    assert len(set(colors)) == len(colors)

    by_distance: dict[int, int] = {}
    for r, c in markers:
        dist = max(abs(int(r) - center), abs(int(c) - center))
        assert dist not in by_distance
        by_distance[dist] = int(g[r, c])

    ordered_colors = [color for _, color in sorted(by_distance.items(), key=lambda item: item[0], reverse=True)]
    return _render_ranked_diamond_rings(h, bg, ordered_colors)


GEN_GRID_SIZES = (5, 7, 9, 11, 13, 15)
GEN_COLORS = tuple(range(10))
GEN_ATTEMPTS = 400


def _sample_marker_count(max_count: int) -> int:
    mode = random.choice(("few", "mixed", "many"))
    if mode == "few":
        return random.randint(1, min(3, max_count))
    if mode == "many":
        return random.randint(max(1, max_count - 2), max_count)
    return random.randint(1, max_count)


def _sample_distances(radius: int, count: int) -> list[int]:
    available = list(range(radius + 1))
    chosen: set[int] = set()

    if count > 0 and random.random() < 0.5:
        chosen.add(0)
    if len(chosen) < count and random.random() < 0.5:
        chosen.add(radius)

    while len(chosen) < count:
        chosen.add(random.choice(available))

    distances = list(chosen)
    random.shuffle(distances)
    return distances


def _gen_attempt() -> np.ndarray | None:
    n = random.choice(GEN_GRID_SIZES)
    radius = n // 2
    marker_count = _sample_marker_count(radius + 1)
    distances = _sample_distances(radius, marker_count)

    bg = random.choice(GEN_COLORS)
    palette = [color for color in GEN_COLORS if color != bg]
    random.shuffle(palette)
    marker_colors = palette[:marker_count]

    grid = np.full((n, n), bg, dtype=int)
    for color, dist in zip(marker_colors, distances):
        candidates = _centered_chebyshev_ring(n, dist)
        r, c = random.choice(candidates)
        grid[r, c] = color

    try:
        out = main(grid)
    except Exception:
        return None

    if not isinstance(out, np.ndarray) or out.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
