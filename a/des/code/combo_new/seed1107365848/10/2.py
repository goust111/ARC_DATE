from __future__ import annotations

from itertools import combinations

import numpy as np


ARM_LEN = 2
BG = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
STAR_COUNT_MIN = 2
STAR_COUNT_MAX_EXCLUSIVE = 7
GRID_EXTRA_MAX = 10
MAX_GENERATION_ATTEMPTS = 400

_RNG = np.random.default_rng()


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _plus_mask(shape: tuple[int, int], center: tuple[int, int]) -> np.ndarray:
    h, w = shape
    r, c = center
    assert ARM_LEN <= r < h - ARM_LEN
    assert ARM_LEN <= c < w - ARM_LEN
    mask = np.zeros(shape, dtype=bool)
    mask[r - ARM_LEN : r + ARM_LEN + 1, c] = True
    mask[r, c - ARM_LEN : c + ARM_LEN + 1] = True
    return mask


def _candidate_centers(coords: np.ndarray, shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    candidates: set[tuple[int, int]] | None = None
    for r_raw, c_raw in coords:
        r = int(r_raw)
        c = int(c_raw)
        local = set()
        for delta in range(-ARM_LEN, ARM_LEN + 1):
            cc = c + delta
            rr = r + delta
            if 0 <= cc < w:
                local.add((r, cc))
            if 0 <= rr < h:
                local.add((rr, c))
        candidates = local if candidates is None else (candidates & local)
    assert candidates is not None and candidates, "no feasible star center"
    return sorted(candidates)


def _recover_star(grid: np.ndarray, color: int) -> dict[str, object]:
    coords = np.argwhere(grid == color)
    assert coords.size > 0, "each foreground color must remain visible"
    candidates = _candidate_centers(coords, grid.shape)

    scored: list[tuple[tuple[int, int, int, int, int], tuple[int, int]]] = []
    for center in candidates:
        r, c = center
        on_cross = (
            ((coords[:, 0] == r) & (np.abs(coords[:, 1] - c) <= ARM_LEN))
            | ((coords[:, 1] == c) & (np.abs(coords[:, 0] - r) <= ARM_LEN))
        )
        if not bool(np.all(on_cross)):
            continue

        row_hits = int(np.sum(coords[:, 0] == r))
        col_hits = int(np.sum(coords[:, 1] == c))
        score = (
            int(grid[r, c] == color),
            min(row_hits, col_hits),
            row_hits + col_hits,
            row_hits,
            col_hits,
        )
        scored.append((score, center))

    assert scored, f"failed to score star center for color {color}"
    scored.sort(reverse=True)
    best_score = scored[0][0]
    best = [center for score, center in scored if score == best_score]
    assert len(best) == 1, f"ambiguous star center for color {color}: {best}"

    center = best[0]
    mask = _plus_mask(grid.shape, center)
    assert bool(np.all(mask[coords[:, 0], coords[:, 1]])), "visible pixels must lie on inferred cross"
    return {"color": int(color), "center": center, "mask": mask}


def _infer_edges(grid: np.ndarray, stars: dict[int, dict[str, object]], bg: int) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for a, b in combinations(sorted(stars), 2):
        overlap = np.argwhere(stars[a]["mask"] & stars[b]["mask"])
        if overlap.size == 0:
            continue
        seen = {
            int(grid[int(r), int(c)])
            for r, c in overlap
            if int(grid[int(r), int(c)]) in {a, b}
        }
        if len(seen) != 1:
            continue
        top = next(iter(seen))
        if top == a:
            edges.add((b, a))
        else:
            edges.add((a, b))

    if len(stars) > 1:
        assert edges, "expected visible overlap evidence between stars"
    return edges


def _unique_order(colors: list[int], edges: set[tuple[int, int]]) -> list[int]:
    if len(colors) <= 1:
        return list(colors)

    predecessors = {color: set() for color in colors}
    for lower, upper in edges:
        if lower in predecessors and upper in predecessors:
            predecessors[upper].add(lower)

    remaining = set(colors)
    order: list[int] = []
    while remaining:
        roots = [color for color in colors if color in remaining and not (predecessors[color] & remaining)]
        assert len(roots) == 1, f"overlap graph does not determine a unique order: {roots}"
        order.append(roots[0])
        remaining.remove(roots[0])
    return order


def _render_diamonds(order: list[int], bg: int) -> np.ndarray:
    n = len(order)
    size = 2 * n - 1
    center = n - 1
    rows, cols = np.indices((size, size))
    dist = np.abs(rows - center) + np.abs(cols - center)
    out = np.full((size, size), bg, dtype=int)
    for layer, color in enumerate(order):
        out[dist == (n - 1 - layer)] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Untangle overlapping colored plus-stars into concentric diamond rings ordered by occlusion.

    Input structure:
    - The background is the dominant color.
    - Each foreground color belongs to one radius-2 plus-star: a center pixel with four orthogonal arms.
    - Several stars overlap, so some arm crossings are overwritten by later-painted colors while each star still has visible pixels.

    Rule:
    - Reconstruct each color's full radius-2 plus-star from its visible pixels.
    - At every pairwise crossing, treat the visible color as the upper star and the hidden one as the lower star; collect these relations into one unique bottom-to-top order.
    - Output a square grid of size (2N-1) x (2N-1), where N is the number of star colors, and redraw that order as tightly nested diamond shells from outside to inside.

    Key insight:
    - The input geometry is only evidence for occlusion order: once the crossing graph reveals which star is below which, the stars themselves disappear and only their color order survives as concentric Manhattan-distance rings.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _background_color(g)
    colors = [int(color) for color in np.unique(g) if int(color) != bg]
    assert colors, "expected at least one foreground star"

    stars = {color: _recover_star(g, color) for color in colors}
    edges = _infer_edges(g, stars, bg)
    order = _unique_order(colors, edges)
    return _render_diamonds(order, bg)


def _axis_positions(n: int) -> list[int]:
    positions = [0]
    for _ in range(n - 1):
        positions.append(positions[-1] + int(_RNG.integers(1, ARM_LEN + 1)))
    return positions


def _oriented_positions(base: list[int], span: int, direction: int, shift: int) -> list[int]:
    if direction > 0:
        return [shift + value for value in base]
    return [shift + (span - 1 - value) for value in base]


def _sample_centers(n: int) -> tuple[tuple[int, int], list[tuple[int, int]]]:
    row_base = _axis_positions(n)
    col_base = _axis_positions(n)
    row_span = row_base[-1] + 1
    col_span = col_base[-1] + 1

    min_h = row_span + 2 * ARM_LEN
    min_w = col_span + 2 * ARM_LEN
    if min_h > 30 or min_w > 30:
        raise AssertionError("star chain does not fit ARC size limit")

    h = int(min_h + _RNG.integers(0, min(GRID_EXTRA_MAX, 30 - min_h) + 1))
    w = int(min_w + _RNG.integers(0, min(GRID_EXTRA_MAX, 30 - min_w) + 1))

    row_shift = int(_RNG.integers(ARM_LEN, h - ARM_LEN - row_span + 1))
    col_shift = int(_RNG.integers(ARM_LEN, w - ARM_LEN - col_span + 1))
    row_dir = 1 if bool(_RNG.integers(0, 2)) else -1
    col_dir = 1 if bool(_RNG.integers(0, 2)) else -1

    rows = _oriented_positions(row_base, row_span, row_dir, row_shift)
    cols = _oriented_positions(col_base, col_span, col_dir, col_shift)
    centers = [(int(r), int(c)) for r, c in zip(rows, cols, strict=True)]
    return (h, w), centers


def _paint_stars(shape: tuple[int, int], colors: list[int], centers: list[tuple[int, int]]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for color, center in zip(colors, centers, strict=True):
        grid[_plus_mask(shape, center)] = int(color)
    return grid


def _expected_output(order: list[int]) -> np.ndarray:
    return _render_diamonds(order, BG)


def _gen_attempt() -> np.ndarray | None:
    n = int(_RNG.integers(STAR_COUNT_MIN, STAR_COUNT_MAX_EXCLUSIVE))
    colors = [int(color) for color in _RNG.choice(PALETTE, size=n, replace=False)]
    shape, centers = _sample_centers(n)
    grid = _paint_stars(shape, colors, centers)

    try:
        actual = main(grid)
    except Exception:
        return None

    expected = _expected_output(colors)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid overlapping-star puzzle")
