from __future__ import annotations

import numpy as np


BLACK = 0
GREY = 5
MARKER_PALETTE = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)
QUADRANTS = ("ul", "ur", "ll", "lr")
OPPOSITE = {"ul": "lr", "ur": "ll", "ll": "ur", "lr": "ul"}
CORNER_OFFSET = {"ul": (0, 0), "ur": (0, 1), "ll": (1, 0), "lr": (1, 1)}
RAY_STEP = {"ul": (-1, -1), "ur": (-1, 1), "ll": (1, -1), "lr": (1, 1)}
DISTANCES = (2, 3, 4, 5)
_GEN_RNG = np.random.default_rng()


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def _classify_quadrant(r: int, c: int, r0: int, c0: int, r1: int, c1: int) -> str:
    if r < r0 and c < c0:
        return "ul"
    if r < r0 and c > c1:
        return "ur"
    if r > r1 and c < c0:
        return "ll"
    if r > r1 and c > c1:
        return "lr"
    raise AssertionError("seed must lie in a strict diagonal quadrant")


def _nearest_corner(block_r0: int, block_c0: int, quad: str) -> tuple[int, int]:
    dr, dc = CORNER_OFFSET[quad]
    return int(block_r0 + dr), int(block_c0 + dc)


def _ray_cells(shape: tuple[int, int], start: tuple[int, int], step: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = map(int, start)
    dr, dc = map(int, step)
    cells = []
    while _in_bounds(shape, r, c):
        cells.append((int(r), int(c)))
        r += dr
        c += dc
    return cells


def _parse_input(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    marker_coords = np.argwhere((g != bg) & (g != GREY))
    assert marker_coords.shape == (4, 2), "expected exactly four non-grey marker pixels"
    r0, c0, r1, c1 = _bbox_coords(marker_coords)
    assert (r1 - r0, c1 - c0) == (1, 1), "marker pixels must form a 2x2 block"
    marker_patch = g[r0 : r1 + 1, c0 : c1 + 1]
    assert np.all(marker_patch != bg) and np.all(marker_patch != GREY), "marker block must be fully colored"
    marker_colors = tuple(int(v) for v in marker_patch.reshape(-1))
    assert len(set(marker_colors)) == 4, "marker colors must all be distinct"

    grey_coords = np.argwhere(g == GREY)
    assert grey_coords.shape == (4, 2), "expected exactly four grey seed pixels"
    assert int(np.count_nonzero(g != bg)) == 8, "input must contain only the marker block and four seeds"

    seeds: dict[str, tuple[int, int]] = {}
    dists: dict[str, int] = {}
    for sr, sc in grey_coords:
        quad = _classify_quadrant(int(sr), int(sc), r0, c0, r1, c1)
        assert quad not in seeds, "expected exactly one seed in each diagonal quadrant"
        seeds[quad] = (int(sr), int(sc))
        cr, cc = _nearest_corner(r0, c0, quad)
        dists[quad] = abs(int(sr) - cr) + abs(int(sc) - cc)

    assert set(seeds) == set(QUADRANTS), "all four diagonal quadrants must contain a seed"
    assert set(dists.values()) == set(DISTANCES), "seed distances must be exactly 2, 3, 4, and 5"

    colors = {
        "ul": int(marker_patch[0, 0]),
        "ur": int(marker_patch[0, 1]),
        "ll": int(marker_patch[1, 0]),
        "lr": int(marker_patch[1, 1]),
    }
    return {
        "bg": int(bg),
        "block_r0": int(r0),
        "block_c0": int(c0),
        "block_r1": int(r1),
        "block_c1": int(c1),
        "colors": colors,
        "seeds": seeds,
        "dists": dists,
    }


def _render_output(grid: np.ndarray, parsed: dict) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for seed_quad in QUADRANTS:
        sr, sc = parsed["seeds"][seed_quad]
        d = int(parsed["dists"][seed_quad])
        color = int(parsed["colors"][OPPOSITE[seed_quad]])
        cells = _ray_cells(out.shape, (sr, sc), RAY_STEP[seed_quad])
        for idx, (rr, cc) in enumerate(cells):
            if (idx // d) % 2 == 0:
                out[rr, cc] = color
            else:
                out[rr, cc] = parsed["bg"]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire four blinking diagonal rays whose colors come from the opposite marker-block corners and whose periods come from seed-to-block distances.

    Input structure:
    - The background is black, with a single 2x2 colored marker block whose four colors are distinct and non-grey.
    - There is exactly one grey seed pixel in each strict diagonal quadrant around that 2x2 block.
    - No other foreground pixels are present.

    Rule:
    - Keep the 2x2 marker block unchanged.
    - For each seed, measure its Manhattan distance to the nearest corner of the marker block in the same quadrant; that distance is the blink period.
    - Recolor the seed and extend a diagonal ray away from the marker block, using the color from the diagonally opposite marker-block cell.
    - Along that ray, paint `d` consecutive cells, leave the next `d` cells black, and continue alternating until the ray exits the grid.

    Key insight:
    - The 2x2 marker block compactly encodes two coupled parameters at once: opposite-corner matching chooses each ray color, while the seed's spatial distance to its local corner chooses that ray's paint/skip period.
    """
    parsed = _parse_input(grid)
    return _render_output(grid, parsed)


def _outward_room(shape: tuple[int, int], seed: tuple[int, int], quad: str) -> int:
    h, w = map(int, shape)
    sr, sc = map(int, seed)
    if quad == "ul":
        return min(sr, sc)
    if quad == "ur":
        return min(sr, w - 1 - sc)
    if quad == "ll":
        return min(h - 1 - sr, sc)
    return min(h - 1 - sr, w - 1 - sc)


def _choose_biased(lo: int, hi: int, target: int, *, prefer_left: bool = False) -> int:
    options = np.arange(int(lo), int(hi) + 1, dtype=int)
    weights = 1.0 / (1.0 + np.abs(options - int(target)))
    if prefer_left:
        midpoint = (lo + hi) / 2.0
        weights *= np.where(options <= midpoint, 1.5, 0.75)
    weights /= weights.sum()
    return int(_GEN_RNG.choice(options, p=weights))


def _seed_from_block(block_r0: int, block_c0: int, quad: str, row_gap: int, col_gap: int) -> tuple[int, int]:
    if quad == "ul":
        return int(block_r0 - row_gap), int(block_c0 - col_gap)
    if quad == "ur":
        return int(block_r0 - row_gap), int(block_c0 + 1 + col_gap)
    if quad == "ll":
        return int(block_r0 + 1 + row_gap), int(block_c0 - col_gap)
    return int(block_r0 + 1 + row_gap), int(block_c0 + 1 + col_gap)


def _generate_candidate() -> np.ndarray | None:
    h = int(_GEN_RNG.integers(18, 27))
    w = int(_GEN_RNG.integers(18, 27))

    dist_perm = list(_GEN_RNG.permutation(np.array(DISTANCES, dtype=int)))
    dists = {quad: int(dist_perm[i]) for i, quad in enumerate(QUADRANTS)}
    splits = {}
    for quad in QUADRANTS:
        d = dists[quad]
        row_gap = int(_GEN_RNG.integers(1, d))
        splits[quad] = (row_gap, int(d - row_gap))

    r_lo = max(6 + splits["ul"][0], 6 + splits["ur"][0])
    r_hi = min(h - 8 - splits["ll"][0], h - 8 - splits["lr"][0])
    c_lo = max(6 + splits["ul"][1], 6 + splits["ll"][1])
    c_hi = min(w - 8 - splits["ur"][1], w - 8 - splits["lr"][1])
    if r_lo > r_hi or c_lo > c_hi:
        return None

    target_r = (h - 2) // 2
    target_c = max(c_lo, min(c_hi, w // 2 - 2))
    block_r0 = _choose_biased(r_lo, r_hi, target_r)
    block_c0 = _choose_biased(c_lo, c_hi, target_c, prefer_left=True)

    colors = list(map(int, _GEN_RNG.choice(MARKER_PALETTE, size=4, replace=False)))
    grid = np.full((h, w), BLACK, dtype=int)
    grid[block_r0 : block_r0 + 2, block_c0 : block_c0 + 2] = np.array(
        [[colors[0], colors[1]], [colors[2], colors[3]]],
        dtype=int,
    )

    for quad in QUADRANTS:
        row_gap, col_gap = splits[quad]
        sr, sc = _seed_from_block(block_r0, block_c0, quad, row_gap, col_gap)
        if not _in_bounds(grid.shape, sr, sc) or grid[sr, sc] != BLACK:
            return None
        grid[sr, sc] = GREY
    return grid


def _validate_semantics(grid: np.ndarray) -> bool:
    try:
        parsed = _parse_input(grid)
    except Exception:
        return False

    for quad in QUADRANTS:
        room = _outward_room(grid.shape, parsed["seeds"][quad], quad)
        d = int(parsed["dists"][quad])
        if room < 6 or room <= d:
            return False

    out = _render_output(grid, parsed)
    if GREY in set(map(int, np.unique(out))):
        return False
    if np.array_equal(out, grid):
        return False

    for quad in QUADRANTS:
        d = int(parsed["dists"][quad])
        color = int(parsed["colors"][OPPOSITE[quad]])
        cells = _ray_cells(grid.shape, parsed["seeds"][quad], RAY_STEP[quad])
        if len(cells) <= d:
            return False
        painted = 0
        blank_after_paint = 0
        for idx, (r, c) in enumerate(cells):
            expected = color if (idx // d) % 2 == 0 else BLACK
            if int(out[r, c]) != expected:
                return False
            if expected == color:
                painted += 1
            else:
                blank_after_paint += 1
        if painted < d or blank_after_paint == 0:
            return False

    return np.array_equal(main(grid), out)


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _generate_candidate()
        if grid is not None and _validate_semantics(grid):
            return grid
    raise RuntimeError("failed to generate a valid marker-ray instance")
