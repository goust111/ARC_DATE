from __future__ import annotations

from collections import Counter
import math

import numpy as np


BLACK = 0
GREY = 5
MAGENTA = 6
PALETTE = (1, 2, 3, 4, 7, 8, 9)

OUTPUT_SIZE = 9
OUTPUT_CROSS = 4
OUTPUT_BLOCKS = (
    (slice(0, 3), slice(0, 3)),
    (slice(0, 3), slice(6, 9)),
    (slice(6, 9), slice(0, 3)),
    (slice(6, 9), slice(6, 9)),
)

# Direction order matches the outer 3x3 ring order used by _arrow_pixels.
DIRECTIONS = (
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
)
PERIMETER = (
    (0, 1),
    (0, 2),
    (1, 2),
    (2, 2),
    (2, 1),
    (2, 0),
    (1, 0),
    (0, 0),
)

GRID_SIDE_MIN = 4
GRID_SIDE_MAX = 8
MAIN_COUNT_MIN = 4
MAIN_COUNT_MAX = 12
NOISE_COUNT_MAX = 3
GEN_RETRIES = 300
LOCAL_RETRIES = 80


def _find_separator_cross(grid: np.ndarray) -> tuple[int, int, int]:
    height, width = grid.shape
    row_candidates = [
        r
        for r in range(height)
        if int(grid[r, 0]) != BLACK and np.all(grid[r, :] == grid[r, 0])
    ]
    col_candidates = [
        c
        for c in range(width)
        if int(grid[0, c]) != BLACK and np.all(grid[:, c] == grid[0, c])
    ]
    matches = [
        (int(grid[r, 0]), int(r), int(c))
        for r in row_candidates
        for c in col_candidates
        if int(grid[r, 0]) == int(grid[0, c])
    ]
    assert len(matches) == 1, "expected one solid separator row/column cross"
    return matches[0]


def _quadrant_slices(row: int, col: int, height: int, width: int) -> list[tuple[slice, slice]]:
    return [
        (slice(0, row), slice(0, col)),
        (slice(0, row), slice(col + 1, width)),
        (slice(row + 1, height), slice(0, col)),
        (slice(row + 1, height), slice(col + 1, width)),
    ]


def _dominant_color(region: np.ndarray) -> tuple[int, int]:
    mask = (region != BLACK) & (region != MAGENTA)
    counts = Counter(map(int, region[mask].reshape(-1)))
    assert counts, "quadrant must contain a dominant non-background color"
    ranked = counts.most_common()
    if len(ranked) > 1:
        assert ranked[0][1] > ranked[1][1], "dominant color must be unique"
    return int(ranked[0][0]), int(ranked[0][1])


def _direction_index(dr: float, dc: float) -> int:
    assert abs(dr) > 1e-9 or abs(dc) > 1e-9, "direction vector must be non-zero"
    best_idx = 0
    best_score = -float("inf")
    vec_norm = math.hypot(dr, dc)
    for idx, (rr, cc) in enumerate(DIRECTIONS):
        score = (dr * rr + dc * cc) / (vec_norm * math.hypot(rr, cc))
        if score > best_score + 1e-12:
            best_score = score
            best_idx = idx
    return best_idx


def _analyze_quadrant(region: np.ndarray) -> tuple[int, int, int]:
    beacons = np.argwhere(region == MAGENTA)
    assert len(beacons) == 1, "each quadrant must contain exactly one magenta beacon"
    main_color, strength = _dominant_color(region)
    main_pixels = np.argwhere(region == main_color)
    center_r, center_c = main_pixels.mean(axis=0)
    beacon_r, beacon_c = beacons[0]
    dir_idx = _direction_index(float(center_r - beacon_r), float(center_c - beacon_c))
    return main_color, strength, dir_idx


def _arrow_pixels(dir_idx: int) -> tuple[tuple[int, int], ...]:
    # The 3-cell arrow is the center plus the two ring cells adjacent to the implied tip.
    return (
        (1, 1),
        PERIMETER[(dir_idx - 1) % 8],
        PERIMETER[(dir_idx + 1) % 8],
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Re-rank four quadrant major colors by frequency, then draw directional 3x3 corner arrows.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep_color, sep_row, sep_col = _find_separator_cross(g)
    regions = _quadrant_slices(sep_row, sep_col, *g.shape)

    infos = [_analyze_quadrant(g[row_slice, col_slice]) for row_slice, col_slice in regions]
    strengths = [strength for _, strength, _ in infos]
    assert len(set(strengths)) == 4, "quadrant strengths must have distinct ranks"

    order = sorted(range(4), key=lambda idx: -infos[idx][1])

    out = np.full((OUTPUT_SIZE, OUTPUT_SIZE), BLACK, dtype=int)
    out[OUTPUT_CROSS, :] = sep_color
    out[:, OUTPUT_CROSS] = sep_color

    for block_idx, src_idx in enumerate(order):
        main_color, _strength, dir_idx = infos[src_idx]
        row_slice, col_slice = OUTPUT_BLOCKS[block_idx]
        out[row_slice, col_slice] = main_color
        for local_r, local_c in _arrow_pixels(dir_idx):
            out[row_slice.start + local_r, col_slice.start + local_c] = BLACK

    return out


def _make_main_blob(height: int, width: int, count: int, rng: np.random.Generator) -> set[tuple[int, int]] | None:
    options: list[tuple[int, int]] = []
    for blob_h in range(2, height + 1):
        for blob_w in range(2, width + 1):
            area = blob_h * blob_w
            if count <= area <= count + 3:
                options.append((blob_h, blob_w))
    if not options:
        return None

    blob_h, blob_w = options[int(rng.integers(0, len(options)))]
    top = int(rng.integers(0, height - blob_h + 1))
    left = int(rng.integers(0, width - blob_w + 1))
    cells = {(top + r, left + c) for r in range(blob_h) for c in range(blob_w)}

    excess = blob_h * blob_w - count
    corners = [
        (top, left),
        (top, left + blob_w - 1),
        (top + blob_h - 1, left),
        (top + blob_h - 1, left + blob_w - 1),
    ]
    if excess:
        for corner_idx in rng.choice(len(corners), size=excess, replace=False):
            cells.remove(corners[int(corner_idx)])
    assert len(cells) == count
    return cells


def _adjacent_noise_positions(
    height: int, width: int, main_cells: set[tuple[int, int]], beacon: tuple[int, int]
) -> list[tuple[int, int]]:
    neighbors = set()
    for r, c in main_cells:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if 0 <= rr < height and 0 <= cc < width and (rr, cc) not in main_cells and (rr, cc) != beacon:
                    neighbors.add((rr, cc))
    return sorted(neighbors)


def _pick_beacon(
    height: int,
    width: int,
    main_cells: set[tuple[int, int]],
    dir_idx: int,
    rng: np.random.Generator,
) -> tuple[int, int] | None:
    coords = np.array(sorted(main_cells), dtype=float)
    center_r, center_c = coords.mean(axis=0)
    candidates: list[tuple[float, tuple[int, int]]] = []
    for r in range(height):
        for c in range(width):
            if (r, c) in main_cells:
                continue
            dr = float(center_r - r)
            dc = float(center_c - c)
            if dr * dr + dc * dc < 4.0:
                continue
            if _direction_index(dr, dc) != dir_idx:
                continue
            candidates.append((dr * dr + dc * dc, (r, c)))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    top_k = candidates[: min(5, len(candidates))]
    return top_k[int(rng.integers(0, len(top_k)))][1]


def _build_quadrant(
    height: int,
    width: int,
    main_color: int,
    strength: int,
    dir_idx: int,
    rng: np.random.Generator,
) -> np.ndarray | None:
    for _ in range(LOCAL_RETRIES):
        main_cells = _make_main_blob(height, width, strength, rng)
        if main_cells is None:
            return None
        beacon = _pick_beacon(height, width, main_cells, dir_idx, rng)
        if beacon is None:
            continue

        noise_positions = _adjacent_noise_positions(height, width, main_cells, beacon)
        if not noise_positions:
            continue
        noise_count = int(rng.integers(1, min(NOISE_COUNT_MAX, len(noise_positions)) + 1))
        chosen_noise = rng.choice(len(noise_positions), size=noise_count, replace=False)
        noise_palette = [color for color in PALETTE if color != main_color]
        noise_colors = rng.choice(noise_palette, size=noise_count, replace=False)

        region = np.full((height, width), BLACK, dtype=int)
        for r, c in main_cells:
            region[r, c] = main_color
        region[beacon] = MAGENTA
        for idx, color in zip(chosen_noise, noise_colors):
            region[noise_positions[int(idx)]] = int(color)
        return region
    return None


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    top_h = int(rng.integers(GRID_SIDE_MIN, GRID_SIDE_MAX + 1))
    bottom_h = int(rng.integers(GRID_SIDE_MIN, GRID_SIDE_MAX + 1))
    left_w = int(rng.integers(GRID_SIDE_MIN, GRID_SIDE_MAX + 1))
    right_w = int(rng.integers(GRID_SIDE_MIN, GRID_SIDE_MAX + 1))

    height = top_h + 1 + bottom_h
    width = left_w + 1 + right_w
    sep_row = top_h
    sep_col = left_w

    areas = [
        top_h * left_w,
        top_h * right_w,
        bottom_h * left_w,
        bottom_h * right_w,
    ]
    max_strength = min(MAIN_COUNT_MAX, min(areas) - 3)
    if max_strength < MAIN_COUNT_MIN + 3:
        return None

    strength_pool = np.arange(MAIN_COUNT_MIN, max_strength + 1)
    strengths = rng.choice(strength_pool, size=4, replace=False).tolist()
    rng.shuffle(strengths)

    main_colors = rng.choice(PALETTE, size=4, replace=False).tolist()
    dir_indices = rng.integers(0, len(DIRECTIONS), size=4).tolist()

    grid = np.full((height, width), BLACK, dtype=int)
    grid[sep_row, :] = GREY
    grid[:, sep_col] = GREY

    quadrant_shapes = [
        (top_h, left_w),
        (top_h, right_w),
        (bottom_h, left_w),
        (bottom_h, right_w),
    ]
    quadrant_slices = _quadrant_slices(sep_row, sep_col, height, width)
    for idx, ((quad_h, quad_w), (row_slice, col_slice)) in enumerate(zip(quadrant_shapes, quadrant_slices)):
        quadrant = _build_quadrant(quad_h, quad_w, int(main_colors[idx]), int(strengths[idx]), int(dir_indices[idx]), rng)
        if quadrant is None:
            return None
        grid[row_slice, col_slice] = quadrant

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is None:
            continue
        try:
            _ = main(grid)
        except Exception:
            continue
        return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
