from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
RED = 2
GREY = 5
SEED_COLORS = (1, 3, 4, 6, 7, 8, 9)

GENERATION_ATTEMPTS = 300
LAYOUT_ATTEMPTS = 120
MIN_REGION_AREA = 6
MAX_CANDIDATES_PER_REGION = 20
LAYOUT_MODES = ("no_bounce", "one_bounce", "multi_bounce", "right_edge")
LAYOUT_WEIGHTS = (2, 4, 4, 2)

Coord = tuple[int, int]


def _trace_bouncing_path(height: int, width: int, start_col: int) -> tuple[list[Coord], int]:
    height = int(height)
    width = int(width)
    start_col = int(start_col)
    assert height >= 2 and width >= 2
    assert 0 <= start_col < width

    row = height - 1
    col = start_col
    dc = 1
    bounces = 0
    path = [(row, col)]

    while row > 0:
        next_col = col + dc
        if not (0 <= next_col < width):
            dc = -dc
            bounces += 1
            next_col = col + dc
        row -= 1
        col = next_col
        path.append((row, col))

    return path, bounces


def _path_mask(shape: tuple[int, int], start_col: int) -> tuple[np.ndarray, list[Coord], int]:
    height, width = map(int, shape)
    path, bounces = _trace_bouncing_path(height, width, start_col)
    mask = np.zeros((height, width), dtype=bool)
    for row, col in path:
        mask[row, col] = True
    return mask, path, bounces


def _components(open_mask: np.ndarray) -> list[np.ndarray]:
    height, width = map(int, open_mask.shape)
    seen = np.zeros_like(open_mask, dtype=bool)
    comps: list[np.ndarray] = []

    for row in range(height):
        for col in range(width):
            if not open_mask[row, col] or seen[row, col]:
                continue

            queue: deque[Coord] = deque([(row, col)])
            seen[row, col] = True
            coords: list[Coord] = []

            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < height and 0 <= nc < width and open_mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))

            comps.append(np.array(coords, dtype=int))

    return comps


def _distance_to_path(row: int, col: int, path: list[Coord]) -> int:
    return min(abs(row - pr) + abs(col - pc) for pr, pc in path)


def _region_candidates(
    region: np.ndarray,
    open_mask: np.ndarray,
    path: list[Coord],
) -> list[Coord]:
    height, width = map(int, open_mask.shape)
    ranked: list[tuple[tuple[int, int, int], Coord]] = []

    for row_raw, col_raw in region:
        row = int(row_raw)
        col = int(col_raw)
        clearance = _distance_to_path(row, col, path)
        degree = 0
        for nr, nc in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
            if 0 <= nr < height and 0 <= nc < width and open_mask[nr, nc]:
                degree += 1
        edge_margin = min(row, height - 1 - row, col, width - 1 - col)
        ranked.append(((clearance, degree, edge_margin), (row, col)))

    random.shuffle(ranked)
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [coord for _score, coord in ranked[: min(MAX_CANDIDATES_PER_REGION, len(ranked))]]


def _non_adjacent(a: Coord, b: Coord) -> bool:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1])) > 1


def _choose_seed_positions(
    regions: list[np.ndarray],
    open_mask: np.ndarray,
    path: list[Coord],
) -> list[Coord] | None:
    candidates = [_region_candidates(region, open_mask, path) for region in regions]
    order = sorted(range(len(regions)), key=lambda idx: len(candidates[idx]))
    chosen: list[Coord | None] = [None] * len(regions)

    def backtrack(depth: int) -> bool:
        if depth == len(order):
            return True
        idx = order[depth]
        for coord in candidates[idx]:
            if all(prev is None or _non_adjacent(coord, prev) for prev in chosen):
                chosen[idx] = coord
                if backtrack(depth + 1):
                    return True
        chosen[idx] = None
        return False

    if not backtrack(0):
        return None
    return [(coord[0], coord[1]) for coord in chosen if coord is not None]


def _build_expected(
    shape: tuple[int, int],
    path_mask: np.ndarray,
    regions: list[np.ndarray],
    colors: list[int],
) -> np.ndarray:
    out = np.full(shape, BLACK, dtype=int)
    for region, color in zip(regions, colors):
        out[region[:, 0], region[:, 1]] = int(color)
    out[path_mask] = GREY
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Project the hidden bouncing ray, then color each partition by its unique seed.

    Input structure:
    - A black background contains exactly one red start pixel on the bottom row.
    - The other non-black pixels are isolated colored seeds, with no pre-drawn walls.
    - The left and right canvas borders act as invisible reflection walls.

    Rule:
    - Starting from the red pixel, trace a one-cell-wide diagonal ray upward-right.
    - Whenever the next step would leave the grid horizontally, reverse only the horizontal direction; stop after painting the top-row cell.
    - Treat the traced grey path as an impassable wall, split all remaining cells into 4-connected regions, and recolor each whole region with the color of its unique seed.

    Key insight:
    - The seeds do not interact directly; the hidden reflected ray must be reconstructed first because it creates the unseen territorial boundaries.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height >= 2 and width >= 2, "grid must be at least 2x2"

    red_pixels = np.argwhere(g == RED)
    assert red_pixels.shape == (1, 2), "expected exactly one red start pixel"
    start_row, start_col = map(int, red_pixels[0])
    assert start_row == height - 1, "red start pixel must be on the bottom row"
    assert not np.any(g == GREY), "input should not already contain the wall color"

    path_mask, _path, _bounces = _path_mask(g.shape, start_col)
    seed_mask = (g != BLACK) & (g != RED) & (g != GREY)
    assert not np.any(path_mask & seed_mask), "seed pixels cannot lie on the hidden path"

    regions = _components(~path_mask)
    out = np.full((height, width), BLACK, dtype=int)
    seen_seed_count = 0

    for region in regions:
        region_seeds = region[seed_mask[region[:, 0], region[:, 1]]]
        assert len(region_seeds) == 1, "each region must contain exactly one seed"
        row, col = map(int, region_seeds[0])
        color = int(g[row, col])
        out[region[:, 0], region[:, 1]] = color
        seen_seed_count += 1

    assert seen_seed_count == int(seed_mask.sum()), "all seeds must belong to exactly one region"
    out[path_mask] = GREY
    return out


def _sample_layout(mode: str) -> tuple[int, int, int, np.ndarray, list[np.ndarray]] | None:
    for _ in range(LAYOUT_ATTEMPTS):
        if mode == "no_bounce":
            height = random.randint(8, 15)
            width = random.randint(height, min(22, height + 8))
            start_max = width - height
            if start_max < 0:
                continue
            start_col = random.randint(0, start_max)
        elif mode == "one_bounce":
            width = random.randint(5, 10)
            height = random.randint(width + 2, min(22, 2 * width + 4))
            start_col = random.randint(0, width - 1)
        elif mode == "multi_bounce":
            width = random.randint(4, 8)
            height = random.randint(max(12, 2 * width + 1), 24)
            start_col = random.randint(0, width - 1)
        else:
            width = random.randint(5, 10)
            height = random.randint(8, 22)
            start_col = width - 1

        path_mask, path, bounces = _path_mask((height, width), start_col)
        regions = _components(~path_mask)

        if mode == "no_bounce" and bounces != 0:
            continue
        if mode == "one_bounce" and bounces != 1:
            continue
        if mode == "multi_bounce" and bounces < 2:
            continue
        if mode == "right_edge" and (start_col != width - 1 or bounces < 1):
            continue

        if not (2 <= len(regions) <= len(SEED_COLORS)):
            continue
        if any(len(region) < MIN_REGION_AREA for region in regions):
            continue
        if max(len(region) for region in regions) > (height * width) - height - MIN_REGION_AREA:
            continue

        return height, width, start_col, path_mask, regions

    return None


def _gen_attempt() -> np.ndarray | None:
    mode = random.choices(LAYOUT_MODES, weights=LAYOUT_WEIGHTS, k=1)[0]
    sampled = _sample_layout(mode)
    if sampled is None:
        return None

    height, width, start_col, path_mask, regions = sampled
    open_mask = ~path_mask
    path = [(int(row), int(col)) for row, col in np.argwhere(path_mask)]
    seed_positions = _choose_seed_positions(regions, open_mask, path)
    if seed_positions is None or len(seed_positions) != len(regions):
        return None

    colors = random.sample(SEED_COLORS, len(regions))
    inp = np.full((height, width), BLACK, dtype=int)
    inp[height - 1, start_col] = RED

    for (row, col), color in zip(seed_positions, colors):
        inp[row, col] = int(color)

    expected = _build_expected(inp.shape, path_mask, regions, colors)
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid invisible-bouncing-wall input")
