from __future__ import annotations

from collections import Counter

import numpy as np


BACKGROUND = 0
BLUE = 1
RED = 2
GREY = 5
TEAL = 8

Coord = tuple[int, int]
BBox = tuple[int, int, int, int]

BLOCK_COUNT_RANGE = (4, 6)
HEIGHT_RANGE = (15, 24)
WIDTH_MAX = 26
MIN_GAP_BETWEEN_LEFT_COLS = 3
MAX_GENERATION_ATTEMPTS = 500

_RNG = np.random.default_rng()


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    """Minimal 4-connected component finder, matching arc_utils' (r, c) convention."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []

    for start_r, start_c in np.argwhere(mask):
        start_r = int(start_r)
        start_c = int(start_c)
        if seen[start_r, start_c]:
            continue

        stack = [(start_r, start_c)]
        seen[start_r, start_c] = True
        cells: list[Coord] = []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    stack.append((rr, cc))
        components.append(np.array(cells, dtype=int))

    return components


def _find_blue_blocks(grid: np.ndarray) -> list[dict]:
    blocks: list[dict] = []
    for coords in _components_mask(grid == BLUE):
        bbox = _bbox_coords(coords)
        blocks.append(
            {
                "coords": coords,
                "bbox": bbox,
                "area": int(coords.shape[0]),
            }
        )
    return blocks


def _height_rank_order(blocks: list[dict], marker_row: int) -> list[dict]:
    return sorted(blocks, key=lambda block: marker_row - block["bbox"][2], reverse=True)


def main(grid: np.ndarray) -> np.ndarray:
    """Drop blue 2x2 squares onto a grey baseline, then recolor them by original height rank.

    Input structure:
    - A black background contains one full-width horizontal grey marker row near the bottom.
    - Several disjoint blue 2x2 square objects lie strictly above that row in distinct columns.
    - The squares' original vertical distances to the marker row are all different.

    Rule:
    - Keep the grey marker row unchanged.
    - For each blue square, record its original bottom row and compute its vertical distance to the grey row.
    - Remove the original blue squares, then move every square so it sits immediately above the grey row while keeping its column unchanged.
    - Sort the moved squares by original distance from farthest to nearest; recolor even ranks teal and odd ranks red.

    Key insight:
    - The downward alignment erases the original height information geometrically, so the final teal/red alternation must be determined from the input heights rather than from the aligned output positions.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _get_bg(g)
    assert bg == BACKGROUND, "expected black background"
    assert set(map(int, np.unique(g))) <= {BACKGROUND, BLUE, GREY}, "unexpected colors"

    grey_rows = [r for r in range(g.shape[0]) if np.all(g[r, :] == GREY)]
    assert len(grey_rows) == 1, "expected a unique full-width grey marker row"
    marker_row = int(grey_rows[0])
    assert marker_row >= 2, "marker row must leave room for aligned 2x2 blocks"
    assert int((g == GREY).sum()) == g.shape[1], "grey may only appear on the marker row"

    blocks = _find_blue_blocks(g)
    assert blocks, "expected at least one blue block"

    left_cols: list[int] = []
    distances: list[int] = []
    for block in blocks:
        r0, c0, r1, c1 = block["bbox"]
        assert block["area"] == 4, "each blue object must have area 4"
        assert r1 == r0 + 1 and c1 == c0 + 1, "each blue object must be a 2x2 square"
        assert r1 < marker_row, "blue blocks must lie above the marker row"
        left_cols.append(c0)
        distances.append(marker_row - r1)

    assert len(set(left_cols)) == len(left_cols), "each block must occupy a distinct column band"
    assert len(set(distances)) == len(distances), "original heights must be unique"
    sorted_cols = sorted(left_cols)
    assert all(
        sorted_cols[i + 1] - sorted_cols[i] >= MIN_GAP_BETWEEN_LEFT_COLS
        for i in range(len(sorted_cols) - 1)
    ), "blocks must leave at least one empty column between them"

    out = g.copy()
    out[g == BLUE] = BACKGROUND
    aligned_top = marker_row - 2

    for rank, block in enumerate(_height_rank_order(blocks, marker_row)):
        color = TEAL if rank % 2 == 0 else RED
        left = int(block["bbox"][1])
        out[aligned_top : aligned_top + 2, left : left + 2] = color

    return out


def _sample_left_cols(width: int, n_blocks: int) -> list[int] | None:
    candidates = np.arange(width - 1)
    for _ in range(200):
        chosen: list[int] = []
        for col in _RNG.permutation(candidates):
            col = int(col)
            if all(abs(col - prev) >= MIN_GAP_BETWEEN_LEFT_COLS for prev in chosen):
                chosen.append(col)
                if len(chosen) == n_blocks:
                    return sorted(chosen)
    return None


def _sample_layout() -> tuple[int, int, int, list[int], list[int]] | None:
    n_blocks = int(_RNG.integers(BLOCK_COUNT_RANGE[0], BLOCK_COUNT_RANGE[1] + 1))

    min_height = max(HEIGHT_RANGE[0], n_blocks + 7)
    if min_height > HEIGHT_RANGE[1]:
        return None
    height = int(_RNG.integers(min_height, HEIGHT_RANGE[1] + 1))

    min_width = max(14, 3 * n_blocks + 2)
    width = int(_RNG.integers(min_width, WIDTH_MAX + 1))

    marker_low = max(n_blocks + 3, height // 2)
    marker_high = height - 3
    if marker_low > marker_high:
        return None
    marker_row = int(_RNG.integers(marker_low, marker_high + 1))

    left_cols = _sample_left_cols(width, n_blocks)
    if left_cols is None:
        return None

    possible_distances = np.arange(2, marker_row)
    if possible_distances.size < n_blocks:
        return None
    distances = [int(v) for v in _RNG.choice(possible_distances, size=n_blocks, replace=False)]
    return height, width, marker_row, left_cols, distances


def _build_input(height: int, width: int, marker_row: int, left_cols: list[int], distances: list[int]) -> np.ndarray:
    grid = np.full((height, width), BACKGROUND, dtype=int)
    grid[marker_row, :] = GREY
    for left, distance in zip(left_cols, distances):
        top = marker_row - distance - 1
        grid[top : top + 2, left : left + 2] = BLUE
    return grid


def _expected_output(
    height: int,
    width: int,
    marker_row: int,
    left_cols: list[int],
    distances: list[int],
) -> np.ndarray:
    out = np.full((height, width), BACKGROUND, dtype=int)
    out[marker_row, :] = GREY
    aligned_top = marker_row - 2
    order = np.argsort(np.asarray(distances))[::-1]
    for rank, idx in enumerate(order):
        color = TEAL if rank % 2 == 0 else RED
        left = int(left_cols[int(idx)])
        out[aligned_top : aligned_top + 2, left : left + 2] = color
    return out


def _height_parity_is_not_column_parity(left_cols: list[int], distances: list[int]) -> bool:
    order = list(map(int, np.argsort(np.asarray(distances))[::-1]))
    height_teal = tuple(sorted(left_cols[idx] for rank, idx in enumerate(order) if rank % 2 == 0))

    left_to_right = tuple(sorted(col for rank, col in enumerate(sorted(left_cols)) if rank % 2 == 0))
    right_to_left = tuple(
        sorted(col for rank, col in enumerate(sorted(left_cols, reverse=True)) if rank % 2 == 0)
    )
    return height_teal not in {left_to_right, right_to_left}


def _gen_attempt() -> np.ndarray | None:
    layout = _sample_layout()
    if layout is None:
        return None

    height, width, marker_row, left_cols, distances = layout
    if not _height_parity_is_not_column_parity(left_cols, distances):
        return None

    inp = _build_input(height, width, marker_row, left_cols, distances)
    expected = _expected_output(height, width, marker_row, left_cols, distances)

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
