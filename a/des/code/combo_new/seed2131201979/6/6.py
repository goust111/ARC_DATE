from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BLUE = 1
AZURE = 8
OBJECT_COLORS = [0, 2, 3, 4, 5, 6, 7, 9]
RECT_DIMS = [(h, w) for h in range(1, 7) for w in range(1, 7) if 2 <= h * w <= 6]
PAIR_DIMS_BY_PARITY = {
    parity: [(a, b) for a in RECT_DIMS for b in RECT_DIMS if ((a[0] * a[1] + b[0] * b[1]) % 2) == parity]
    for parity in (0, 1)
}

MIN_PAIRS = 2
MAX_PAIRS = 4
GRID_MAX = 30
GEN_ATTEMPTS = 300
PAIR_PLACEMENT_ATTEMPTS = 500
MIN_BBOX_GAP = 1


def _checkerboard(h: int, w: int, even_color: int, odd_color: int) -> np.ndarray:
    rows, cols = np.indices((h, w))
    parity = (rows + cols) % 2
    return np.where(parity == 0, even_color, odd_color).astype(int)


def _find_background_colors(grid: np.ndarray) -> tuple[int, int]:
    colors = [int(c) for c in np.unique(grid)]
    rows, cols = np.indices(grid.shape)
    parity = (rows + cols) % 2
    best: tuple[int, int, int] | None = None
    best_pair: tuple[int, int] | None = None
    for i, a in enumerate(colors):
        for b in colors[i + 1 :]:
            match_ab = int(np.count_nonzero((parity == 0) & (grid == a))) + int(
                np.count_nonzero((parity == 1) & (grid == b))
            )
            match_ba = int(np.count_nonzero((parity == 0) & (grid == b))) + int(
                np.count_nonzero((parity == 1) & (grid == a))
            )
            if match_ab >= match_ba:
                ordered = (a, b)
                matches = match_ab
            else:
                ordered = (b, a)
                matches = match_ba
            total = int(np.count_nonzero((grid == a) | (grid == b)))
            cand = (matches, total, -abs(a - b))
            if best is None or cand > best:
                best = cand
                best_pair = ordered
    assert best_pair is not None, "expected at least two colors in the grid"
    return best_pair


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.asarray(coords, dtype=int))
    return components


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _rect_info(coords: np.ndarray) -> dict[str, object]:
    r0, c0, r1, c1 = _bbox(coords)
    area = (r1 - r0 + 1) * (c1 - c0 + 1)
    assert int(coords.shape[0]) == area, "each object must be a solid rectangle"
    cells = {(int(r), int(c)) for r, c in coords}
    return {
        "bbox": (r0, c0, r1, c1),
        "area": int(coords.shape[0]),
        "center": ((r0 + r1) // 2, (c0 + c1) // 2),
        "cells": cells,
    }


def _segment_cells(a: tuple[int, int], b: tuple[int, int]) -> set[tuple[int, int]]:
    (r0, c0), (r1, c1) = a, b
    if r0 == r1:
        lo, hi = sorted((c0, c1))
        return {(r0, c) for c in range(lo, hi + 1)}
    assert c0 == c1, "segments must be axis-aligned"
    lo, hi = sorted((r0, r1))
    return {(r, c0) for r in range(lo, hi + 1)}


def _visible_path(block_a: dict[str, object], block_b: dict[str, object], parity: int) -> set[tuple[int, int]]:
    upper, lower = sorted(
        (block_a, block_b),
        key=lambda item: (item["center"][0], item["center"][1]),  # type: ignore[index]
    )
    ur, uc = upper["center"]  # type: ignore[misc]
    lr, lc = lower["center"]  # type: ignore[misc]
    assert ur != lr and uc != lc, "paired centers must differ on both axes"
    corner = (ur, lc) if parity == 0 else (lr, uc)
    path = _segment_cells((ur, uc), corner) | _segment_cells(corner, (lr, lc))
    return path - upper["cells"] - lower["cells"]  # type: ignore[operator]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each same-colored pair of rectangles with a parity-controlled L-shaped line on the checkerboard.

    Input structure:
    - The background is a 1x1 checkerboard made from two alternating colors.
    - Several non-background colors each form exactly two solid rectangular blocks.
    - Within each same-colored pair, the two block centers are in different rows and different columns.

    Rule:
    - Detect the two checkerboard colors and keep them as the background palette.
    - For each non-background color, find its two rectangles, order them by center row from top to bottom, and compute the sum of their pixel areas.
    - If the total area is even, draw the L path that goes horizontally from the upper center and then vertically to the lower center; if odd, draw the L path that goes vertically first and then horizontally.
    - Paint only the background cells on that path with the pair color, leaving the existing rectangles unchanged.

    Key insight:
    - Color pairing is direct, but the two possible L-corners are disambiguated by the parity of the pair's combined area rather than by position alone.
    """
    g = np.asarray(grid, dtype=int)
    even_bg, odd_bg = _find_background_colors(g)
    bg_colors = {even_bg, odd_bg}

    out = g.copy()
    object_colors = [int(color) for color in np.unique(g) if int(color) not in bg_colors]
    for color in object_colors:
        components = _component_coords(g == color)
        assert len(components) == 2, "each non-background color must appear in exactly two blocks"
        blocks = [_rect_info(coords) for coords in components]
        parity = (blocks[0]["area"] + blocks[1]["area"]) % 2  # type: ignore[operator]
        for r, c in _visible_path(blocks[0], blocks[1], int(parity)):
            if int(out[r, c]) in bg_colors:
                out[r, c] = int(color)
    return out


def _rect_cells(r0: int, c0: int, h: int, w: int) -> set[tuple[int, int]]:
    return {(r, c) for r in range(r0, r0 + h) for c in range(c0, c0 + w)}


def _paint_cells(grid: np.ndarray, cells: Iterable[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = int(color)


def _disjoint_interval_gap(a0: int, a1: int, b0: int, b1: int) -> int | None:
    if a1 < b0:
        return b0 - a1 - 1
    if b1 < a0:
        return a0 - b1 - 1
    return None


def _sample_pair(num_pairs: int) -> list[int]:
    parities = [0, 1]
    while len(parities) < num_pairs:
        parities.append(random.randint(0, 1))
    random.shuffle(parities)
    return parities


def _place_pair(
    occupied: np.ndarray,
    reserved_paths: np.ndarray,
    color: int,
    target_parity: int,
) -> tuple[dict[str, object], dict[str, object], set[tuple[int, int]]] | None:
    h, w = occupied.shape
    for _ in range(PAIR_PLACEMENT_ATTEMPTS):
        (h1, w1), (h2, w2) = random.choice(PAIR_DIMS_BY_PARITY[target_parity])
        r1 = random.randint(0, h - h1)
        c1 = random.randint(0, w - w1)
        r2 = random.randint(0, h - h2)
        c2 = random.randint(0, w - w2)

        cells1 = _rect_cells(r1, c1, h1, w1)
        cells2 = _rect_cells(r2, c2, h2, w2)
        if cells1 & cells2:
            continue
        if any(occupied[r, c] or reserved_paths[r, c] for r, c in cells1 | cells2):
            continue

        gap_r = _disjoint_interval_gap(r1, r1 + h1 - 1, r2, r2 + h2 - 1)
        gap_c = _disjoint_interval_gap(c1, c1 + w1 - 1, c2, c2 + w2 - 1)
        if gap_r is None or gap_c is None:
            continue
        if gap_r < MIN_BBOX_GAP or gap_c < MIN_BBOX_GAP:
            continue

        block1 = {
            "bbox": (r1, c1, r1 + h1 - 1, c1 + w1 - 1),
            "area": h1 * w1,
            "center": ((2 * r1 + h1 - 1) // 2, (2 * c1 + w1 - 1) // 2),
            "cells": cells1,
            "color": color,
        }
        block2 = {
            "bbox": (r2, c2, r2 + h2 - 1, c2 + w2 - 1),
            "area": h2 * w2,
            "center": ((2 * r2 + h2 - 1) // 2, (2 * c2 + w2 - 1) // 2),
            "cells": cells2,
            "color": color,
        }
        if block1["center"][0] == block2["center"][0] or block1["center"][1] == block2["center"][1]:
            continue

        path_cells = _visible_path(block1, block2, target_parity)
        if not path_cells:
            continue
        if any(occupied[r, c] or reserved_paths[r, c] for r, c in path_cells):
            continue
        return block1, block2, path_cells
    return None


def _gen_attempt() -> np.ndarray | None:
    num_pairs = random.randint(MIN_PAIRS, MAX_PAIRS)
    min_dim = 12 + 2 * num_pairs
    h = random.randint(min_dim, GRID_MAX)
    w = random.randint(min_dim, GRID_MAX)
    even_bg, odd_bg = (BLUE, AZURE) if random.choice([False, True]) else (AZURE, BLUE)
    grid = _checkerboard(h, w, even_bg, odd_bg)

    occupied = np.zeros((h, w), dtype=bool)
    reserved_paths = np.zeros((h, w), dtype=bool)
    pairs: list[tuple[int, set[tuple[int, int]]]] = []
    colors = random.sample(OBJECT_COLORS, num_pairs)
    parities = _sample_pair(num_pairs)

    for color, parity in zip(colors, parities):
        placed = _place_pair(occupied, reserved_paths, color, parity)
        if placed is None:
            return None
        block1, block2, path_cells = placed
        for cells in (block1["cells"], block2["cells"]):
            for r, c in cells:  # type: ignore[assignment]
                occupied[r, c] = True
                grid[r, c] = color
        for r, c in path_cells:
            reserved_paths[r, c] = True
        pairs.append((color, path_cells))

    expected = grid.copy()
    for color, path_cells in pairs:
        _paint_cells(expected, path_cells, color)

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC input for puzzle 6")
