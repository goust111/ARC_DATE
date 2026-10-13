from __future__ import annotations

import itertools
from collections import Counter

import numpy as np


_TOP_LEFT = 0
_TOP_RIGHT = 1
_BOTTOM_LEFT = 2
_BOTTOM_RIGHT = 3

_PATTERN_TO_RANKS = {
    "all4": (0, 1, 2, 3),
    "123": (1, 2, 3),
    "23": (2, 3),
    "0": (0,),
    "1": (1,),
    "2": (2,),
    "3": (3,),
}

_MAX_GEN_ATTEMPTS = 400
_MAX_CLUSTER_ATTEMPTS = 120
_MAX_MASK_ATTEMPTS = 200


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _find_separator_cross(grid: np.ndarray, bg: int) -> tuple[int, int, int]:
    h, w = map(int, grid.shape)
    assert h >= 5 and w >= 5, "grid must be large enough for four non-empty quadrants"

    row_candidates = []
    for r in range(h):
        color = int(grid[r, 0])
        if color == bg:
            continue
        if np.all(grid[r, :] == color):
            row_candidates.append((r, color))

    col_candidates = []
    for c in range(w):
        color = int(grid[0, c])
        if color == bg:
            continue
        if np.all(grid[:, c] == color):
            col_candidates.append((c, color))

    valid = []
    for r, row_color in row_candidates:
        if r == 0 or r == h - 1 or 2 * r + 1 != h:
            continue
        for c, col_color in col_candidates:
            if c == 0 or c == w - 1 or 2 * c + 1 != w:
                continue
            if row_color == col_color:
                valid.append((r, c, row_color))

    assert len(valid) == 1, "expected exactly one centered separator cross"
    return valid[0]


def _extract_quadrants(grid: np.ndarray, split_r: int, split_c: int) -> list[np.ndarray]:
    return [
        grid[:split_r, :split_c].copy(),
        grid[:split_r, split_c + 1 :].copy(),
        grid[split_r + 1 :, :split_c].copy(),
        grid[split_r + 1 :, split_c + 1 :].copy(),
    ]


def _quadrant_color_and_area(quad: np.ndarray, bg: int) -> tuple[int, int]:
    colors = [int(color) for color in np.unique(quad) if int(color) != bg]
    assert len(colors) == 1, "each quadrant must contain exactly one foreground color"
    color = colors[0]
    area = int(np.count_nonzero(quad == color))
    assert area > 0, "foreground mask must be non-empty"
    return color, area


def _overlay_in_order(quadrants: list[np.ndarray], order: tuple[int, ...] | list[int], bg: int) -> np.ndarray:
    out = np.full(quadrants[0].shape, bg, dtype=int)
    for idx in reversed(tuple(int(i) for i in order)):
        patch = quadrants[idx]
        mask = patch != bg
        out[mask] = patch[mask]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Overlay the four quadrant masks into one tile, with sparser quadrants floating above denser ones.

    Input structure:
    - One solid separator row and one solid separator column of the same non-background color split the canvas into four equal quadrants.
    - Each quadrant is a background tile containing a binary mask in one unique foreground color.
    - The four foreground colors are all different, and their foreground pixel counts are pairwise different.

    Rule:
    - Count the foreground pixels in each quadrant.
    - Sort the quadrants from fewest foreground pixels to most; fewer pixels means a higher layer.
    - Remove the separator cross and overlay the four same-sized quadrant tiles cell by cell, taking the first non-background color seen from top layer to bottom layer; if every layer is background at a cell, keep the background.

    Key insight:
    - The decisive order is not geometric position but relative sparsity, so overlap conflicts must be resolved by counting each quadrant mask before stacking them.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    split_r, split_c, _ = _find_separator_cross(g, bg)
    quadrants = _extract_quadrants(g, split_r, split_c)
    shape = quadrants[0].shape
    assert all(quad.shape == shape for quad in quadrants), "quadrants must be equal-sized"

    infos = [_quadrant_color_and_area(quad, bg) for quad in quadrants]
    colors = [color for color, _ in infos]
    counts = [count for _, count in infos]
    assert len(set(colors)) == 4, "quadrant foreground colors must be pairwise distinct"
    assert len(set(counts)) == 4, "quadrant foreground counts must be pairwise distinct"

    order = tuple(sorted(range(4), key=lambda idx: counts[idx]))
    return _overlay_in_order(quadrants, order, bg)


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _grow_cluster(
    h: int, w: int, size: int, occupied: np.ndarray, rng: np.random.Generator
) -> set[tuple[int, int]] | None:
    free_cells = [(r, c) for r in range(h) for c in range(w) if not occupied[r, c]]
    if len(free_cells) < size:
        return None

    for _ in range(_MAX_CLUSTER_ATTEMPTS):
        seed = free_cells[int(rng.integers(len(free_cells)))]
        cluster = {seed}
        while len(cluster) < size:
            frontier = set()
            for r, c in cluster:
                for nr, nc in _neighbors4(r, c, h, w):
                    if occupied[nr, nc] or (nr, nc) in cluster:
                        continue
                    frontier.add((nr, nc))
            if not frontier:
                break
            frontier = list(frontier)
            cluster.add(frontier[int(rng.integers(len(frontier)))])
        if len(cluster) == size:
            return cluster
    return None


def _place_pattern_clusters(
    h: int, w: int, pattern_sizes: dict[str, int], rng: np.random.Generator
) -> dict[str, set[tuple[int, int]]] | None:
    occupied = np.zeros((h, w), dtype=bool)
    clusters: dict[str, set[tuple[int, int]]] = {}

    for name, size in sorted(pattern_sizes.items(), key=lambda item: (-item[1], item[0])):
        cluster = _grow_cluster(h, w, size, occupied, rng)
        if cluster is None:
            return None
        clusters[name] = cluster
        for r, c in cluster:
            occupied[r, c] = True
    return clusters


def _make_rank_masks(h: int, w: int, rng: np.random.Generator) -> list[np.ndarray] | None:
    cell_count = h * w
    bg_reserve = max(2, cell_count // 8)

    for _ in range(_MAX_MASK_ATTEMPTS):
        max_small = max(1, min(3, cell_count // 16 + 1))
        size_all4 = int(rng.integers(1, max_small + 1))
        size_123 = int(rng.integers(1, max_small + 1))
        size_23 = int(rng.integers(1, max_small + 1))

        min_a0 = size_all4 + 1
        min_a1 = size_all4 + size_123 + 1
        min_a2 = size_all4 + size_123 + size_23 + 1
        min_a3 = min_a2 + 1

        max_area = min(cell_count - 1, max(min_a3 + 3, int(round(cell_count * 0.42))))
        if max_area < min_a3:
            continue

        max_a0 = max_area - 3
        if max_a0 < min_a0:
            continue
        a0 = int(rng.integers(min_a0, max_a0 + 1))

        min_a1_eff = max(min_a1, a0 + 1)
        max_a1 = max_area - 2
        if max_a1 < min_a1_eff:
            continue
        a1 = int(rng.integers(min_a1_eff, max_a1 + 1))

        min_a2_eff = max(min_a2, a1 + 1)
        max_a2 = max_area - 1
        if max_a2 < min_a2_eff:
            continue
        a2 = int(rng.integers(min_a2_eff, max_a2 + 1))

        min_a3_eff = max(min_a3, a2 + 1)
        if max_area < min_a3_eff:
            continue
        a3 = int(rng.integers(min_a3_eff, max_area + 1))

        pattern_sizes = {
            "all4": size_all4,
            "123": size_123,
            "23": size_23,
            "0": a0 - size_all4,
            "1": a1 - size_all4 - size_123,
            "2": a2 - size_all4 - size_123 - size_23,
            "3": a3 - size_all4 - size_123 - size_23,
        }
        if min(pattern_sizes.values()) < 1:
            continue

        union_area = sum(pattern_sizes.values())
        if union_area > cell_count - bg_reserve:
            continue

        clusters = _place_pattern_clusters(h, w, pattern_sizes, rng)
        if clusters is None:
            continue

        masks = [np.zeros((h, w), dtype=bool) for _ in range(4)]
        for name, ranks in _PATTERN_TO_RANKS.items():
            for r, c in clusters[name]:
                for rank in ranks:
                    masks[rank][r, c] = True

        areas = [int(mask.sum()) for mask in masks]
        if not (areas[0] < areas[1] < areas[2] < areas[3]):
            continue

        cover = np.zeros((h, w), dtype=int)
        for mask in masks:
            cover += mask.astype(int)
        if int(cover.max()) < 3 or int(np.count_nonzero(cover == 0)) < bg_reserve:
            continue

        return masks
    return None


def _compose_grid(
    rank_masks: list[np.ndarray],
    rank_to_quadrant: tuple[int, int, int, int],
    bg: int,
    sep: int,
    quadrant_colors: list[int],
) -> np.ndarray:
    qh, qw = rank_masks[0].shape
    quadrants = [np.full((qh, qw), bg, dtype=int) for _ in range(4)]

    for rank, quadrant_idx in enumerate(rank_to_quadrant):
        color = int(quadrant_colors[quadrant_idx])
        quadrants[quadrant_idx][rank_masks[rank]] = color

    grid = np.full((2 * qh + 1, 2 * qw + 1), bg, dtype=int)
    grid[qh, :] = sep
    grid[:, qw] = sep
    grid[:qh, :qw] = quadrants[_TOP_LEFT]
    grid[:qh, qw + 1 :] = quadrants[_TOP_RIGHT]
    grid[qh + 1 :, :qw] = quadrants[_BOTTOM_LEFT]
    grid[qh + 1 :, qw + 1 :] = quadrants[_BOTTOM_RIGHT]
    return grid


def _expected_overlay(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, ...], list[np.ndarray], int]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    split_r, split_c, _ = _find_separator_cross(g, bg)
    quadrants = _extract_quadrants(g, split_r, split_c)
    counts = [_quadrant_color_and_area(quad, bg)[1] for quad in quadrants]
    order = tuple(sorted(range(4), key=lambda idx: counts[idx]))
    return _overlay_in_order(quadrants, order, bg), order, quadrants, bg


def _validate_generated(grid: np.ndarray) -> None:
    out, order, quadrants, bg = _expected_overlay(grid)
    actual = main(grid)
    assert np.array_equal(actual, out), "main() disagrees with the generated semantics"

    counts = []
    colors = []
    for quad in quadrants:
        color, area = _quadrant_color_and_area(quad, bg)
        colors.append(color)
        counts.append(area)

    assert len(set(colors)) == 4
    assert len(set(counts)) == 4
    assert int(np.count_nonzero(out == bg)) > 0, "output should retain some background"
    assert set(map(int, np.unique(out))) == {bg, *colors}, "every layer should contribute visibly"
    assert all(not np.array_equal(out, quad) for quad in quadrants), "overlay must not collapse to one quadrant"

    cover = np.zeros(quadrants[0].shape, dtype=int)
    for quad in quadrants:
        cover += (quad != bg).astype(int)
    assert int(cover.max()) >= 3, "need genuine overlap to force layer priority"
    assert np.any(cover == 0), "some cells should stay background after overlay"

    matching_orders = []
    for perm in itertools.permutations(range(4)):
        candidate = _overlay_in_order(quadrants, perm, bg)
        if np.array_equal(candidate, out):
            matching_orders.append(perm)
    assert matching_orders == [order], "the visible masks should determine a unique overlay order"


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    qh = int(rng.integers(5, 15))
    qw = int(rng.integers(5, 15))

    rank_masks = _make_rank_masks(qh, qw, rng)
    if rank_masks is None:
        return None

    palette = [int(color) for color in rng.choice(np.arange(10), size=6, replace=False)]
    bg, sep = palette[:2]
    quadrant_colors = palette[2:]
    rank_to_quadrant = tuple(int(x) for x in rng.permutation(4))

    grid = _compose_grid(rank_masks, rank_to_quadrant, bg, sep, quadrant_colors)
    _validate_generated(grid)
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(_MAX_GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid layered-overlay ARC input")
