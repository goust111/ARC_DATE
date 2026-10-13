from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
PINK = 6
ORANGE = 7

LEGEND_PALETTE = (BLUE, GREEN, ORANGE, PINK)
DIAGONALS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

GRID_MIN = 12
GRID_MAX = 18
GEN_ATTEMPTS = 60
BRANCHES = ("k3_m2", "k4_m2", "k4_m3")
BRANCH_WEIGHTS = (0.38, 0.46, 0.16)
BRANCH_ATTEMPTS = {"k3_m2": 80, "k4_m2": 80, "k4_m3": 1600}


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _legend_length(grid: np.ndarray) -> int:
    row = grid[0]
    k = 0
    while k < row.size and int(row[k]) != BLACK:
        k += 1
    return k


def _parse_legend(grid: np.ndarray) -> list[int]:
    k = _legend_length(grid)
    assert k in (3, 4), "legend must be a 1x3 or 1x4 strip"
    legend = [int(x) for x in grid[0, :k]]
    assert len(set(legend)) == k, "legend colors must be distinct"
    assert set(legend).issubset(set(LEGEND_PALETTE)), "legend colors must use the allowed palette"
    return legend


def _find_markers(grid: np.ndarray, k: int) -> list[tuple[int, int]]:
    markers = [(int(r), int(c)) for r, c in np.argwhere(grid == RED)]
    assert 2 <= len(markers) <= 3, "expected 2 or 3 red markers"
    for r, c in markers:
        assert r > 0 and c >= k, "markers must lie inside the bottom-right main rectangle"
    return markers


def _markers_are_valid(markers: list[tuple[int, int]]) -> bool:
    for i, (r1, c1) in enumerate(markers):
        for r2, c2 in markers[i + 1 :]:
            if r1 == r2 or c1 == c2:
                return False
            if r1 - c1 == r2 - c2 or r1 + c1 == r2 + c2:
                return False
    return True


def _ray_mask(shape: tuple[int, int], k: int, markers: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r, c in markers:
        for dr, dc in DIAGONALS:
            rr, cc = r, c
            while _in_bounds(shape, rr, cc):
                if not (rr == 0 and cc < k):
                    mask[rr, cc] = True
                rr += dr
                cc += dc
    return mask


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _analyze_partition(shape: tuple[int, int], k: int, markers: list[tuple[int, int]]) -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
    rays = _ray_mask(shape, k, markers)
    main_open = ~rays[1:, k:]
    regions_local = _components(main_open)
    regions = [[(r + 1, c + k) for r, c in comp] for comp in regions_local]
    return rays, regions


def _area_signature(shape: tuple[int, int], k: int, markers: list[tuple[int, int]]) -> tuple[int, ...]:
    _, regions = _analyze_partition(shape, k, markers)
    return tuple(sorted(len(region) for region in regions))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend diagonal rays from red markers, rank the bottom-right black regions by area, and recolor them from the top-row legend.

    Input structure:
    - A black background contains a 1xK legend strip in the top-left corner, occupying row 0 and columns 0..K-1.
    - The usable main rectangle is the bottom-right area below that row and to the right of those legend columns.
    - Inside the main rectangle there are 2 or 3 red single-pixel markers, and no two markers share a row, column, or diagonal.

    Rule:
    - From each red marker, extend both diagonals all the way to the grid edges, painting those ray cells red while never overwriting the legend strip.
    - Restrict region counting to the bottom-right main rectangle only, and treat its remaining black cells as 4-connected regions separated by the red rays and the main-rectangle boundary.
    - Rank those regions by area from smallest to largest, then recolor the rank-1 region with the leftmost legend color, the rank-2 region with the next legend color, and so on.
    - Keep the legend, the red markers, and all red ray cells unchanged.

    Key insight:
    - The legend does not correspond to position; it is an area-ranking key, so the diagonal geometry must be built first and only then can the resulting regions be compared and recolored.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape

    legend = _parse_legend(g)
    k = len(legend)
    markers = _find_markers(g, k)
    assert _markers_are_valid(markers), "markers must not share a row, column, or diagonal"

    rays, regions = _analyze_partition((h, w), k, markers)
    assert len(regions) == k, "ray partition must produce exactly as many regions as legend cells"

    ranked = sorted(regions, key=len)
    areas = [len(region) for region in ranked]
    assert len(set(areas)) == k, "region areas must be pairwise distinct"

    out = g.copy()
    out[rays] = RED
    out[0, :k] = g[0, :k]
    for color, region in zip(legend, ranked):
        for r, c in region:
            out[r, c] = color
    return out


def _randint(lo: int, hi: int) -> int:
    return random.randint(lo, hi)


def _sample_pair_k3(h: int, w: int, k: int) -> list[tuple[int, int]]:
    return [
        (_randint(1, min(2, h - 1)), _randint(max(k, w - 3), w - 1)),
        (_randint(max(1, h - 3), h - 1), _randint(k, min(w - 1, k + 2))),
    ]


def _sample_pair_k4(h: int, w: int, k: int) -> list[tuple[int, int]]:
    return [
        (_randint(1, min(3, h - 1)), _randint(max(k, w - 4), w - 1)),
        (_randint(max(1, h // 2), h - 1), _randint(k, min(w - 1, k + 3))),
    ]


def _sample_triplet_k4(h: int, w: int, k: int) -> list[tuple[int, int]]:
    return [
        (_randint(1, min(2, h - 1)), _randint(max(k, w - 3), w - 1)),
        (_randint(max(1, h // 2), max(1, h - 2)), _randint(k, min(w - 1, k + 2))),
        (_randint(max(1, h - 3), h - 1), _randint(max(k, w // 2), max(k, w - 2))),
    ]


def _sample_markers(branch: str, h: int, w: int, k: int) -> list[tuple[int, int]]:
    if branch == "k3_m2":
        return _sample_pair_k3(h, w, k)
    if branch == "k4_m2":
        return _sample_pair_k4(h, w, k)
    return _sample_triplet_k4(h, w, k)


def _build_input(h: int, w: int, legend: list[int], markers: list[tuple[int, int]]) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    grid[0, : len(legend)] = np.array(legend, dtype=int)
    for r, c in markers:
        grid[r, c] = RED
    return grid


def _markers_are_essential(shape: tuple[int, int], k: int, markers: list[tuple[int, int]]) -> bool:
    base = _area_signature(shape, k, markers)
    for i in range(len(markers)):
        reduced = markers[:i] + markers[i + 1 :]
        if _area_signature(shape, k, reduced) == base:
            return False
    return True


def _output_is_semantic(inp: np.ndarray) -> bool:
    h, w = inp.shape
    legend = _parse_legend(inp)
    k = len(legend)
    markers = _find_markers(inp, k)
    if not _markers_are_valid(markers):
        return False

    rays, regions = _analyze_partition((h, w), k, markers)
    if len(regions) != k:
        return False

    ranked = sorted(regions, key=len)
    areas = [len(region) for region in ranked]
    if len(set(areas)) != k:
        return False
    if min(areas) < 2:
        return False
    if max(areas) > ((h - 1) * (w - k) * 4) // 5:
        return False
    if not _markers_are_essential((h, w), k, markers):
        return False

    out = main(inp)
    if not np.array_equal(out[0, :k], inp[0, :k]):
        return False
    if np.any(out[rays] != RED):
        return False

    used_colors = set()
    for color, region in zip(legend, ranked):
        used_colors.add(color)
        if any(int(out[r, c]) != color for r, c in region):
            return False
    if used_colors != set(legend):
        return False

    untouched = np.ones_like(inp, dtype=bool)
    untouched[0, :k] = False
    untouched[rays] = False
    for region in ranked:
        for r, c in region:
            untouched[r, c] = False
    if np.any(out[untouched] != inp[untouched]):
        return False
    if np.array_equal(out, inp):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        branch = random.choices(BRANCHES, weights=BRANCH_WEIGHTS, k=1)[0]
        for _inner in range(BRANCH_ATTEMPTS[branch]):
            h = _randint(GRID_MIN, GRID_MAX)
            w = _randint(GRID_MIN, GRID_MAX)
            k = 3 if branch == "k3_m2" else 4
            legend = random.sample(list(LEGEND_PALETTE), k)
            markers = _sample_markers(branch, h, w, k)
            if not _markers_are_valid(markers):
                continue

            grid = _build_input(h, w, legend, markers)
            if _output_is_semantic(grid):
                return grid
    raise RuntimeError("failed to generate a valid diagonal-ray ranking puzzle")
