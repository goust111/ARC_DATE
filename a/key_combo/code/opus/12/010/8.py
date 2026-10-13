from __future__ import annotations

import itertools
import math
import random

import numpy as np


BG = 0
GREY = 5
LEGEND_PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _triangular_columns(limit: int) -> list[int]:
    cols: list[int] = []
    col = 0
    step = 1
    while col < limit:
        cols.append(col)
        col += step
        step += 1
    return cols


def _is_triangular(n: int) -> bool:
    if n < 0:
        return False
    disc = 8 * n + 1
    root = math.isqrt(disc)
    return root * root == disc and (root - 1) % 2 == 0


def _noise_filtered(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    h, w = g.shape
    for r, c in np.argwhere(g == GREY):
        rr = int(r)
        cc = int(c)
        degree = 0
        for nr, nc in _neighbors4(rr, cc):
            if 0 <= nr < h and 0 <= nc < w and int(g[nr, nc]) == GREY:
                degree += 1
        if degree < 2:
            out[rr, cc] = BG
    return out


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
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
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _legend_colors(grid: np.ndarray) -> list[int]:
    top = np.asarray(grid[0], dtype=int)
    cols = np.flatnonzero(top != BG)
    assert len(cols) >= 1
    assert np.array_equal(cols, np.arange(cols[0], cols[-1] + 1))
    return [int(top[c]) for c in cols]


def main(grid: np.ndarray) -> np.ndarray:
    """Filter grey noise, keep only sprites whose left column is triangular, then recolor them by the cyclic top-row legend.

    Input structure:
    - The grid has black background, a compact contiguous legend strip on the top row, several grey 4-connected sprites in a middle two-row band, and extra grey singleton noise pixels elsewhere.
    - All sprites share one small shape, but only some of their bounding-box left columns lie on triangular numbers 0, 1, 3, 6, 10, 15, ...

    Rule:
    - First remove every grey pixel with fewer than two grey 4-neighbors.
    - Treat the remaining grey pixels as complete 4-connected sprites; erase every sprite whose bounding-box left column is not triangular.
    - Sort the surviving triangular-position sprites by that left column and recolor them cyclically with the top-row legend colors from left to right.
    - Leave the legend and background unchanged.

    Key insight:
    - The target sprites are isolated by two independent filters: local grey support removes singleton noise, then triangular left-column positions remove same-shaped distractor sprites before legend-driven cyclic recoloring.
    """
    g = np.asarray(grid, dtype=int)
    filtered = _noise_filtered(g)
    legend = _legend_colors(g)

    grey_mask = filtered == GREY
    comps = _find_components(grey_mask)
    kept = []
    out = filtered.copy()

    for coords in comps:
        left = int(coords[:, 1].min())
        if _is_triangular(left):
            kept.append((left, coords))
        else:
            out[coords[:, 0], coords[:, 1]] = BG

    kept.sort(key=lambda item: item[0])
    for rank, (_left, coords) in enumerate(kept):
        color = int(legend[rank % len(legend)])
        out[coords[:, 0], coords[:, 1]] = color

    return out


def _place_sprite(grid: np.ndarray, top: int, left: int) -> None:
    grid[top : top + 2, left : left + 2] = GREY


def _valid_left_subsets(candidates: list[int], size: int) -> list[tuple[int, ...]]:
    good = []
    for subset in itertools.combinations(candidates, size):
        if all(b - a >= 3 for a, b in zip(subset, subset[1:])):
            good.append(subset)
    return good


def _free_noise_cells(
    h: int,
    w: int,
    legend_len: int,
    sprite_cells: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    blocked = set(sprite_cells)
    for c in range(legend_len):
        blocked.add((0, c))
        blocked.add((1, c))

    cells = []
    for r in range(1, h):
        for c in range(w):
            pos = (r, c)
            if pos in blocked:
                continue
            if any((nr, nc) in sprite_cells for nr, nc in _neighbors4(r, c)):
                continue
            cells.append(pos)
    return cells


def _pick_noise_cells(
    h: int,
    w: int,
    legend_len: int,
    sprite_cells: set[tuple[int, int]],
    count: int,
) -> list[tuple[int, int]] | None:
    candidates = _free_noise_cells(h, w, legend_len, sprite_cells)
    random.shuffle(candidates)
    chosen: list[tuple[int, int]] = []
    for cell in candidates:
        r, c = cell
        if any(abs(r - rr) + abs(c - cc) == 1 for rr, cc in chosen):
            continue
        chosen.append(cell)
        if len(chosen) == count:
            return chosen
    return None


def _semantic_summary(grid: np.ndarray) -> dict[str, object]:
    filtered = _noise_filtered(grid)
    removed_noise = int(np.count_nonzero((grid == GREY) & (filtered == BG)))
    comps = _find_components(filtered == GREY)
    lefts = sorted(int(coords[:, 1].min()) for coords in comps)
    tri_lefts = [left for left in lefts if _is_triangular(left)]
    non_tri_lefts = [left for left in lefts if not _is_triangular(left)]
    legend = _legend_colors(grid)
    return {
        "removed_noise": removed_noise,
        "legend_len": len(legend),
        "legend_distinct": len(set(legend)) == len(legend),
        "component_count": len(comps),
        "tri_lefts": tri_lefts,
        "non_tri_lefts": non_tri_lefts,
    }


def _generate_candidate() -> np.ndarray | None:
    h = random.randint(7, 10)
    w = random.randint(20, 29)
    band_top = h // 2 - 1

    legend_len = random.randint(3, 4)
    legend_colors = random.sample(LEGEND_PALETTE, legend_len)

    tri_candidates = [c for c in _triangular_columns(w - 1) if c <= w - 2]
    tri_options = []
    for size in range(legend_len + 1, min(len(tri_candidates), 6) + 1):
        tri_options.extend(_valid_left_subsets(tri_candidates, size))
    if not tri_options:
        return None
    target_lefts = list(random.choice(tri_options))

    occupied = set(target_lefts)
    non_tri_candidates = [
        c
        for c in range(w - 1)
        if c not in tri_candidates and all(abs(c - t) >= 3 for t in occupied)
    ]
    distractor_options = []
    max_distractors = min(3, len(non_tri_candidates))
    for size in range(1, max_distractors + 1):
        distractor_options.extend(_valid_left_subsets(non_tri_candidates, size))
    if not distractor_options:
        return None
    distractor_lefts = list(random.choice(distractor_options))

    grid = np.zeros((h, w), dtype=int)
    grid[0, :legend_len] = np.array(legend_colors, dtype=int)

    sprite_cells: set[tuple[int, int]] = set()
    for left in target_lefts + distractor_lefts:
        _place_sprite(grid, band_top, left)
        sprite_cells.update((r, c) for r in range(band_top, band_top + 2) for c in range(left, left + 2))

    noise_count = random.randint(2, min(7, max(2, h * w // 10)))
    noise_cells = _pick_noise_cells(h, w, legend_len, sprite_cells, noise_count)
    if noise_cells is None:
        return None
    for r, c in noise_cells:
        grid[r, c] = GREY

    summary = _semantic_summary(grid)
    if summary["removed_noise"] < 2:
        return None
    if not summary["legend_distinct"]:
        return None
    if len(summary["non_tri_lefts"]) < 1:
        return None
    if len(summary["tri_lefts"]) <= int(summary["legend_len"]):
        return None
    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
