from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
RED = 2
BLUE = 1
GREEN = 3
LEGEND_KEYS = np.array([RED, BLUE, GREEN], dtype=int)
FILL_COLORS = (4, 5, 6, 7, 8, 9)

GEN_ATTEMPTS = 400
UNIT_ATTEMPTS = 200
PLACEMENT_ATTEMPTS = 200


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _find_legend(grid: np.ndarray) -> tuple[dict[int, int], np.ndarray]:
    h, w = grid.shape
    candidates = [
        (0, 0),
        (0, w - 2),
        (h - 3, 0),
        (h - 3, w - 2),
    ]
    for r0, c0 in candidates:
        if r0 < 0 or c0 < 0:
            continue
        block = grid[r0 : r0 + 3, c0 : c0 + 2]
        if block.shape != (3, 2):
            continue
        if not np.array_equal(block[:, 0], LEGEND_KEYS):
            continue
        fills = [int(v) for v in block[:, 1]]
        if any(v == BG for v in fills) or len(set(fills)) != 3:
            continue
        mask = np.zeros_like(grid, dtype=bool)
        mask[r0 : r0 + 3, c0 : c0 + 2] = True
        mapping = {int(block[i, 0]): int(block[i, 1]) for i in range(3)}
        return mapping, mask
    raise AssertionError("expected a 3x2 legend in one corner")


def _same_color_component(
    grid: np.ndarray,
    start: tuple[int, int],
    color: int,
    blocked: np.ndarray,
    visited: np.ndarray,
) -> list[tuple[int, int]]:
    h, w = grid.shape
    q = deque([start])
    visited[start] = True
    comp = [start]
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if visited[nr, nc] or blocked[nr, nc] or int(grid[nr, nc]) != color:
                continue
            visited[nr, nc] = True
            q.append((nr, nc))
            comp.append((nr, nc))
    return comp


def _find_seed_cells(
    grid: np.ndarray, legend_mask: np.ndarray, wall_colors: set[int]
) -> list[tuple[int, int, int]]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    seeds: list[tuple[int, int, int]] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color not in wall_colors or legend_mask[r, c] or visited[r, c]:
                continue
            comp = _same_color_component(grid, (r, c), color, legend_mask, visited)
            if len(comp) == 1:
                seeds.append((r, c, color))
    return seeds


def _reachable_background_from_seed(
    grid: np.ndarray, seed: tuple[int, int]
) -> list[tuple[int, int]]:
    h, w = grid.shape
    q = deque([seed])
    seen = {seed}
    cells: list[tuple[int, int]] = []
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if (nr, nc) in seen or int(grid[nr, nc]) != BG:
                continue
            seen.add((nr, nc))
            q.append((nr, nc))
            cells.append((nr, nc))
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill enclosed same-color seed regions using a corner legend and seed-phased checkerboard growth.

    Input structure:
    - The background is black.
    - A corner 3x2 legend has three rows whose left cells are red, blue, and green, and whose right cells give the mapped fill colors.
    - Away from the legend are several non-touching growth units, each made of a one-pixel-wide closed wall and one interior seed pixel of the same color.

    Rule:
    - Read the legend as a mapping from wall/seed color to fill color.
    - For each singleton seed pixel, flood the black region reachable from that seed without crossing the surrounding wall.
    - Recolor only the reachable black cells whose row+column parity matches the seed's parity, while keeping the wall, the seed, and the legend unchanged.

    Key insight:
    - The corner legend determines which color each wall grows with, and the seed fixes the checkerboard phase so the interior is only partially filled.
    """
    g = np.asarray(grid, dtype=int)
    mapping, legend_mask = _find_legend(g)
    out = g.copy()

    seeds = _find_seed_cells(g, legend_mask, set(mapping))
    assert seeds, "expected at least one singleton seed"

    for r, c, color in seeds:
        fill_color = mapping[color]
        parity = (r + c) & 1
        for rr, cc in _reachable_background_from_seed(g, (r, c)):
            if ((rr + cc) & 1) == parity:
                out[rr, cc] = fill_color

    return out


def _mask_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if visited[r, c] or not mask[r, c]:
                continue
            q = deque([(r, c)])
            visited[r, c] = True
            comp = [(r, c)]
            while q:
                cr, cc = q.popleft()
                for nr, nc in _neighbors4(cr, cc):
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if visited[nr, nc] or not mask[nr, nc]:
                        continue
                    visited[nr, nc] = True
                    q.append((nr, nc))
                    comp.append((nr, nc))
            comps.append(comp)
    return comps


def _boundary_mask(region: np.ndarray) -> np.ndarray:
    h, w = region.shape
    wall = np.zeros_like(region, dtype=bool)
    for r, c in np.argwhere(region):
        rr = int(r)
        cc = int(c)
        for nr, nc in _neighbors4(rr, cc):
            if not (0 <= nr < h and 0 <= nc < w) or not region[nr, nc]:
                wall[rr, cc] = True
                break
    return wall


def _make_candidate_region(rng: random.Random) -> np.ndarray:
    ph = rng.randint(9, 13)
    pw = rng.randint(9, 13)
    top = [1] * (pw - 2)
    bottom = [ph - 2] * (pw - 2)
    applied = 0

    for _ in range(rng.randint(3, 6)):
        span = rng.randint(2, pw - 2)
        start = rng.randint(0, (pw - 2) - span)
        delta = rng.randint(1, 2)
        idxs = range(start, start + span)
        if rng.random() < 0.5:
            if all(top[i] + delta <= bottom[i] - 5 for i in idxs):
                for i in idxs:
                    top[i] += delta
                applied += 1
        else:
            if all(bottom[i] - delta >= top[i] + 5 for i in idxs):
                for i in idxs:
                    bottom[i] -= delta
                applied += 1

    if applied < 2:
        return np.zeros((1, 1), dtype=bool)

    region = np.zeros((ph, pw), dtype=bool)
    for idx, col in enumerate(range(1, pw - 1)):
        region[top[idx] : bottom[idx] + 1, col] = True

    region = np.rot90(region, rng.randrange(4))
    if rng.random() < 0.5:
        region = region[:, ::-1]
    if rng.random() < 0.5:
        region = region[::-1, :]
    return region.copy()


def _safe_seed_cells(wall: np.ndarray, comp: list[tuple[int, int]]) -> list[tuple[int, int]]:
    h, w = wall.shape
    safe: list[tuple[int, int]] = []
    for r, c in comp:
        if all(0 <= nr < h and 0 <= nc < w and not wall[nr, nc] for nr, nc in _neighbors4(r, c)):
            safe.append((r, c))
    return safe


def _make_unit_patch(rng: random.Random, color: int) -> np.ndarray:
    for _ in range(UNIT_ATTEMPTS):
        region = _make_candidate_region(rng)
        if region.size <= 1:
            continue
        wall = _boundary_mask(region)
        enclosed = [
            comp
            for comp in _mask_components(~wall)
            if not any(
                r == 0 or c == 0 or r == wall.shape[0] - 1 or c == wall.shape[1] - 1
                for r, c in comp
            )
        ]
        if len(enclosed) != 1 or len(enclosed[0]) < 10:
            continue

        safe = _safe_seed_cells(wall, enclosed[0])
        if not safe:
            continue

        seed = rng.choice(safe)
        patch = np.zeros(wall.shape, dtype=int)
        patch[wall] = color
        patch[seed] = color

        coords = np.argwhere(patch != BG)
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        cropped = patch[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()

        mapping = {RED: 4, BLUE: 5, GREEN: 6}
        try:
            out = main(_embed_unit_for_validation(cropped, mapping))
        except AssertionError:
            continue
        if np.array_equal(out, _embed_unit_for_validation(cropped, mapping)):
            continue
        return cropped
    raise RuntimeError("failed to generate a valid growth unit")


def _embed_unit_for_validation(unit: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    h, w = unit.shape
    canvas = np.zeros((max(10, h + 6), max(10, w + 8)), dtype=int)
    legend = np.column_stack([LEGEND_KEYS, [mapping[k] for k in LEGEND_KEYS]])
    canvas[:3, :2] = legend
    canvas[3 : 3 + h, 4 : 4 + w] = unit
    return canvas


def _legend_block(fill_colors: list[int]) -> np.ndarray:
    return np.column_stack([LEGEND_KEYS, np.array(fill_colors, dtype=int)])


def _reserve(mask: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(mask.shape[0], r0 + h + 1)
    cc1 = min(mask.shape[1], c0 + w + 1)
    mask[rr0:rr1, cc0:cc1] = True


def _place_patch(
    grid: np.ndarray, reserved: np.ndarray, patch: np.ndarray, rng: random.Random
) -> bool:
    h, w = grid.shape
    ph, pw = patch.shape
    for _ in range(PLACEMENT_ATTEMPTS):
        r0 = rng.randint(0, h - ph)
        c0 = rng.randint(0, w - pw)
        rr0 = max(0, r0 - 1)
        cc0 = max(0, c0 - 1)
        rr1 = min(h, r0 + ph + 1)
        cc1 = min(w, c0 + pw + 1)
        if reserved[rr0:rr1, cc0:cc1].any():
            continue
        grid_slice = grid[r0 : r0 + ph, c0 : c0 + pw]
        if np.any((patch != BG) & (grid_slice != BG)):
            continue
        grid_slice[patch != BG] = patch[patch != BG]
        reserved[rr0:rr1, cc0:cc1] = True
        return True
    return False


def _validate_generated_grid(grid: np.ndarray) -> bool:
    try:
        mapping, legend_mask = _find_legend(grid)
        seeds = _find_seed_cells(grid, legend_mask, set(mapping))
        if not seeds:
            return False
        for r, c, _color in seeds:
            region = _reachable_background_from_seed(grid, (r, c))
            if len(region) < 4:
                return False
            if any(
                rr == 0 or cc == 0 or rr == grid.shape[0] - 1 or cc == grid.shape[1] - 1
                for rr, cc in region
            ):
                return False
        out = main(grid)
    except AssertionError:
        return False
    return not np.array_equal(out, grid)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(20, 30)
    w = rng.randint(20, 30)
    grid = np.zeros((h, w), dtype=int)

    fill_colors = rng.sample(list(FILL_COLORS), 3)
    legend = _legend_block(fill_colors)
    corner = rng.choice(("TL", "TR", "BL", "BR"))
    r0 = 0 if "T" in corner else h - 3
    c0 = 0 if "L" in corner else w - 2
    grid[r0 : r0 + 3, c0 : c0 + 2] = legend

    reserved = np.zeros((h, w), dtype=bool)
    _reserve(reserved, r0, c0, 3, 2)

    n_units = rng.randint(2, 4)
    colors = list(rng.sample(list(LEGEND_KEYS), min(3, n_units)))
    while len(colors) < n_units:
        colors.append(rng.choice(list(LEGEND_KEYS)))
    rng.shuffle(colors)

    patches = []
    for color in colors:
        try:
            patches.append(_make_unit_patch(rng, color))
        except RuntimeError:
            return None
    patches.sort(key=lambda patch: patch.shape[0] * patch.shape[1], reverse=True)

    for patch in patches:
        if not _place_patch(grid, reserved, patch, rng):
            return None

    if not _validate_generated_grid(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
