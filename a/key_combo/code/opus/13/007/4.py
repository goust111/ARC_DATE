from __future__ import annotations

from collections import deque
from itertools import combinations
import random

import numpy as np


TILE_SIZE = 5
OUTPUT_MULTIPLIER = 3
BG = 0
OUTLINE = 1
FG = 2
FILLED = 3
NOISE = 5
MAX_ATTEMPTS = 2000


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _erase_noise(tile: np.ndarray) -> np.ndarray:
    cleaned = np.array(tile, dtype=int, copy=True)
    cleaned[cleaned == NOISE] = BG
    return cleaned


def _orient_tile(tile: np.ndarray, tile_r: int, tile_c: int) -> np.ndarray:
    part = tile
    if tile_c % 2 == 1:
        part = np.fliplr(part)
    if tile_r % 2 == 1:
        part = np.flipud(part)
    return part


def _tile_wallpaper(tile: np.ndarray) -> np.ndarray:
    rows = []
    for tile_r in range(OUTPUT_MULTIPLIER):
        row = [_orient_tile(tile, tile_r, tile_c) for tile_c in range(OUTPUT_MULTIPLIER)]
        rows.append(np.concatenate(row, axis=1))
    return np.concatenate(rows, axis=0)


def _flood_exterior_background(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    exterior = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if grid[r, c] == BG and not exterior[r, c]:
                exterior[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if grid[r, c] == BG and not exterior[r, c]:
                exterior[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if _in_bounds(grid.shape, nr, nc) and grid[nr, nc] == BG and not exterior[nr, nc]:
                exterior[nr, nc] = True
                q.append((nr, nc))
    return exterior


def _component_sizes(mask: np.ndarray) -> list[int]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    sizes: list[int] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            size = 0
            while q:
                cr, cc = q.popleft()
                size += 1
                for nr, nc in _neighbors4(cr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            sizes.append(size)
    return sizes


def _closed_regions_for_tile(tile: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tiled = _tile_wallpaper(tile)
    exterior = _flood_exterior_background(tiled)
    holes = (tiled == BG) & ~exterior
    return tiled, holes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase edge noise, mirror-tile the cleaned tile, then fill emergent closed background regions and outline the exterior.

    Input structure:
    - The input is a 5x5 tile with black background, a 4-connected red foreground shape, and 1-2 gray noise pixels on the tile edge.
    - The gray pixels are 4-neighbors of the red shape and can change how mirrored copies connect across tile boundaries.
    - No closed region needs to exist inside the single tile; the important structure appears only after tiling.

    Rule:
    - Recolor every gray(5) pixel to black(0).
    - Build a 3x3 output wallpaper from the cleaned tile: odd tile columns are left-right mirrored, odd tile rows are top-bottom mirrored, and odd-odd positions are mirrored both ways.
    - On the 15x15 tiled result, flood-fill black background from the outer border with 4-connectivity; border-reachable background stays open and unreachable background is closed.
    - Fill every closed background pixel green(3), then recolor every open background pixel that 8-neighbors red or green foreground to blue(1). Red pixels stay red.

    Key insight:
    - The gray edge pixels matter only because they alter cross-tile connections: after removing them, the mirrored seams create or destroy larger loops that determine which background areas become enclosed.
    """
    tile = np.asarray(grid, dtype=int)
    assert tile.shape == (TILE_SIZE, TILE_SIZE), f"expected a {TILE_SIZE}x{TILE_SIZE} tile"
    assert set(np.unique(tile)).issubset({BG, FG, NOISE}), "expected only black, red, and gray"

    cleaned = _erase_noise(tile)
    tiled, holes = _closed_regions_for_tile(cleaned)
    solid = (tiled == FG) | holes
    open_background = (tiled == BG) & ~holes

    outline = np.zeros_like(open_background)
    for r, c in np.argwhere(open_background):
        if any(_in_bounds(tiled.shape, nr, nc) and solid[nr, nc] for nr, nc in _neighbors8(int(r), int(c))):
            outline[r, c] = True

    out = tiled.copy()
    out[holes] = FILLED
    out[outline] = OUTLINE
    return out


def _grow_connected_red_shape(target_cells: int) -> np.ndarray:
    tile = np.zeros((TILE_SIZE, TILE_SIZE), dtype=int)
    start = (random.randrange(TILE_SIZE), random.randrange(TILE_SIZE))
    cells = {start}

    while len(cells) < target_cells:
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if _in_bounds(tile.shape, nr, nc) and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            return tile

        unique_frontier = list(dict.fromkeys(frontier))

        def score(pos: tuple[int, int]) -> tuple[int, int, float]:
            r, c = pos
            support = sum((nr, nc) in cells for nr, nc in _neighbors4(r, c))
            edge_bonus = int(r in (0, TILE_SIZE - 1) or c in (0, TILE_SIZE - 1))
            return (-support, -edge_bonus, random.random())

        unique_frontier.sort(key=score)
        pick = random.choice(unique_frontier[: min(4, len(unique_frontier))])
        cells.add(pick)

    for r, c in cells:
        tile[r, c] = FG
    return tile


def _edge_red_count(tile: np.ndarray) -> int:
    count = 0
    for r in range(TILE_SIZE):
        for c in range(TILE_SIZE):
            if tile[r, c] == FG and (r in (0, TILE_SIZE - 1) or c in (0, TILE_SIZE - 1)):
                count += 1
    return count


def _noise_candidates(tile: np.ndarray) -> list[tuple[int, int]]:
    candidates = []
    for r in range(TILE_SIZE):
        for c in range(TILE_SIZE):
            if tile[r, c] != BG:
                continue
            if r not in (0, TILE_SIZE - 1) and c not in (0, TILE_SIZE - 1):
                continue
            if any(_in_bounds(tile.shape, nr, nc) and tile[nr, nc] == FG for nr, nc in _neighbors4(r, c)):
                candidates.append((r, c))
    random.shuffle(candidates)
    return candidates


def _valid_clean_tile(tile: np.ndarray) -> bool:
    red_count = int(np.sum(tile == FG))
    if not 8 <= red_count <= 14:
        return False
    if _edge_red_count(tile) < 2:
        return False
    if len(_component_sizes(tile == FG)) != 1:
        return False
    tiled, holes = _closed_regions_for_tile(tile)
    hole_sizes = _component_sizes(holes)
    if not hole_sizes:
        return False
    if sum(hole_sizes) > 32:
        return False
    if len(hole_sizes) > 8:
        return False
    if int(np.sum((main(tile) == OUTLINE))) < 8:
        return False
    return bool(np.any(tiled == FG))


def _choose_noise(tile: np.ndarray) -> list[tuple[int, int]] | None:
    _, clean_holes = _closed_regions_for_tile(tile)
    candidates = _noise_candidates(tile)
    if not candidates:
        return None

    desired_noise_count = random.randint(1, 2)
    if len(candidates) < desired_noise_count:
        return None

    combos = list(combinations(candidates, desired_noise_count))
    random.shuffle(combos)
    for combo in combos:
        noisy = tile.copy()
        for r, c in combo:
            noisy[r, c] = NOISE
        _, noisy_holes = _closed_regions_for_tile(noisy)
        if np.array_equal(noisy_holes, clean_holes):
            continue
        return list(combo)
    return None


def _generate_attempt() -> np.ndarray | None:
    target_cells = random.randint(8, 14)
    clean_tile = _grow_connected_red_shape(target_cells)
    if not _valid_clean_tile(clean_tile):
        return None

    noise_cells = _choose_noise(clean_tile)
    if noise_cells is None:
        return None

    tile = clean_tile.copy()
    for r, c in noise_cells:
        tile[r, c] = NOISE

    out = main(tile)
    if out.shape != (TILE_SIZE * OUTPUT_MULTIPLIER, TILE_SIZE * OUTPUT_MULTIPLIER):
        return None
    if not np.any(out == FILLED):
        return None
    if not np.any(out == OUTLINE):
        return None
    return tile


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        tile = _generate_attempt()
        if tile is not None:
            return tile
    raise RuntimeError(f"generate_input failed after {MAX_ATTEMPTS} attempts")
