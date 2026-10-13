from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
COLORS = tuple(range(1, 10))
GENERATION_ATTEMPTS = 300


def _components(mask: np.ndarray, connectivity: int = 8) -> list[np.ndarray]:
    height, width = map(int, mask.shape)
    seen = np.zeros((height, width), dtype=bool)
    if connectivity == 4:
        deltas = ((1, 0), (-1, 0), (0, 1), (0, -1))
    else:
        deltas = tuple(
            (dr, dc)
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
            if not (dr == 0 and dc == 0)
        )

    result: list[np.ndarray] = []
    for sr in range(height):
        for sc in range(width):
            if seen[sr, sc] or not bool(mask[sr, sc]):
                continue
            q = deque([(sr, sc)])
            seen[sr, sc] = True
            coords: list[tuple[int, int]] = []
            while q:
                r, c = q.popleft()
                coords.append((r, c))
                for dr, dc in deltas:
                    rr, cc = r + dr, c + dc
                    if not (0 <= rr < height and 0 <= cc < width):
                        continue
                    if seen[rr, cc] or not bool(mask[rr, cc]):
                        continue
                    seen[rr, cc] = True
                    q.append((rr, cc))
            result.append(np.array(coords, dtype=int))
    return result


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_mask_and_tile(grid: np.ndarray) -> tuple[tuple[int, int, int, int, int], np.ndarray]:
    comps = _components(grid != BLACK, connectivity=8)
    assert len(comps) == 2, "expected exactly two 8-connected foreground objects"

    mask_options: list[tuple[int, int, int, int, int]] = []
    tile_options: list[np.ndarray] = []
    for coords in comps:
        r0, c0, r1, c1 = _bbox(coords)
        values = grid[coords[:, 0], coords[:, 1]]
        colors = set(map(int, np.unique(values)))
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]

        if len(colors) == 1:
            mask_options.append((r0, c0, r1, c1, int(values[0])))

        full_bbox = int(coords.shape[0]) == int(sub.size)
        square = int(sub.shape[0]) == int(sub.shape[1])
        if len(colors) >= 2 and square and full_bbox:
            tile_options.append(np.array(sub, dtype=int, copy=True))

    assert len(mask_options) == 1, "expected one monochrome mask object"
    assert len(tile_options) == 1, "expected one filled multicolor square tile"
    return mask_options[0], tile_options[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand a monochrome control mask by replacing each foreground cell with a multicolor tile copy.

    Input structure:
    - The grid background is Black.
    - There are exactly two 8-connected foreground objects: one monochrome rectangular mask crop with Black holes, and one filled multicolor square tile.
    - The mask object's foreground pixels all share one color; Black cells inside its bounding box are blank template cells.

    Rule:
    - Find the mask bounding box and the separate square tile.
    - For every cell in the mask bounding box, stamp a full tile copy when that cell has the mask color.
    - For every Black cell in the mask bounding box, stamp an equally sized Black block.
    - The output size is mask_height * tile_size by mask_width * tile_size.

    Key insight:
    The monochrome mask is a binary block-scaling template: each foreground pixel inflates into one complete copy of the multicolor tile.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    mask_info, tile = _find_mask_and_tile(g)
    r0, c0, r1, c1, mask_color = mask_info
    mask = g[r0 : r1 + 1, c0 : c1 + 1] == int(mask_color)

    out = np.kron(mask.astype(int), tile)
    assert max(out.shape) <= 30, "output grid exceeds ARC size limit"
    return np.array(out, dtype=int)


def _make_mask(height: int, width: int) -> np.ndarray:
    mask = np.ones((height, width), dtype=bool)
    interior = [(r, c) for r in range(1, height - 1) for c in range(1, width - 1)]
    assert interior, "mask must have room for at least one Black hole"

    mode = random.choice(("scatter", "horizontal_run", "vertical_run", "checker"))
    holes: set[tuple[int, int]] = set()
    if mode == "scatter":
        max_count = max(1, min(len(interior), int(round(len(interior) * 0.35))))
        holes.update(random.sample(interior, random.randint(1, max_count)))
    elif mode == "horizontal_run":
        r = random.randrange(1, height - 1)
        c0 = random.randrange(1, width - 1)
        max_len = width - 1 - c0
        run_len = random.randint(1, max_len)
        holes.update((r, c) for c in range(c0, c0 + run_len))
    elif mode == "vertical_run":
        c = random.randrange(1, width - 1)
        r0 = random.randrange(1, height - 1)
        max_len = height - 1 - r0
        run_len = random.randint(1, max_len)
        holes.update((r, c) for r in range(r0, r0 + run_len))
    else:
        parity = random.randrange(2)
        candidates = [(r, c) for r, c in interior if (r + c) % 2 == parity]
        random.shuffle(candidates)
        limit = max(1, min(len(candidates), int(round(len(interior) * 0.4))))
        holes.update(candidates[: random.randint(1, limit)])

    for r, c in holes:
        mask[r, c] = False
    return mask


def _force_palette_use(tile: np.ndarray, palette: list[int]) -> None:
    positions = [(r, c) for r in range(tile.shape[0]) for c in range(tile.shape[1])]
    random.shuffle(positions)
    for color, (r, c) in zip(palette, positions):
        tile[r, c] = int(color)


def _make_tile(size: int, mask_color: int) -> np.ndarray:
    palette_source = [c for c in COLORS if c != int(mask_color)]
    color_count = random.randint(2, min(4, len(palette_source), size * size))
    palette = random.sample(palette_source, color_count)
    mode = random.choice(("random", "rows", "cols", "checker", "quadrants"))

    tile = np.zeros((size, size), dtype=int)
    if mode == "rows":
        for r in range(size):
            tile[r, :] = random.choice(palette)
    elif mode == "cols":
        for c in range(size):
            tile[:, c] = random.choice(palette)
    elif mode == "checker":
        a, b = random.sample(palette, 2)
        for r in range(size):
            for c in range(size):
                tile[r, c] = a if (r + c) % 2 == 0 else b
    elif mode == "quadrants":
        split_r = random.randint(1, size - 1)
        split_c = random.randint(1, size - 1)
        for r in range(size):
            for c in range(size):
                idx = (0 if r < split_r else 2) + (0 if c < split_c else 1)
                tile[r, c] = palette[idx % len(palette)]
    else:
        for r in range(size):
            for c in range(size):
                tile[r, c] = random.choice(palette)

    _force_palette_use(tile, palette)
    assert len(set(map(int, tile.ravel()))) >= 2
    return tile


def _choose_extent(min_extent: int) -> int:
    assert min_extent <= 30
    low = min_extent + 2 if min_extent + 2 <= 30 else min_extent
    return random.randint(low, 30)


def _gen_attempt() -> np.ndarray | None:
    tile_size = random.randint(2, 5)
    max_mask_side = 30 // tile_size
    if max_mask_side < 3:
        return None

    mask_h = random.randint(3, max_mask_side)
    mask_w = random.randint(3, max_mask_side)
    mask_color = random.choice(COLORS)
    mask = _make_mask(mask_h, mask_w)
    tile = _make_tile(tile_size, mask_color)
    gap = random.randint(1, 3)
    layout = random.choice(("tile_left", "tile_right", "tile_above", "tile_below"))

    if layout in ("tile_left", "tile_right"):
        min_h = max(mask_h, tile_size)
        min_w = mask_w + tile_size + gap
        if min_w > 30:
            return None
        height = _choose_extent(min_h)
        width = _choose_extent(min_w)

        base_c = random.randint(0, width - min_w)
        if layout == "tile_left":
            tile_c = base_c
            mask_c = base_c + tile_size + gap
        else:
            mask_c = base_c
            tile_c = base_c + mask_w + gap
        mask_r = random.randint(0, height - mask_h)
        tile_r = random.randint(0, height - tile_size)
    else:
        min_h = mask_h + tile_size + gap
        min_w = max(mask_w, tile_size)
        if min_h > 30:
            return None
        height = _choose_extent(min_h)
        width = _choose_extent(min_w)

        base_r = random.randint(0, height - min_h)
        if layout == "tile_above":
            tile_r = base_r
            mask_r = base_r + tile_size + gap
        else:
            mask_r = base_r
            tile_r = base_r + mask_h + gap
        mask_c = random.randint(0, width - mask_w)
        tile_c = random.randint(0, width - tile_size)

    grid = np.zeros((height, width), dtype=int)
    grid[mask_r : mask_r + mask_h, mask_c : mask_c + mask_w][mask] = int(mask_color)
    grid[tile_r : tile_r + tile_size, tile_c : tile_c + tile_size] = tile

    try:
        out = main(grid)
    except AssertionError:
        return None
    if max(grid.shape) > 30 or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GENERATION_ATTEMPTS} attempts")
