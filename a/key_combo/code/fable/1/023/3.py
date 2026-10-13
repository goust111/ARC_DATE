import math
import random
from typing import Iterable

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
DIM_CHOICES = tuple(range(2, 7))
MAX_GENERATION_ATTEMPTS = 600
MAX_SHAPE_ATTEMPTS = 200


def neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _divisors(n: int) -> list[int]:
    return [d for d in range(1, n + 1) if n % d == 0]


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(
        mask[0, :].any()
        and mask[-1, :].any()
        and mask[:, 0].any()
        and mask[:, -1].any()
    )


def _render_input(
    layout_mask: np.ndarray,
    tile_mask: np.ndarray,
    colors: np.ndarray,
    *,
    bg: int = BG,
) -> np.ndarray:
    layout_h, layout_w = layout_mask.shape
    tile_h, tile_w = tile_mask.shape
    out = np.full((layout_h * tile_h, layout_w * tile_w), bg, dtype=int)
    for ar, ac in np.argwhere(layout_mask):
        color = int(colors[ar, ac])
        block = np.full(tile_mask.shape, bg, dtype=int)
        block[tile_mask] = color
        r0 = int(ar) * tile_h
        c0 = int(ac) * tile_w
        out[r0 : r0 + tile_h, c0 : c0 + tile_w] = block
    return out


def _render_swapped(
    layout_mask: np.ndarray,
    tile_mask: np.ndarray,
    colors: np.ndarray,
    *,
    bg: int = BG,
) -> np.ndarray:
    layout_h, layout_w = layout_mask.shape
    tile_h, tile_w = tile_mask.shape
    colored_layout = np.full(layout_mask.shape, bg, dtype=int)
    colored_layout[layout_mask] = colors[layout_mask]
    out = np.full((tile_h * layout_h, tile_w * layout_w), bg, dtype=int)
    for br, bc in np.argwhere(tile_mask):
        r0 = int(br) * layout_h
        c0 = int(bc) * layout_w
        out[r0 : r0 + layout_h, c0 : c0 + layout_w] = colored_layout
    return out


def _extract_candidate(grid: np.ndarray, tile_h: int, tile_w: int, *, bg: int = BG):
    H, W = grid.shape
    if H % tile_h != 0 or W % tile_w != 0:
        return None

    layout_h = H // tile_h
    layout_w = W // tile_w
    if layout_h * layout_w <= 1 or tile_h * tile_w <= 1:
        return None

    layout_mask = np.zeros((layout_h, layout_w), dtype=bool)
    colors = np.full((layout_h, layout_w), bg, dtype=int)
    tile_mask = None

    for ar in range(layout_h):
        for ac in range(layout_w):
            block = grid[ar * tile_h : (ar + 1) * tile_h, ac * tile_w : (ac + 1) * tile_w]
            fg = block != bg
            if not fg.any():
                continue

            vals = np.unique(block[fg])
            if len(vals) != 1:
                return None

            if tile_mask is None:
                tile_mask = fg.copy()
            elif not np.array_equal(tile_mask, fg):
                return None

            layout_mask[ar, ac] = True
            colors[ar, ac] = int(vals[0])

    if tile_mask is None or not layout_mask.any():
        return None
    if not _touches_all_sides(layout_mask) or not _touches_all_sides(tile_mask):
        return None
    if layout_mask.shape == tile_mask.shape and np.array_equal(layout_mask, tile_mask):
        return None

    return {
        "layout_mask": layout_mask,
        "tile_mask": tile_mask,
        "colors": colors,
        "layout_h": layout_h,
        "layout_w": layout_w,
        "tile_h": tile_h,
        "tile_w": tile_w,
    }


def _find_candidates(grid: np.ndarray, *, bg: int = BG) -> list[dict]:
    H, W = grid.shape
    candidates = []
    for tile_h in _divisors(H):
        for tile_w in _divisors(W):
            cand = _extract_candidate(grid, tile_h, tile_w, bg=bg)
            if cand is not None:
                candidates.append(cand)
    return candidates


def _recover_unique(grid: np.ndarray, *, bg: int = BG) -> dict:
    candidates = _find_candidates(grid, bg=bg)
    if len(candidates) != 1:
        dims = sorted(
            (cand["layout_h"], cand["layout_w"], cand["tile_h"], cand["tile_w"])
            for cand in candidates
        )
        raise AssertionError(f"expected exactly one feasible partition, found {len(candidates)}: {dims}")
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap the outer occupancy layout with the inner monochrome tile, and sink the outer colors into per-pixel paint inside the new inner copies.

    Input structure:
    - The background is black.
    - There is exactly one nontrivial coarse partition of the grid into equal blocks.
    - Under that partition, each coarse block is either all black or the same tile shape, filled with one solid color per occupied block.

    Rule:
    - Recover the unique block size, the coarse occupancy shape A, the shared inner tile shape B, and the color matrix attached to A's occupied cells.
    - Produce a new expansion where B becomes the outer block layout.
    - Inside every occupied B block, draw a copy of A and color each foreground cell of that copy by the original color matrix entry at the same A position.

    Key insight:
    - The same grid simultaneously encodes a repeated binary tile and a coarse occupancy matrix; the solve step is to factor that encoding uniquely, then swap the two layers while moving color information from block-level labels down to pixel-level paint.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    info = _recover_unique(g, bg=BG)
    return _render_swapped(info["layout_mask"], info["tile_mask"], info["colors"], bg=BG)


def _shape_area_bounds(h: int, w: int) -> tuple[int, int]:
    cells = h * w
    lo = max(3, int(math.ceil(0.35 * cells)))
    hi = cells - 1
    if cells >= 6:
        hi = min(hi, int(math.floor(0.72 * cells)))
    if hi < lo:
        hi = cells - 1
    return lo, hi


def _random_connected_shape(h: int, w: int) -> np.ndarray | None:
    lo, hi = _shape_area_bounds(h, w)
    for _ in range(MAX_SHAPE_ATTEMPTS):
        target = random.randint(lo, hi)
        start = (random.randrange(h), random.randrange(w))
        cells = {start}
        frontier = set()
        for nr, nc in neighbors4(*start):
            if 0 <= nr < h and 0 <= nc < w:
                frontier.add((nr, nc))

        while len(cells) < target and frontier:
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            cells.add(cell)
            for nr, nc in neighbors4(*cell):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.add((nr, nc))

        if len(cells) != target:
            continue

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        if _touches_all_sides(mask):
            return mask
    return None


def _random_dimensions() -> tuple[int, int, int, int] | None:
    for _ in range(200):
        layout_h = random.choice(DIM_CHOICES)
        layout_w = random.choice(DIM_CHOICES)
        tile_h = random.choice(DIM_CHOICES)
        tile_w = random.choice(DIM_CHOICES)
        if layout_h * tile_h > 30 or layout_w * tile_w > 30:
            continue
        if layout_h * tile_h < 6 or layout_w * tile_w < 6:
            continue
        return layout_h, layout_w, tile_h, tile_w
    return None


def _random_colors(layout_mask: np.ndarray) -> np.ndarray:
    coords = [tuple(x) for x in np.argwhere(layout_mask)]
    palette_size = min(len(coords), random.randint(2, 5), len(PALETTE))
    palette = random.sample(PALETTE, palette_size)
    phase = random.randrange(palette_size)
    a = random.choice((1, 2, 3))
    b = random.choice((1, 2, 3))

    colors = np.zeros(layout_mask.shape, dtype=int)
    used = []
    for r, c in coords:
        color = palette[(a * r + b * c + phase) % palette_size]
        colors[r, c] = color
        used.append(color)

    if len(set(used)) < 2:
        random.shuffle(coords)
        for idx, (r, c) in enumerate(coords):
            colors[r, c] = palette[idx % palette_size]
    return colors


def _gen_attempt() -> np.ndarray | None:
    dims = _random_dimensions()
    if dims is None:
        return None
    layout_h, layout_w, tile_h, tile_w = dims

    layout_mask = _random_connected_shape(layout_h, layout_w)
    tile_mask = _random_connected_shape(tile_h, tile_w)
    if layout_mask is None or tile_mask is None:
        return None

    if layout_mask.shape == tile_mask.shape and np.array_equal(layout_mask, tile_mask):
        return None

    colors = _random_colors(layout_mask)
    if len(set(map(int, colors[layout_mask].tolist()))) < 2:
        return None

    inp = _render_input(layout_mask, tile_mask, colors, bg=BG)
    expected = _render_swapped(layout_mask, tile_mask, colors, bg=BG)
    if np.array_equal(inp, expected):
        return None

    try:
        info = _recover_unique(inp, bg=BG)
    except AssertionError:
        return None

    if not np.array_equal(info["layout_mask"], layout_mask):
        return None
    if not np.array_equal(info["tile_mask"], tile_mask):
        return None
    if not np.array_equal(info["colors"], colors):
        return None

    try:
        out = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(out, expected):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
